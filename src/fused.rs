//! Thin Rust consumer of the fused jax-exported prove core.
//!
//! The export pipeline lowers the whole zk `ASForR1CSNark` prove to one
//! StableHLO `.mlirbc` — every commitment, fold, and Fiat-Shamir sponge in a
//! single PJRT call — and the **general** core lifts the rest to runtime: the committer key
//! `bases_h = generators[:rows] ‖ hiding`, the HP placeholder identity `id_pt`,
//! AND the assignment + replayed randomness are all runtime inputs
//! (`artifacts/prove_zk_general.mlirbc`). This module loads that core, runs it
//! once on the shared GPU client, and re-serializes the on-device outputs to the
//! exact `acc.instance ‖ acc.witness ‖ proof` byte stream the per-MSM
//! `GpuBackend` path produces — replacing many MSM dispatches with one fused
//! call (the bellman-zorch "Rust = thin consumer" model).
//!
//! The single-input prove core is now the **general** prover: the
//! assignment + all replayed randomness are runtime PJRT inputs ([`ZkProveInputs`]),
//! so one lowered `prove_zk_general.mlirbc` proves any seed (the IVC fold core
//! still bakes its inputs — see [`prove_fold_zk_fused`]). The consumer
//! is generic over the Pasta cycle curve `C` ([`crate::gpu::PastaCurve`]): the
//! affine inputs carry `C::G1_AFFINE`, the `fr` runtime inputs `C::SF`, the
//! `u8_batch` sponge inputs `C::BF`, and the on-device leaves parse with
//! `C::Params`, so the same code serves the Pallas single-step and the Vesta
//! recursion step. The Python export selects the curve to lower the matching core.
//!
//! ## Output contract (pinned via `jax.tree_flatten` of the lowered core)
//!
//! `Session::run(.., num_outputs = 16)` returns the pytree leaves of the core's
//! return in this order (`G1` = 64B `x‖y`, `Fr` = 32B LE):
//!
//! | idx | leaf | idx | leaf |
//! |-----|------|-----|------|
//! | 0 | combined_input `Vec<Fr>` | 8 | comm_r_c `G` |
//! | 1 | combined_comm_a `G` | 9 | hp.instance `[G;3]` |
//! | 2 | combined_comm_b `G` | 10 | hp.a_open `Vec<Fr>` |
//! | 3 | combined_comm_c `G` | 11 | hp.b_open `Vec<Fr>` |
//! | 4 | combined_blinded_witness `Vec<Fr>` | 12 | hp.rand `[Fr;3]` |
//! | 5 | combined_sigmas `[Fr;3]` | 13 | hp.low `Vec<G>` |
//! | 6 | comm_r_a `G` | 14 | hp.high `Vec<G>` |
//! | 7 | comm_r_b `G` | 15 | hp.hiding_comms `[G;3]` |
//!
//! The serialization below emits these (plus the one host-side `r1cs_r_input`)
//! in the field order of `r1cs_nark_as::{AccumulatorInstance, AccumulatorWitness,
//! Proof}` and `hp_as::{InputInstance, InputWitness, Proof}` — i.e. exactly what
//! their derived `CanonicalSerialize` would emit (verified against the Python
//! serialize seam `python/accumulation_zorch/r1cs_nark_as.py:prove_zk`, which
//! was byte-matched to arkworks on GPU).

use ark_ec::models::ModelParameters;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;

use crate::gpu::PastaCurve;
use crate::wire;

/// The curve's affine point type (`GroupAffine<C::Params>`) and scalar field.
type Affine<C> = GroupAffine<<C as PastaCurve>::Params>;
type Fr<C> = <<C as PastaCurve>::Params as ModelParameters>::ScalarField;

/// Number of pytree leaves the fused core returns (see the module table).
const NUM_OUTPUTS: usize = 16;

/// The three serialized components of one accumulation prove output, matching
/// `tests/gpu_prove_byte_match.rs`'s `acc.instance ‖ acc.witness ‖ proof`.
pub struct FusedProveBytes {
    /// `AccumulatorInstance` CanonicalSerialize bytes.
    pub acc_instance: Vec<u8>,
    /// `AccumulatorWitness` CanonicalSerialize bytes (zk: `Some` randomness).
    pub acc_witness: Vec<u8>,
    /// `Proof` CanonicalSerialize bytes (zk: `Some` `ProofRandomness`).
    pub proof: Vec<u8>,
}

impl FusedProveBytes {
    /// Concatenation `acc.instance ‖ acc.witness ‖ proof` — the full byte stream
    /// the `gpu_prove_byte_match` harness compares.
    pub fn concat(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(self.acc_instance.len() + self.acc_witness.len() + self.proof.len());
        out.extend_from_slice(&self.acc_instance);
        out.extend_from_slice(&self.acc_witness);
        out.extend_from_slice(&self.proof);
        out
    }
}

/// Reads one scalar from each 32-byte little-endian chunk of `bytes` (the wire
/// scalar form — standard, non-Montgomery; canonical, so the mod-order read is
/// exact).
fn parse_scalars<C: PastaCurve>(bytes: &[u8]) -> Vec<Fr<C>> {
    assert_eq!(bytes.len() % wire::SF_BYTES, 0, "Fr leaf not a multiple of 32 bytes");
    bytes.chunks(wire::SF_BYTES).map(Fr::<C>::from_le_bytes_mod_order).collect()
}

/// Reads one affine point from each 64-byte (`x‖y`) chunk of `bytes`; all-zero is
/// the identity (`wire::g1_from_bytes`).
fn parse_points<C: PastaCurve>(bytes: &[u8]) -> Vec<Affine<C>>
where
    <C::Params as ModelParameters>::BaseField: PrimeField,
{
    assert_eq!(bytes.len() % wire::G1_BYTES, 0, "G1 leaf not a multiple of 64 bytes");
    bytes.chunks(wire::G1_BYTES).map(wire::g1_from_bytes::<C::Params>).collect()
}

/// Reads exactly one affine point from a 64-byte leaf (a rank-0 `G` output).
fn parse_point<C: PastaCurve>(bytes: &[u8]) -> Affine<C>
where
    <C::Params as ModelParameters>::BaseField: PrimeField,
{
    wire::g1_from_bytes::<C::Params>(bytes)
}

/// Compile `mlirbc` on the shared (leaked) PJRT client and leak the executable —
/// the client must outlive all threads (dropping it `dlclose`s the plugin mid
/// CUDA/absl teardown → SIGSEGV; see `gpu.rs`). Split from [`run_fused`] so a
/// benchmark can compile once and run the core many times.
pub fn load_fused(mlirbc: &[u8]) -> &'static zkx_pjrt::Executable {
    // Safety: compiled on the one process client, which `crate::gpu::session()`
    // owns and never drops.
    Box::leak(Box::new(unsafe { crate::gpu::session().compile(mlirbc) }))
}

/// The runtime inputs of the **general** zk prove core: the assignment
/// (`r1cs_input` / `witness`) and all replayed randomness lifted to PJRT inputs, in
/// the lowered core's argument order after `(bases_h, id_pt)`. The two `u8_batch`
/// fq sponge inputs are derived here from `r1cs_input` / `r1cs_r_input`
/// ([`wire::u8_batch_field_array`]), so they are not fields. `r1cs_r_input` is also
/// serialized into the proof bytes (the AS `ProofRandomness.r1cs_r_input`).
pub struct ZkProveInputs<C: PastaCurve> {
    /// The public input `r1cs_input` (`fr`).
    pub r1cs_input: Vec<Fr<C>>,
    /// The witness (`fr`).
    pub witness: Vec<Fr<C>>,
    /// The NARK witness blinders `r` (`fr`, witness-length).
    pub nark_r: Vec<Fr<C>>,
    /// The 8 NARK sigma blinders `[a, b, c, r_a, r_b, r_c, 1, 2]` (`fr`).
    pub nark_blinders: Vec<Fr<C>>,
    /// The AS proof randomness `r1cs_r_input = vec![as_r1cs_r_input; input_len]`.
    pub r1cs_r_input: Vec<Fr<C>>,
    /// The AS proof randomness `r1cs_r_witness = vec![as_r1cs_r_witness; witness_len]`.
    pub r1cs_r_witness: Vec<Fr<C>>,
    /// The 3 AS commitment blinders `[as_rand_1, as_rand_2, as_rand_3]` (`fr`).
    pub as_rand: Vec<Fr<C>>,
    /// The 5 HP randomness values `[hiding_a, hiding_b, hr1, hr2, hr3]` (`fr`).
    pub hp_rand: Vec<Fr<C>>,
}

/// Run the general fused zk prove core `mlirbc` once on the GPU with the affine
/// inputs `bases_h` (committer key `generators[:rows] ‖ hiding`) and `id_pt` (HP
/// placeholder identity) plus the assignment + replayed randomness ([`ZkProveInputs`]),
/// then serialize the on-device outputs to `acc.instance ‖ acc.witness ‖ proof`.
///
/// The general prover lifts the witness + randomness to runtime inputs, so one lowered
/// `prove_zk_general.mlirbc` proves any seed; `inputs.r1cs_r_input` is both a
/// runtime input and the AS `ProofRandomness.r1cs_r_input` serialized into the
/// proof bytes (not an output leaf).
///
/// One PJRT call replaces the per-MSM `GpuBackend` dispatch. The curve `C` is the
/// one the core was exported for (its affine inputs carry `C::G1_AFFINE`).
pub fn prove_fused<C: PastaCurve>(
    mlirbc: &[u8],
    bases_h: &[Affine<C>],
    id_pt: &Affine<C>,
    inputs: &ZkProveInputs<C>,
) -> FusedProveBytes
where
    <C::Params as ModelParameters>::BaseField: PrimeField,
{
    run_fused::<C>(load_fused(mlirbc), bases_h, id_pt, inputs)
}

/// Run an already-loaded general fused core `exe` once (one PJRT call) and
/// serialize the outputs. Split from [`load_fused`] so a benchmark can reuse one
/// compiled executable across many runs (the heavy part is the run, not the
/// compile). Feeds the 12 runtime inputs in the lowered core's argument order; the
/// two `u8_batch` sponge inputs are pre-encoded here ([`wire::u8_batch_field_array`])
/// as `C::BF` (base-field) buffers.
pub fn run_fused<C: PastaCurve>(
    exe: &zkx_pjrt::Executable,
    bases_h: &[Affine<C>],
    id_pt: &Affine<C>,
    inputs: &ZkProveInputs<C>,
) -> FusedProveBytes
where
    <C::Params as ModelParameters>::BaseField: PrimeField,
{
    let bases_bytes = wire::g1_array_to_bytes(bases_h);
    let id_bytes = wire::g1_to_bytes(id_pt);
    let in_b = wire::scalars_to_bytes::<C::Params>(&inputs.r1cs_input);
    let wit_b = wire::scalars_to_bytes::<C::Params>(&inputs.witness);
    let r_b = wire::scalars_to_bytes::<C::Params>(&inputs.nark_r);
    let bl_b = wire::scalars_to_bytes::<C::Params>(&inputs.nark_blinders);
    let rin_b = wire::scalars_to_bytes::<C::Params>(&inputs.r1cs_r_input);
    let rwit_b = wire::scalars_to_bytes::<C::Params>(&inputs.r1cs_r_witness);
    let asr_b = wire::scalars_to_bytes::<C::Params>(&inputs.as_rand);
    let hpr_b = wire::scalars_to_bytes::<C::Params>(&inputs.hp_rand);
    let in_u8b = wire::u8_batch_field_array::<C::Params>(&inputs.r1cs_input);
    let rin_u8b = wire::u8_batch_field_array::<C::Params>(&inputs.r1cs_r_input);
    let n_fe = |b: &[u8]| -> i64 { (b.len() / wire::SF_BYTES) as i64 };
    // 12 inputs, in the general zk core's `_core(bases_h, id_pt, in, wit, r,
    // blinders, r_in, r_wit, as_rand, hp_rand, in_u8b, r_in_u8b)` argument order:
    // the two affine inputs, the eight `fr` runtime arrays, then the
    // two pre-encoded `u8_batch` base-field (`fq`) sponge arrays.
    let args = [
        (bases_bytes.as_slice(), vec![bases_h.len() as i64], C::G1_AFFINE),
        (id_bytes.as_slice(), vec![1i64], C::G1_AFFINE),
        (in_b.as_slice(), vec![inputs.r1cs_input.len() as i64], C::SF),
        (wit_b.as_slice(), vec![inputs.witness.len() as i64], C::SF),
        (r_b.as_slice(), vec![inputs.nark_r.len() as i64], C::SF),
        (bl_b.as_slice(), vec![inputs.nark_blinders.len() as i64], C::SF),
        (rin_b.as_slice(), vec![inputs.r1cs_r_input.len() as i64], C::SF),
        (rwit_b.as_slice(), vec![inputs.r1cs_r_witness.len() as i64], C::SF),
        (asr_b.as_slice(), vec![inputs.as_rand.len() as i64], C::SF),
        (hpr_b.as_slice(), vec![inputs.hp_rand.len() as i64], C::SF),
        (in_u8b.as_slice(), vec![n_fe(&in_u8b)], C::BF),
        (rin_u8b.as_slice(), vec![n_fe(&rin_u8b)], C::BF),
    ];
    // Safety: `exe` was compiled by this same client; the 12 input shapes/types
    // match the lowered general zk core's argument list (see the comment above).
    let out = unsafe { crate::gpu::session().run(exe, &args, NUM_OUTPUTS) };
    assert_eq!(out.len(), NUM_OUTPUTS, "fused core returned {} leaves, expected {NUM_OUTPUTS}", out.len());

    serialize_prove_outputs::<C>(&out, &inputs.r1cs_r_input)
}

/// Parse the 16 pytree leaves of a fused zk prove/fold core's output (the module
/// table order at the top of this file) and serialize them to the
/// `acc.instance ‖ acc.witness ‖ proof` byte stream. Shared by the single-input
/// prove ([`run_fused`]) and the IVC fold ([`prove_fold_zk_fused`]): both cores
/// return the identical `AccumulatorInstance ‖ AccumulatorWitness ‖ Proof` pytree,
/// so only their runtime inputs differ (the fold adds the old accumulator's
/// commitments). `r1cs_r_input` is the one serialized field that is not an output
/// leaf — the AS `ProofRandomness.r1cs_r_input`, supplied by the caller (a runtime
/// input on the general prove path; a baked constant on the fold path).
fn serialize_prove_outputs<C: PastaCurve>(out: &[Vec<u8>], r1cs_r_input: &[Fr<C>]) -> FusedProveBytes
where
    <C::Params as ModelParameters>::BaseField: PrimeField,
{
    // Parse the 16 leaves (module table order).
    let combined_input = parse_scalars::<C>(&out[0]);
    let comm_a = parse_point::<C>(&out[1]);
    let comm_b = parse_point::<C>(&out[2]);
    let comm_c = parse_point::<C>(&out[3]);
    let blinded_witness = parse_scalars::<C>(&out[4]);
    let sigmas = parse_scalars::<C>(&out[5]);
    let comm_r_a = parse_point::<C>(&out[6]);
    let comm_r_b = parse_point::<C>(&out[7]);
    let comm_r_c = parse_point::<C>(&out[8]);
    let hp_instance = parse_points::<C>(&out[9]); // comm_1, comm_2, comm_3
    let hp_a_vec = parse_scalars::<C>(&out[10]);
    let hp_b_vec = parse_scalars::<C>(&out[11]);
    let hp_rand = parse_scalars::<C>(&out[12]); // rand_1, rand_2, rand_3
    let low = parse_points::<C>(&out[13]);
    let high = parse_points::<C>(&out[14]);
    let hiding_comms = parse_points::<C>(&out[15]); // comm_1, comm_2, comm_3

    // --- AccumulatorInstance: r1cs_input, comm_a, comm_b, comm_c, hp_instance
    //     (src/r1cs_nark_as/data_structures.rs AccumulatorInstance;
    //      hp_instance = hp_as InputInstance{comm_1,comm_2,comm_3}).
    let mut acc_instance = Vec::new();
    combined_input.serialize(&mut acc_instance).unwrap(); // Vec<Fr>: u64 len + 32B each
    comm_a.serialize(&mut acc_instance).unwrap();
    comm_b.serialize(&mut acc_instance).unwrap();
    comm_c.serialize(&mut acc_instance).unwrap();
    serialize_each(&hp_instance, &mut acc_instance);

    // --- AccumulatorWitness: r1cs_blinded_witness, hp_witness{a_vec,b_vec,
    //     Some(rand_1,2,3)}, Some(sigma_a,b,c).
    let mut acc_witness = Vec::new();
    blinded_witness.serialize(&mut acc_witness).unwrap();
    hp_a_vec.serialize(&mut acc_witness).unwrap();
    hp_b_vec.serialize(&mut acc_witness).unwrap();
    acc_witness.push(1); // Some InputWitnessRandomness
    serialize_each(&hp_rand, &mut acc_witness);
    acc_witness.push(1); // Some AccumulatorWitnessRandomness
    serialize_each(&sigmas, &mut acc_witness);

    // --- Proof: hp_proof{ product_poly_comm{low,high}, Some(hiding_comms) },
    //     Some(ProofRandomness{ r1cs_r_input, comm_r_a, comm_r_b, comm_r_c }).
    let mut proof = Vec::new();
    low.serialize(&mut proof).unwrap(); // Vec<G>
    high.serialize(&mut proof).unwrap(); // Vec<G>
    proof.push(1); // Some ProofHidingCommitments
    serialize_each(&hiding_comms, &mut proof);
    proof.push(1); // Some ProofRandomness
    r1cs_r_input.to_vec().serialize(&mut proof).unwrap(); // Vec<Fr>
    comm_r_a.serialize(&mut proof).unwrap();
    comm_r_b.serialize(&mut proof).unwrap();
    comm_r_c.serialize(&mut proof).unwrap();

    FusedProveBytes { acc_instance, acc_witness, proof }
}

/// Run the fused zk **fold** core (the full IVC step — forward (Vesta) /
/// reverse (Pallas), `num_addends = 3`) once on the GPU and serialize the
/// folded accumulator's `acc.instance ‖ acc.witness ‖ proof`. The fold core takes
/// **three** runtime affine arrays — the committer key `bases_h =
/// generators[:rows] ‖ hiding`, the HP placeholder identity `id_pt`, and the old
/// accumulator's `(6,)` commitments `acc_comms = [comm_a, comm_b, comm_c, hp_1,
/// hp_2, hp_3]` — versus the single-input prove's two; the prover's replayed
/// randomness + both inputs' witnesses are baked into the core as constants. The
/// output pytree (and thus the serialization, [`serialize_prove_outputs`]) is
/// identical to the single-input prove's — a fold IS an `AccumulatorInstance ‖
/// AccumulatorWitness ‖ Proof`. `r1cs_r_input` is the AS
/// `ProofRandomness.r1cs_r_input` host constant. One PJRT call replaces the per-MSM
/// `GpuBackend` fold dispatch; byte-identical to `ASForR1CSNark::prove` (the fold)
/// and to the host `r1cs_nark_as.prove_zk_fold`.
pub fn prove_fold_zk_fused<C: PastaCurve>(
    mlirbc: &[u8],
    bases_h: &[Affine<C>],
    id_pt: &Affine<C>,
    acc_comms: &[Affine<C>],
    r1cs_r_input: &[Fr<C>],
) -> FusedProveBytes
where
    <C::Params as ModelParameters>::BaseField: PrimeField,
{
    run_fold_fused::<C>(load_fused(mlirbc), bases_h, id_pt, acc_comms, r1cs_r_input)
}

/// Run an already-loaded fused fold core `exe` once (one PJRT call) and serialize
/// the folded `acc.instance ‖ acc.witness ‖ proof`. Split from [`load_fused`] so a
/// benchmark can compile the fold core once and time many warm runs (the [`run`,
/// not the compile, is the steady-state cost).
pub fn run_fold_fused<C: PastaCurve>(
    exe: &zkx_pjrt::Executable,
    bases_h: &[Affine<C>],
    id_pt: &Affine<C>,
    acc_comms: &[Affine<C>],
    r1cs_r_input: &[Fr<C>],
) -> FusedProveBytes
where
    <C::Params as ModelParameters>::BaseField: PrimeField,
{
    let bases_bytes = wire::g1_array_to_bytes(bases_h);
    let id_bytes = wire::g1_to_bytes(id_pt);
    let acc_bytes = wire::g1_array_to_bytes(acc_comms);
    let inputs = [
        (bases_bytes.as_slice(), vec![bases_h.len() as i64], C::G1_AFFINE),
        (id_bytes.as_slice(), vec![1i64], C::G1_AFFINE),
        (acc_bytes.as_slice(), vec![acc_comms.len() as i64], C::G1_AFFINE),
    ];
    // Safety: `exe` compiled by this same (leaked) client; the three rank-1 affine
    // inputs match the lowered fold core's argument order (`bases_h`, `id_pt` len 1,
    // `acc_comms` len 6 — the `_build_zk_fold_core` `_core(bases_h, id_pt, acc_comms)`).
    let out = unsafe { crate::gpu::session().run(exe, &inputs, NUM_OUTPUTS) };
    assert_eq!(
        out.len(),
        NUM_OUTPUTS,
        "fused fold core returned {} leaves, expected {NUM_OUTPUTS}",
        out.len()
    );
    serialize_prove_outputs::<C>(&out, r1cs_r_input)
}

/// Run the fused **no-zk** `ASForR1CSNark` prove core (the
/// no-zk twin of [`prove_fused`]) once on the GPU and serialize its
/// `acc.instance ‖ acc.witness ‖ proof`. The committer key `bases =
/// generators[:rows]` (no hiding base on the no-zk path, so — unlike the zk core —
/// no `id_pt`) is the core's sole runtime affine input; the circuit / witness are
/// baked constants. The core returns six leaves
/// (`comm_a, comm_b, comm_c, hp.instance[G;3], hp.a_open, hp.b_open`); the
/// `r1cs_input` and `blinded_witness` (= the raw witness, no `gamma·r`) are the two
/// host constants the caller supplies (the exporter baked their values in). Every
/// randomness / hiding `Option` is `None`, and the single-input fold leaves the HP
/// `Proof`'s low/high commitments empty. Byte-identical to the no-zk
/// `ASForR1CSNark::prove` and to the host `r1cs_nark_as.prove_no_zk`.
pub fn prove_no_zk_fused<C: PastaCurve>(
    mlirbc: &[u8],
    bases: &[Affine<C>],
    r1cs_input: &[Fr<C>],
    blinded_witness: &[Fr<C>],
) -> FusedProveBytes
where
    <C::Params as ModelParameters>::BaseField: PrimeField,
{
    // (comm_a, comm_b, comm_c, hp.instance[G;3], hp.a_open, hp.b_open).
    const NO_ZK_OUTPUTS: usize = 6;
    let exe = load_fused(mlirbc);
    let bases_bytes = wire::g1_array_to_bytes(bases);
    let input_bytes = wire::scalars_to_bytes::<C::Params>(r1cs_input);
    let witness_bytes = wire::scalars_to_bytes::<C::Params>(blinded_witness);
    let inputs = [
        (bases_bytes.as_slice(), vec![bases.len() as i64], C::G1_AFFINE),
        (input_bytes.as_slice(), vec![r1cs_input.len() as i64], C::SF),
        (witness_bytes.as_slice(), vec![blinded_witness.len() as i64], C::SF),
    ];
    // Safety: `exe` compiled by this same (leaked) client; the three rank-1 inputs
    // (committer key G1, `r1cs_input` fr, `blinded_witness` fr) match the general
    // no-zk core's `_core(bases, r1cs_input, blinded_witness)` argument order
    // (the assignment is a runtime input, no longer baked).
    let out = unsafe { crate::gpu::session().run(exe, &inputs, NO_ZK_OUTPUTS) };
    assert_eq!(
        out.len(),
        NO_ZK_OUTPUTS,
        "no-zk AS core returned {} leaves, expected {NO_ZK_OUTPUTS}",
        out.len()
    );
    let comm_a = parse_point::<C>(&out[0]);
    let comm_b = parse_point::<C>(&out[1]);
    let comm_c = parse_point::<C>(&out[2]);
    let hp_instance = parse_points::<C>(&out[3]); // comm_1, comm_2, comm_3
    let hp_a_vec = parse_scalars::<C>(&out[4]);
    let hp_b_vec = parse_scalars::<C>(&out[5]);

    // --- AccumulatorInstance: r1cs_input, comm_a, comm_b, comm_c, hp_instance.
    let mut acc_instance = Vec::new();
    r1cs_input.to_vec().serialize(&mut acc_instance).unwrap();
    comm_a.serialize(&mut acc_instance).unwrap();
    comm_b.serialize(&mut acc_instance).unwrap();
    comm_c.serialize(&mut acc_instance).unwrap();
    serialize_each(&hp_instance, &mut acc_instance);

    // --- AccumulatorWitness: r1cs_blinded_witness, hp{a_vec,b_vec, None}, None.
    let mut acc_witness = Vec::new();
    blinded_witness.to_vec().serialize(&mut acc_witness).unwrap();
    hp_a_vec.serialize(&mut acc_witness).unwrap();
    hp_b_vec.serialize(&mut acc_witness).unwrap();
    acc_witness.push(0); // InputWitnessRandomness = None (no-zk HP)
    acc_witness.push(0); // AccumulatorWitnessRandomness = None

    // --- Proof: empty low/high, None hiding, None ProofRandomness (single-input
    //     no-zk fold: no product-poly commitments, no AS proof randomness).
    let mut proof = Vec::new();
    Vec::<Affine<C>>::new().serialize(&mut proof).unwrap(); // low
    Vec::<Affine<C>>::new().serialize(&mut proof).unwrap(); // high
    proof.push(0); // ProofHidingCommitments = None
    proof.push(0); // ProofRandomness = None

    FusedProveBytes { acc_instance, acc_witness, proof }
}

/// Run the fused **no-zk NARK** core (the recursion half-step) once
/// on the GPU and serialize its `Proof`. The committer key `bases` (the recursion
/// generators — no hiding term, no-zk has no blinders) is the core's sole runtime
/// affine input; the core returns the three first-round commitments
/// (`comm_a, comm_b, comm_c`) as affine leaves, and the rest of the proof is the
/// host wrapper: the `None` first-round randomness flag, the blinded witness (the
/// raw `witness` on the no-zk path — a host constant the exporter baked into the
/// core), and the `None` second-round flag. Byte-identical to `R1CSNark::prove`
/// (no-zk) and to the host `nark.prove_no_zk_fused`. One PJRT call replaces the
/// three per-MSM `GpuBackend` commit dispatches the per-MSM strategy uses.
pub fn prove_nark_no_zk_fused<C: PastaCurve>(
    mlirbc: &[u8],
    bases: &[Affine<C>],
    witness: &[Fr<C>],
) -> Vec<u8>
where
    <C::Params as ModelParameters>::BaseField: PrimeField,
{
    const NARK_OUTPUTS: usize = 3; // NoZkNarkCore(comm_a, comm_b, comm_c)
    let exe = load_fused(mlirbc);
    let bases_bytes = wire::g1_array_to_bytes(bases);
    let inputs = [(bases_bytes.as_slice(), vec![bases.len() as i64], C::G1_AFFINE)];
    // Safety: `exe` compiled by this same (leaked) client; `bases` is the one
    // rank-1 affine input the no-zk core was lowered for.
    let out = unsafe { crate::gpu::session().run(exe, &inputs, NARK_OUTPUTS) };
    assert_eq!(
        out.len(),
        NARK_OUTPUTS,
        "no-zk NARK core returned {} leaves, expected {NARK_OUTPUTS}",
        out.len()
    );
    let comm_a = parse_point::<C>(&out[0]);
    let comm_b = parse_point::<C>(&out[1]);
    let comm_c = parse_point::<C>(&out[2]);

    // ark `CanonicalSerialize` of the no-zk `Proof` — mirrors `_serialize_proof`
    // in `python/accumulation_zorch/nark.py`. Affine `serialize` is the 33-byte
    // compressed form (= `curve.point_to_bytes`); `Vec<Fr>::serialize` is a u64
    // length prefix + 32 bytes each.
    let mut proof = Vec::new();
    comm_a.serialize(&mut proof).unwrap();
    comm_b.serialize(&mut proof).unwrap();
    comm_c.serialize(&mut proof).unwrap();
    proof.push(0); // FirstRoundMessage.randomness = None
    witness.to_vec().serialize(&mut proof).unwrap();
    proof.push(0); // SecondRoundMessage.randomness = None
    proof
}

/// Run the fused **zk NARK** core (the recursion half-step, make_zk
/// path) once on the GPU and serialize its zk `Proof`. The committer key + hiding
/// base (`bases_h` = `generators[:rows] ‖ hiding`) is the core's sole runtime
/// affine input; the prover's sampled randomness (the `r` witness blinders + the
/// 8 sigma blinders) is baked into the core as constants. The core returns, in
/// `NarkZkCore` field order, the eight commitments, the blinded witness, the three
/// sigma_{a,b,c}, sigma_o, and gamma (gamma is computed for the AS path but is not
/// part of the standalone proof). Byte-identical to the zk `R1CSNark::prove`
/// (`recursion_step_proves_on_vesta`, make_zk=true) and to the host
/// `nark.prove_zk`. One PJRT call replaces the eight per-MSM `GpuBackend` commit
/// dispatches.
pub fn prove_nark_zk_fused<C: PastaCurve>(mlirbc: &[u8], bases_h: &[Affine<C>]) -> Vec<u8>
where
    <C::Params as ModelParameters>::BaseField: PrimeField,
{
    // NarkZkCore(comm_a, comm_b, comm_c, comm_r_a, comm_r_b, comm_r_c, comm_1,
    //            comm_2, blinded_witness, sigma_abc, sigma_o, gamma).
    const NARK_ZK_OUTPUTS: usize = 12;
    let exe = load_fused(mlirbc);
    let bases_bytes = wire::g1_array_to_bytes(bases_h);
    let inputs = [(bases_bytes.as_slice(), vec![bases_h.len() as i64], C::G1_AFFINE)];
    // Safety: `exe` compiled by this same (leaked) client; `bases_h` is the one
    // rank-1 affine input the zk core was lowered for.
    let out = unsafe { crate::gpu::session().run(exe, &inputs, NARK_ZK_OUTPUTS) };
    assert_eq!(
        out.len(),
        NARK_ZK_OUTPUTS,
        "zk NARK core returned {} leaves, expected {NARK_ZK_OUTPUTS}",
        out.len()
    );
    let comm_a = parse_point::<C>(&out[0]);
    let comm_b = parse_point::<C>(&out[1]);
    let comm_c = parse_point::<C>(&out[2]);
    let comm_r_a = parse_point::<C>(&out[3]);
    let comm_r_b = parse_point::<C>(&out[4]);
    let comm_r_c = parse_point::<C>(&out[5]);
    let comm_1 = parse_point::<C>(&out[6]);
    let comm_2 = parse_point::<C>(&out[7]);
    let blinded_witness = parse_scalars::<C>(&out[8]);
    let sigma_abc = parse_scalars::<C>(&out[9]); // [sigma_a, sigma_b, sigma_c]
    let sigma_o = parse_scalars::<C>(&out[10]); // [sigma_o]
    // out[11] = gamma — retained for the AS path, not part of the proof bytes.

    // ark `CanonicalSerialize` of the zk `Proof` — mirrors `serialize_zk_proof`
    // in `python/accumulation_zorch/nark.py`. Affine `serialize` is the 33-byte
    // compressed form; `Vec<Fr>::serialize` is a u64 length prefix + 32B each; a
    // bare `Fr::serialize` is 32B with no prefix.
    let mut proof = Vec::new();
    comm_a.serialize(&mut proof).unwrap();
    comm_b.serialize(&mut proof).unwrap();
    comm_c.serialize(&mut proof).unwrap();
    proof.push(1); // FirstRoundMessage.randomness = Some
    comm_r_a.serialize(&mut proof).unwrap();
    comm_r_b.serialize(&mut proof).unwrap();
    comm_r_c.serialize(&mut proof).unwrap();
    comm_1.serialize(&mut proof).unwrap();
    comm_2.serialize(&mut proof).unwrap();
    blinded_witness.serialize(&mut proof).unwrap(); // Vec<Fr>: u64 len + 32B each
    proof.push(1); // SecondRoundMessage.randomness = Some
    serialize_each(&sigma_abc, &mut proof); // sigma_a, sigma_b, sigma_c
    serialize_each(&sigma_o, &mut proof); // sigma_o
    proof
}

/// Run the fused **IPA-PC accumulation decider** core (the size-`d` MSM) once on
/// the GPU and return the resulting affine point. The decider's only GPU-value
/// work is `final_key = Σ generators_i · coeffs_i` — `coeffs` the dense
/// `compute_coeffs(succinct_check(accumulator))` of the accumulator's check
/// polynomial, `generators` the IPA committer key — and the decider accepts iff
/// this equals `accumulator.final_comm_key` (`ipa_pc_as.decide_final_key` /
/// `IpaPC::check`'s final equality). Both the check-poly `coeffs` (scalar input)
/// and the committer-key `generators` (bases) are runtime inputs, so one lowered
/// `ipa_decider_msm_<curve>.mlirbc` decides any accumulator at that degree. One
/// PJRT call replaces the per-MSM `GpuBackend` decider dispatch; byte-identical to
/// the host `ipa_pc_as.decide_final_key` and (the byte-match gate) to the
/// accumulator's arkworks `final_comm_key`.
pub fn decide_ipa_msm_fused<C: PastaCurve>(
    mlirbc: &[u8],
    coeffs: &[Fr<C>],
    generators: &[Affine<C>],
) -> Affine<C>
where
    <C::Params as ModelParameters>::BaseField: PrimeField,
{
    run_decide_ipa_msm::<C>(load_fused(mlirbc), coeffs, generators)
}

/// Run an already-loaded decider MSM core `exe` once (one PJRT call) and return
/// the resulting point. Split from [`load_fused`] so the scale bench can compile
/// the core once and time many warm runs (the run, not the compile, is the
/// steady-state cost).
pub fn run_decide_ipa_msm<C: PastaCurve>(
    exe: &zkx_pjrt::Executable,
    coeffs: &[Fr<C>],
    generators: &[Affine<C>],
) -> Affine<C>
where
    <C::Params as ModelParameters>::BaseField: PrimeField,
{
    const DECIDER_OUTPUTS: usize = 1; // the single folded `final_key` point.
    let coeffs_bytes = wire::scalars_to_bytes::<C::Params>(coeffs);
    let gens_bytes = wire::g1_array_to_bytes(generators);
    let inputs = [
        (coeffs_bytes.as_slice(), vec![coeffs.len() as i64], C::SF),
        (gens_bytes.as_slice(), vec![generators.len() as i64], C::G1_AFFINE),
    ];
    // Safety: `exe` compiled by this same (leaked) client; the two rank-1 inputs
    // (`coeffs` fr scalars, `generators` G1 affine) match the decider core's
    // `_core(scalars, bases)` argument order (`export/export_ipa.py`).
    let out = unsafe { crate::gpu::session().run(exe, &inputs, DECIDER_OUTPUTS) };
    assert_eq!(
        out.len(),
        DECIDER_OUTPUTS,
        "decider MSM core returned {} leaves, expected {DECIDER_OUTPUTS}",
        out.len()
    );
    parse_point::<C>(&out[0])
}

/// Serialize each element of `items` individually (struct fields, no `Vec`
/// length prefix).
fn serialize_each<T: CanonicalSerialize>(items: &[T], out: &mut Vec<u8>) {
    for it in items {
        it.serialize(&mut *out).unwrap();
    }
}
