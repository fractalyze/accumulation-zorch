//! R1CS-NARK-AS prove (no-zk) end-to-end fixtures for the frx port, over either
//! Pasta cycle curve (Pallas or Vesta).
//!
//! This is the acceptance criterion: the ported `prove` must reproduce the
//! serialized `(acc.instance ‖ acc.witness ‖ proof)` that
//! `src/oracle.rs`'s `prove_byte_identical_to_arkworks_no_zk` test pins to
//! arkworks — seeds {0, 42}, num_inputs=5, num_constraints=10.
//!
//! The flow mirrors `oracle.rs`'s `prove_bytes!` macro exactly (same `StdRng`
//! draw order) so the golden bytes are the oracle's bytes. Per seed it dumps the
//! replay inputs (the single input's `r1cs_input` and `blinded_witness`) and the
//! golden output split into `acc.instance` / `acc.witness` / `proof` for
//! localization, plus the **decider oracle**: it runs the unmodified arkworks
//! `ASForR1CSNark::decide` on the produced accumulator and asserts it accepts, so
//! the fixture is emitted only for an accumulator arkworks decides `true`. The
//! decider recomputes `comm_{a,b,c}` (the size-`n` MSMs the frx decide port
//! reproduces) and accepts iff they equal the accumulator's stored commitments —
//! which live in `acc_instance_hex` — so no extra golden is dumped beyond the
//! `decide` flag.
//!
//! The seed-independent structural inputs (matrices a/b/c, the committer-key
//! generators, `supported_num_elems`) are dumped once.
//!
//! The no-zk single-input path draws no randomness; `gamma`, `hash_matrices`,
//! and the `beta` absorb are no-ops for these bytes (beta = [1] when there is a
//! single addend, gamma is gated on first-round randomness) — they ride with
//! the zk path. So the Python side recomputes `comm_a/b/c` from the
//! matrices + assignments (the NARK prove path) rather than reading them.
//!
//! Generic over the curve (`PastaCurve`-style, no per-curve copy) — the curve is
//! a CLI arg, defaulting to Pallas:
//!
//!   cargo run --example dump_as -- pallas > python/testdata/as_fixtures.json
//!   cargo run --example dump_as -- vesta  > python/testdata/as_vesta_fixtures.json

use ark_ec::models::ModelParameters;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ec::SWModelParameters;
use ark_ff::{Field, PrimeField, UniformRand};
use ark_poly_commit::trivial_pc::PedersenCommitment;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, OptimizationGoal, SynthesisMode,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_sponge::{Absorbable, CryptographicSponge};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;

use fixture_json::{
    curve_main, fe_list, matrix_json, point_list, ser_hex, DummyCircuit, MatrixJson, PointJson,
};

use ark_accumulation::r1cs_nark_as::r1cs_nark::R1CSNark;
use ark_accumulation::r1cs_nark_as::{ASForR1CSNark, InputInstance};
use ark_accumulation::{AccumulationScheme, Accumulator, Input, MakeZK};

/// `ConstraintF<G>` (the sponge / constraint field), re-derived (it is
/// `pub(crate)` upstream). For the Pasta curves the base field is already prime.
type CF<P> = <<P as ModelParameters>::BaseField as Field>::BasePrimeField;
type Sponge<P> = ark_sponge::poseidon::PoseidonSponge<CF<P>>;
type AS<P> = ASForR1CSNark<GroupAffine<P>, Sponge<P>>;

const NUM_INPUTS: usize = 5;
const NUM_CONSTRAINTS: usize = 10;
const SEEDS: [u64; 2] = [0, 42];

/// One seed's replay inputs + golden output. Field order is the fixture's key
/// order.
#[derive(Serialize)]
struct SeedJson {
    seed: u64,
    r1cs_input: Vec<String>,
    blinded_witness: Vec<String>,
    acc_instance_hex: String,
    acc_witness_hex: String,
    proof_hex: String,
    decide: bool,
}

/// The whole fixture. Field order is the fixture's key order.
#[derive(Serialize)]
struct AsFixture {
    note: String,
    curve: String,
    num_inputs: usize,
    num_constraints: usize,
    supported_num_elems: usize,
    a: MatrixJson,
    b: MatrixJson,
    c: MatrixJson,
    generators: Vec<PointJson>,
    hiding: PointJson,
    seeds: Vec<SeedJson>,
}

/// One seeded no-zk accumulation step, mirroring `oracle.rs`'s `prove_bytes!`,
/// followed by the arkworks decider on the produced accumulator. Returns the
/// replay inputs, the golden serialized accumulator + proof, and the decider's
/// verdict (asserted `true` here).
fn run_seed<P>(seed: u64) -> SeedJson
where
    P: SWModelParameters,
    P::BaseField: PrimeField,
    GroupAffine<P>: Absorbable<CF<P>>,
    CF<P>: PrimeField + Absorbable<CF<P>>,
{
    let mut rng = StdRng::seed_from_u64(seed);

    // NARK setup + index over a freshly sampled circuit (draws a, b).
    let nark_pp = R1CSNark::<GroupAffine<P>, Sponge<P>>::setup();
    let index_circuit = DummyCircuit::<P::ScalarField> {
        a: Some(P::ScalarField::rand(&mut rng)),
        b: Some(P::ScalarField::rand(&mut rng)),
        num_inputs: NUM_INPUTS,
        num_constraints: NUM_CONSTRAINTS,
    };
    let (ipk, ivk) = R1CSNark::<GroupAffine<P>, Sponge<P>>::index(&nark_pp, index_circuit).unwrap();

    // Accumulation-scheme setup + index (no-zk setup draws no randomness). The
    // decider key `dk` (discarded by the prove-only dumps) drives the decider.
    let as_pp = AS::<P>::setup(&mut rng).unwrap();
    let (pk, _vk, dk) = AS::<P>::index(&as_pp, &(), &(ipk.clone(), ivk.clone())).unwrap();

    // Build one accumulation input by running the NARK prover (draws a, b).
    let circuit = DummyCircuit::<P::ScalarField> {
        a: Some(P::ScalarField::rand(&mut rng)),
        b: Some(P::ScalarField::rand(&mut rng)),
        num_inputs: NUM_INPUTS,
        num_constraints: NUM_CONSTRAINTS,
    };
    let nark_sponge = ASForR1CSNark::<GroupAffine<P>, Sponge<P>>::nark_sponge(&Sponge::<P>::new());
    let nark_proof = R1CSNark::<GroupAffine<P>, Sponge<P>>::prove(
        &ipk,
        circuit.clone(),
        false,
        Some(nark_sponge),
        Some(&mut rng),
    )
    .unwrap();

    // R1CS input + witness assignments (Weight mode, as oracle.rs extracts the
    // r1cs_input). Assignments are optimization-goal-invariant; no-zk
    // blinded_witness = witness.
    let pcs = ConstraintSystem::new_ref();
    pcs.set_optimization_goal(OptimizationGoal::Weight);
    pcs.set_mode(SynthesisMode::Prove {
        construct_matrices: false,
    });
    circuit.generate_constraints(pcs.clone()).unwrap();
    pcs.finalize();
    let (r1cs_input, blinded_witness) = {
        let cs = pcs.borrow().unwrap();
        (cs.instance_assignment.clone(), cs.witness_assignment.clone())
    };

    let input = Input::<CF<P>, Sponge<P>, AS<P>> {
        instance: InputInstance {
            r1cs_input: r1cs_input.clone(),
            first_round_message: nark_proof.first_msg.clone(),
        },
        witness: nark_proof.second_msg,
    };
    let inputs = vec![input];
    let no_accumulators: Vec<Accumulator<CF<P>, Sponge<P>, AS<P>>> = Vec::new();

    let (accumulator, proof) = AS::<P>::prove(
        &pk,
        Input::<CF<P>, Sponge<P>, AS<P>>::map_to_refs(&inputs),
        Accumulator::<CF<P>, Sponge<P>, AS<P>>::map_to_refs(&no_accumulators),
        MakeZK::Disabled,
        None,
    )
    .unwrap();

    // The decider oracle: recompute comm_{a,b,c} + the HP check and accept iff
    // they equal the accumulator's stored commitments. The frx decide port
    // reproduces this; the produced fixture is valid only if arkworks accepts.
    let decided = AS::<P>::decide(&dk, accumulator.as_ref(), None::<Sponge<P>>).unwrap();
    assert!(decided, "arkworks decider must accept the produced accumulator (seed {})", seed);

    SeedJson {
        seed,
        r1cs_input: fe_list(&r1cs_input),
        blinded_witness: fe_list(&blinded_witness),
        acc_instance_hex: ser_hex(&accumulator.instance),
        acc_witness_hex: ser_hex(&accumulator.witness),
        proof_hex: ser_hex(&proof),
        decide: decided,
    }
}

fn dump<P>(curve: &str)
where
    P: SWModelParameters,
    P::BaseField: PrimeField,
    GroupAffine<P>: Absorbable<CF<P>>,
    CF<P>: PrimeField + Absorbable<CF<P>>,
{
    // Seed-independent structural inputs: matrices a/b/c + committer key. The
    // matrices depend only on the circuit shape (coefficients are 1), so derive
    // them from a value-free Setup synthesis; the committer key is deterministic
    // in num_constraints.
    let shape_circuit = DummyCircuit::<P::ScalarField> {
        a: None,
        b: None,
        num_inputs: NUM_INPUTS,
        num_constraints: NUM_CONSTRAINTS,
    };
    let mcs = ConstraintSystem::<P::ScalarField>::new_ref();
    mcs.set_optimization_goal(OptimizationGoal::Constraints);
    mcs.set_mode(SynthesisMode::Setup);
    shape_circuit.generate_constraints(mcs.clone()).unwrap();
    mcs.finalize();
    let matrices = mcs.to_matrices().unwrap();
    let num_constraints = mcs.num_constraints();

    // Committer-key generators (recovered from the key's uncompressed form,
    // as dump_nark.rs does).
    let cpp = PedersenCommitment::<GroupAffine<P>>::setup(num_constraints);
    let ck = PedersenCommitment::<GroupAffine<P>>::trim(&cpp, num_constraints);
    let supported_num_elems = ck.supported_num_elems();
    // generators + the hiding generator: the decider's fused GPU core commits over
    // `generators ‖ hiding` (the randomizer rides as the trailing MSM term, 0 on the
    // no-zk path), so the hiding base is dumped even though the no-zk commitments
    // are non-hiding.
    let (generators, hiding) = {
        let mut b = Vec::new();
        ck.serialize_uncompressed(&mut b).unwrap();
        let mut r = &b[..];
        let g = Vec::<GroupAffine<P>>::deserialize_uncompressed(&mut r).unwrap();
        let h = GroupAffine::<P>::deserialize_uncompressed(&mut r).unwrap();
        (g, h)
    };
    let fixture = AsFixture {
        note: format!("R1CS-NARK-AS no-zk prove + decide fixtures ({} curve)", curve),
        curve: curve.to_string(),
        num_inputs: NUM_INPUTS,
        num_constraints,
        supported_num_elems,
        a: matrix_json(&matrices.a),
        b: matrix_json(&matrices.b),
        c: matrix_json(&matrices.c),
        generators: point_list(&generators),
        hiding: PointJson::from_affine(&hiding),
        seeds: SEEDS.iter().map(|&seed| run_seed::<P>(seed)).collect(),
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}

curve_main!(dump);
