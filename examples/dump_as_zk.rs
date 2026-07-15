//! R1CS-NARK-AS prove (zk) end-to-end fixtures for the frx port, over either
//! Pasta cycle curve (Pallas or Vesta) — the zk acceptance criterion. Drives
//! the unmodified arkworks zk prove (seeds {0, 42}, num_inputs=5,
//! num_constraints=10), so the golden bytes are the oracle's.
//!
//! Per seed it dumps the replay inputs (the single input's `r1cs_input` + raw
//! witness) and the golden zk `(acc.instance ‖ acc.witness ‖ proof)`, plus the
//! whole cross-prover randomness, recovered by replaying the exact `Fr::rand`
//! draw schedule on a fresh same-seed `StdRng`: 4 harness draws (index/circuit
//! a,b — discarded; the assignments are dumped directly), then NARK (`r` ×
//! num_witness, then `a/b/c_blinder`, `r_a/r_b/r_c_blinder`, `blinder_1`,
//! `blinder_2`), then AS `generate_prover_randomness` (`r1cs_r_input`,
//! `r1cs_r_witness` — `vec![rand;n]`, one draw each — then `rand_1/2/3`), then
//! HP `generate_prover_randomness` (`hiding_a`, `hiding_b`, `rand_1/2/3`).
//!
//! It also pins the **decider oracle**: the unmodified arkworks
//! `ASForR1CSNark::decide` is run on the produced accumulator and asserted to
//! accept, so the fixture is emitted only for an accumulator arkworks decides
//! `true`. The zk decider recomputes the hiding commitments `comm_{a,b,c} =
//! commit(M·z, σ_*)` (the size-`n` MSMs the frx decide port reproduces) and
//! accepts iff they equal the accumulator's stored commitments — which live in
//! `acc_instance_hex` — so no extra golden is dumped beyond the `decide` flag.
//!
//! Run: cargo run --example dump_as_zk -- pallas > python/testdata/as_zk_fixtures.json
//!      cargo run --example dump_as_zk -- vesta  > python/testdata/as_zk_vesta_fixtures.json

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
    curve_main, fe_hex, fe_list, hash_matrices, hex, matrix_json, point_list, ser_hex, DummyCircuit,
    MatrixJson, PointJson,
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
/// Circuit size — env `AS_ZK_NUM_CONSTRAINTS`, default 10 (the committed
/// size-10 fixture). Scales the R1CS rows, the committer key, and the MSM
/// length; override only for off-tree scale/bench runs.
fn cfg_num_constraints() -> usize {
    std::env::var("AS_ZK_NUM_CONSTRAINTS").ok().and_then(|s| s.parse().ok()).unwrap_or(10)
}
const SEEDS: [u64; 2] = [0, 42];
const NARK_PROTOCOL_NAME: &[u8] = b"R1CS-NARK-2020";
const AS_PROTOCOL_NAME: &[u8] = b"AS-FOR-R1CS-NARK-2020";

/// One seed's replay inputs, replayed randomness, and golden output. Field order
/// is the fixture's key order.
#[derive(Serialize)]
struct SeedJson {
    seed: u64,
    r1cs_input: Vec<String>,
    witness: Vec<String>,
    r: Vec<String>,
    a_blinder: String,
    b_blinder: String,
    c_blinder: String,
    r_a_blinder: String,
    r_b_blinder: String,
    r_c_blinder: String,
    blinder_1: String,
    blinder_2: String,
    as_r1cs_r_input: String,
    as_r1cs_r_witness: String,
    as_rand_1: String,
    as_rand_2: String,
    as_rand_3: String,
    hp_hiding_a: String,
    hp_hiding_b: String,
    hp_rand_1: String,
    hp_rand_2: String,
    hp_rand_3: String,
    acc_instance_hex: String,
    acc_witness_hex: String,
    proof_hex: String,
    decide: bool,
}

/// The whole fixture. Field order is the fixture's key order.
#[derive(Serialize)]
struct AsZkFixture {
    note: String,
    curve: String,
    num_inputs: usize,
    num_constraints: usize,
    supported_num_elems: usize,
    nark_matrices_hash_hex: String,
    as_matrices_hash_hex: String,
    a: MatrixJson,
    b: MatrixJson,
    c: MatrixJson,
    generators: Vec<PointJson>,
    hiding: PointJson,
    seeds: Vec<SeedJson>,
}

/// One seeded zk accumulation step on the unmodified arkworks prover, the
/// replayed randomness, and the arkworks decider verdict on the produced
/// accumulator (asserted `true`).
fn run_seed<P>(seed: u64) -> SeedJson
where
    P: SWModelParameters,
    P::BaseField: PrimeField,
    GroupAffine<P>: Absorbable<CF<P>>,
    CF<P>: PrimeField + Absorbable<CF<P>>,
{
    let mut rng = StdRng::seed_from_u64(seed);

    let nark_pp = R1CSNark::<GroupAffine<P>, Sponge<P>>::setup();
    let index_circuit = DummyCircuit::<P::ScalarField> {
        a: Some(P::ScalarField::rand(&mut rng)),
        b: Some(P::ScalarField::rand(&mut rng)),
        num_inputs: NUM_INPUTS,
        num_constraints: cfg_num_constraints(),
    };
    let (ipk, ivk) = R1CSNark::<GroupAffine<P>, Sponge<P>>::index(&nark_pp, index_circuit).unwrap();

    let as_pp = AS::<P>::setup(&mut rng).unwrap();
    let (pk, _vk, dk) = AS::<P>::index(&as_pp, &(), &(ipk.clone(), ivk.clone())).unwrap();

    let circuit = DummyCircuit::<P::ScalarField> {
        a: Some(P::ScalarField::rand(&mut rng)),
        b: Some(P::ScalarField::rand(&mut rng)),
        num_inputs: NUM_INPUTS,
        num_constraints: cfg_num_constraints(),
    };
    let _bench_t0 = std::time::Instant::now();
    let nark_sponge = ASForR1CSNark::<GroupAffine<P>, Sponge<P>>::nark_sponge(&Sponge::<P>::new());
    let nark_proof = R1CSNark::<GroupAffine<P>, Sponge<P>>::prove(
        &ipk,
        circuit.clone(),
        true,
        Some(nark_sponge),
        Some(&mut rng),
    )
    .unwrap();

    let pcs = ConstraintSystem::new_ref();
    pcs.set_optimization_goal(OptimizationGoal::Weight);
    pcs.set_mode(SynthesisMode::Prove { construct_matrices: false });
    circuit.generate_constraints(pcs.clone()).unwrap();
    pcs.finalize();
    let (r1cs_input, witness) = {
        let cs = pcs.borrow().unwrap();
        (cs.instance_assignment.clone(), cs.witness_assignment.clone())
    };
    let num_witness = witness.len();

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
        MakeZK::Enabled(&mut rng),
        None,
    )
    .unwrap();
    eprintln!(
        "[bench] seed {} n={}: full zk prove (NARK+AS) = {:?} (CPU arkworks)",
        seed,
        cfg_num_constraints(),
        _bench_t0.elapsed()
    );

    // The decider oracle: arkworks recomputes the hiding commitments and accepts
    // iff they equal the accumulator's stored commitments. The frx zk decide
    // port reproduces this; the fixture is valid only if arkworks accepts.
    let decided = AS::<P>::decide(&dk, accumulator.as_ref(), None::<Sponge<P>>).unwrap();
    assert!(decided, "arkworks decider must accept the produced zk accumulator (seed {})", seed);

    // Replay the full draw schedule on a fresh same-seed rng: 4 harness draws
    // (index/circuit a,b — discarded), then NARK, AS-gen, HP-gen.
    let mut rep = StdRng::seed_from_u64(seed);
    for _ in 0..4 {
        let _ = P::ScalarField::rand(&mut rep);
    }
    let r: Vec<P::ScalarField> = (0..num_witness).map(|_| P::ScalarField::rand(&mut rep)).collect();
    let nark_blinders: Vec<P::ScalarField> = (0..8).map(|_| P::ScalarField::rand(&mut rep)).collect();
    let as_r1cs_r_input = P::ScalarField::rand(&mut rep);
    let as_r1cs_r_witness = P::ScalarField::rand(&mut rep);
    let as_rand: Vec<P::ScalarField> = (0..3).map(|_| P::ScalarField::rand(&mut rep)).collect();
    let hp_hiding_a = P::ScalarField::rand(&mut rep);
    let hp_hiding_b = P::ScalarField::rand(&mut rep);
    let hp_rand: Vec<P::ScalarField> = (0..3).map(|_| P::ScalarField::rand(&mut rep)).collect();

    SeedJson {
        seed,
        r1cs_input: fe_list(&r1cs_input),
        witness: fe_list(&witness),
        r: fe_list(&r),
        a_blinder: fe_hex(&nark_blinders[0]),
        b_blinder: fe_hex(&nark_blinders[1]),
        c_blinder: fe_hex(&nark_blinders[2]),
        r_a_blinder: fe_hex(&nark_blinders[3]),
        r_b_blinder: fe_hex(&nark_blinders[4]),
        r_c_blinder: fe_hex(&nark_blinders[5]),
        blinder_1: fe_hex(&nark_blinders[6]),
        blinder_2: fe_hex(&nark_blinders[7]),
        as_r1cs_r_input: fe_hex(&as_r1cs_r_input),
        as_r1cs_r_witness: fe_hex(&as_r1cs_r_witness),
        as_rand_1: fe_hex(&as_rand[0]),
        as_rand_2: fe_hex(&as_rand[1]),
        as_rand_3: fe_hex(&as_rand[2]),
        hp_hiding_a: fe_hex(&hp_hiding_a),
        hp_hiding_b: fe_hex(&hp_hiding_b),
        hp_rand_1: fe_hex(&hp_rand[0]),
        hp_rand_2: fe_hex(&hp_rand[1]),
        hp_rand_3: fe_hex(&hp_rand[2]),
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
    // Seed-independent structural inputs (matrices, committer key, matrix hashes).
    let shape_circuit = DummyCircuit::<P::ScalarField> {
        a: None,
        b: None,
        num_inputs: NUM_INPUTS,
        num_constraints: cfg_num_constraints(),
    };
    let mcs = ConstraintSystem::<P::ScalarField>::new_ref();
    mcs.set_optimization_goal(OptimizationGoal::Constraints);
    mcs.set_mode(SynthesisMode::Setup);
    shape_circuit.generate_constraints(mcs.clone()).unwrap();
    mcs.finalize();
    let matrices = mcs.to_matrices().unwrap();
    let num_constraints = mcs.num_constraints();
    let nark_matrices_hash = hash_matrices(NARK_PROTOCOL_NAME, &matrices.a, &matrices.b, &matrices.c);
    let as_matrices_hash = hash_matrices(AS_PROTOCOL_NAME, &matrices.a, &matrices.b, &matrices.c);

    let cpp = PedersenCommitment::<GroupAffine<P>>::setup(num_constraints);
    let ck = PedersenCommitment::<GroupAffine<P>>::trim(&cpp, num_constraints);
    let supported_num_elems = ck.supported_num_elems();
    let (generators, hiding) = {
        let mut b = Vec::new();
        ck.serialize_uncompressed(&mut b).unwrap();
        let mut r = &b[..];
        let g = Vec::<GroupAffine<P>>::deserialize_uncompressed(&mut r).unwrap();
        let h = GroupAffine::<P>::deserialize_uncompressed(&mut r).unwrap();
        (g, h)
    };
    let fixture = AsZkFixture {
        note: format!("R1CS-NARK-AS zk prove + decide fixtures ({} curve)", curve),
        curve: curve.to_string(),
        num_inputs: NUM_INPUTS,
        num_constraints,
        supported_num_elems,
        nark_matrices_hash_hex: hex(&nark_matrices_hash),
        as_matrices_hash_hex: hex(&as_matrices_hash),
        a: matrix_json(&matrices.a),
        b: matrix_json(&matrices.b),
        c: matrix_json(&matrices.c),
        generators: point_list(&generators),
        hiding: PointJson::from_affine(&hiding),
        seeds: SEEDS.iter().map(|&s| run_seed::<P>(s)).collect(),
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}

curve_main!(dump);
