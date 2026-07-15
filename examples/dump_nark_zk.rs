//! NARK prove (zk) fixtures for the frx port.
//!
//! Drives the crate's real `R1CSNark::prove` with `make_zk = true` over a fixed
//! `DummyCircuit` and dumps the golden serialized zk `Proof`, the replay inputs
//! (matrices, instance/witness assignments, committer-key generators + hiding
//! generator), the `nark_matrices_hash` (via `fixture_json::hash_matrices`; the
//! crate's own `hash_matrices` is `pub(crate)`, so it cannot be called here),
//! and the prover's sampled randomness, recovered by replaying the exact
//! `Fr::rand` draw schedule on a fresh same-seed `StdRng`.
//!
//! Draw order in `R1CSNark::prove` (make_zk): `r` (one per witness var, a loop —
//! distinct), then `a/b/c_blinder`, `r_a/r_b/r_c_blinder`, `blinder_1`,
//! `blinder_2`. The golden proof is produced from an identically-seeded run, so
//! the replayed values are the ones the proof used; the byte-match validates it.
//!
//! Run: `cargo run --example dump_nark_zk > python/testdata/nark_zk_fixtures.json`

use ark_ff::UniformRand;
use ark_pallas::{Affine, Fr};
use ark_poly_commit::trivial_pc::PedersenCommitment;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, OptimizationGoal, SynthesisMode,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_sponge::CryptographicSponge;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;

use ark_accumulation::r1cs_nark_as::r1cs_nark::R1CSNark;
use ark_accumulation::r1cs_nark_as::ASForR1CSNark;
use fixture_json::{
    fe_hex, fe_list, hash_matrices, hex, matrix_json, point_list, ser_hex, DummyCircuit, MatrixJson,
    PointJson,
};

type Sponge = ark_sponge::poseidon::PoseidonSponge<ark_pallas::Fq>;

const NUM_INPUTS: usize = 5;
const NUM_CONSTRAINTS: usize = 10;
const SEED: u64 = 7;
const NARK_PROTOCOL_NAME: &[u8] = b"R1CS-NARK-2020";

/// The fixture schema. Field order is the emitted key order.
#[derive(Serialize)]
struct NarkZkFixture {
    note: String,
    num_inputs: usize,
    num_constraints: usize,
    nark_matrices_hash_hex: String,
    a: MatrixJson,
    b: MatrixJson,
    c: MatrixJson,
    input: Vec<String>,
    witness: Vec<String>,
    generators: Vec<PointJson>,
    hiding: PointJson,
    r: Vec<String>,
    a_blinder: String,
    b_blinder: String,
    c_blinder: String,
    r_a_blinder: String,
    r_b_blinder: String,
    r_c_blinder: String,
    blinder_1: String,
    blinder_2: String,
    proof_hex: String,
}

fn main() {
    let circuit = DummyCircuit::<Fr> {
        a: Some(Fr::from(3u64)),
        b: Some(Fr::from(5u64)),
        num_inputs: NUM_INPUTS,
        num_constraints: NUM_CONSTRAINTS,
    };

    let pp = R1CSNark::<Affine, Sponge>::setup();
    let (ipk, _ivk) = R1CSNark::<Affine, Sponge>::index(&pp, circuit.clone()).unwrap();

    // Golden zk proof, seeded. gamma uses the NARK-forked sponge (as the AS path
    // supplies), which the Python `compute_challenge` reconstructs.
    let mut rng = StdRng::seed_from_u64(SEED);
    let nark_sponge = ASForR1CSNark::<Affine, Sponge>::nark_sponge(&Sponge::new());
    let proof =
        R1CSNark::<Affine, Sponge>::prove(&ipk, circuit.clone(), true, Some(nark_sponge), Some(&mut rng))
            .unwrap();
    let proof_hex = ser_hex(&proof);

    // Matrices a/b/c (Setup mode) — match the ipk's and feed hash_matrices.
    let mcs = ConstraintSystem::<Fr>::new_ref();
    mcs.set_optimization_goal(OptimizationGoal::Constraints);
    mcs.set_mode(SynthesisMode::Setup);
    circuit.clone().generate_constraints(mcs.clone()).unwrap();
    mcs.finalize();
    let matrices = mcs.to_matrices().unwrap();
    let nark_matrices_hash = hash_matrices(NARK_PROTOCOL_NAME, &matrices.a, &matrices.b, &matrices.c);

    // Instance + witness assignments (Prove mode).
    let pcs = ConstraintSystem::<Fr>::new_ref();
    pcs.set_optimization_goal(OptimizationGoal::Constraints);
    pcs.set_mode(SynthesisMode::Prove { construct_matrices: false });
    circuit.clone().generate_constraints(pcs.clone()).unwrap();
    pcs.finalize();
    let (input, witness, num_constraints) = {
        let cs = pcs.borrow().unwrap();
        (cs.instance_assignment.clone(), cs.witness_assignment.clone(), cs.num_constraints)
    };
    let num_witness = witness.len();

    // Committer key: generators + the hiding generator (the zk path commits with
    // a blinder, so the hiding generator is needed too).
    let cpp = PedersenCommitment::<Affine>::setup(num_constraints);
    let ck = PedersenCommitment::<Affine>::trim(&cpp, num_constraints);
    let (generators, hiding) = {
        let mut b = Vec::new();
        ck.serialize_uncompressed(&mut b).unwrap();
        let mut r = &b[..];
        let g = Vec::<Affine>::deserialize_uncompressed(&mut r).unwrap();
        let h = Affine::deserialize_uncompressed(&mut r).unwrap();
        (g, h)
    };
    // Replay the make_zk draw schedule on a fresh same-seed rng to recover the
    // sampled randomness (the proof above was produced from the same seed).
    let mut rep = StdRng::seed_from_u64(SEED);
    let r: Vec<Fr> = (0..num_witness).map(|_| Fr::rand(&mut rep)).collect();
    let a_blinder = Fr::rand(&mut rep);
    let b_blinder = Fr::rand(&mut rep);
    let c_blinder = Fr::rand(&mut rep);
    let r_a_blinder = Fr::rand(&mut rep);
    let r_b_blinder = Fr::rand(&mut rep);
    let r_c_blinder = Fr::rand(&mut rep);
    let blinder_1 = Fr::rand(&mut rep);
    let blinder_2 = Fr::rand(&mut rep);

    let fixture = NarkZkFixture {
        note: "NARK zk prove fixtures".to_string(),
        num_inputs: input.len(),
        num_constraints,
        nark_matrices_hash_hex: hex(&nark_matrices_hash),
        a: matrix_json(&matrices.a),
        b: matrix_json(&matrices.b),
        c: matrix_json(&matrices.c),
        input: fe_list(&input),
        witness: fe_list(&witness),
        generators: point_list(&generators),
        hiding: PointJson::from_affine(&hiding),
        r: fe_list(&r),
        a_blinder: fe_hex(&a_blinder),
        b_blinder: fe_hex(&b_blinder),
        c_blinder: fe_hex(&c_blinder),
        r_a_blinder: fe_hex(&r_a_blinder),
        r_b_blinder: fe_hex(&r_b_blinder),
        r_c_blinder: fe_hex(&r_c_blinder),
        blinder_1: fe_hex(&blinder_1),
        blinder_2: fe_hex(&blinder_2),
        proof_hex,
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}
