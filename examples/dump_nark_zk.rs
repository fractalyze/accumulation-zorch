//! Slice-6a NARK prove (zk) fixtures for the jax port (zorch#303).
//!
//! Drives the crate's real `R1CSNark::prove` with `make_zk = true` over a fixed
//! `DummyCircuit` and dumps the golden serialized zk `Proof`, the replay inputs
//! (matrices, instance/witness assignments, committer-key generators + hiding
//! generator), the `nark_matrices_hash` (recomputed here via blake2, since the
//! crate's `hash_matrices` is `pub(crate)`), and — the new piece for the zk path
//! — the prover's sampled randomness, recovered by replaying the exact
//! `Fr::rand` draw schedule on a fresh same-seed `StdRng`.
//!
//! Draw order in `R1CSNark::prove` (make_zk): `r` (one per witness var, a loop —
//! distinct), then `a/b/c_blinder`, `r_a/r_b/r_c_blinder`, `blinder_1`,
//! `blinder_2`. The golden proof is produced from an identically-seeded run, so
//! the replayed values are the ones the proof used; the byte-match validates it.
//!
//! Run: `cargo run --example dump_nark_zk > python/testdata/nark_zk_fixtures.json`

use ark_ff::{BigInteger, PrimeField, UniformRand};
use ark_pallas::{Affine, Fr};
use ark_poly_commit::trivial_pc::PedersenCommitment;
use ark_relations::lc;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, Matrix, OptimizationGoal,
    SynthesisError, SynthesisMode,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_sponge::CryptographicSponge;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use blake2::VarBlake2b;
use digest::{Update, VariableOutput};

use ark_accumulation::r1cs_nark_as::r1cs_nark::R1CSNark;
use ark_accumulation::r1cs_nark_as::ASForR1CSNark;

type Sponge = ark_sponge::poseidon::PoseidonSponge<ark_pallas::Fq>;

const NUM_INPUTS: usize = 5;
const NUM_CONSTRAINTS: usize = 10;
const SEED: u64 = 7;
const NARK_PROTOCOL_NAME: &[u8] = b"R1CS-NARK-2020";

#[derive(Clone)]
struct DummyCircuit {
    a: Option<Fr>,
    b: Option<Fr>,
    num_inputs: usize,
    num_constraints: usize,
}

impl ConstraintSynthesizer<Fr> for DummyCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        let a = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        let b = cs.new_witness_variable(|| self.b.ok_or(SynthesisError::AssignmentMissing))?;
        let c = cs.new_input_variable(|| {
            let a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
            let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;
            Ok(a * b)
        })?;
        for _ in 0..(self.num_inputs - 1) {
            cs.new_input_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        }
        for _ in 0..(self.num_constraints - 1) {
            cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        }
        cs.enforce_constraint(lc!(), lc!(), lc!())?;
        Ok(())
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn fr_hex(f: &Fr) -> String {
    hex(&f.into_repr().to_bytes_le())
}

fn ser_hex<T: CanonicalSerialize>(v: &T) -> String {
    let mut b = Vec::new();
    v.serialize(&mut b).unwrap();
    hex(&b)
}

fn fr_list_json(xs: &[Fr]) -> String {
    let v: Vec<String> = xs.iter().map(|f| format!("\"{}\"", fr_hex(f))).collect();
    format!("[{}]", v.join(","))
}

fn matrix_json(m: &Matrix<Fr>) -> String {
    let rows: Vec<String> = m
        .iter()
        .map(|row| {
            let entries: Vec<String> = row
                .iter()
                .map(|(coeff, idx)| format!("[\"{}\",{}]", fr_hex(coeff), idx))
                .collect();
            format!("[{}]", entries.join(","))
        })
        .collect();
    format!("[{}]", rows.join(","))
}

fn point_json(p: &Affine) -> String {
    use ark_ff::Zero;
    let (x, y) = if p.is_zero() {
        (hex(&[0u8; 32]), hex(&[0u8; 32]))
    } else {
        (hex(&p.x.into_repr().to_bytes_le()), hex(&p.y.into_repr().to_bytes_le()))
    };
    format!("{{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}}", x, y)
}

/// Reimplementation of `r1cs_nark::hash_matrices` (the crate's is `pub(crate)`),
/// used to dump the golden `nark_matrices_hash` the Python port must reproduce.
fn hash_matrices(domain: &[u8], a: &Matrix<Fr>, b: &Matrix<Fr>, c: &Matrix<Fr>) -> [u8; 32] {
    let mut serialized = domain.to_vec();
    a.serialize(&mut serialized).unwrap();
    b.serialize(&mut serialized).unwrap();
    c.serialize(&mut serialized).unwrap();
    let mut hasher = VarBlake2b::new(32).unwrap();
    hasher.update(&serialized);
    let mut out = [0u8; 32];
    hasher.finalize_variable(|res| out.copy_from_slice(res));
    out
}

fn main() {
    let circuit = DummyCircuit {
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
    let gens_json: Vec<String> = generators.iter().map(point_json).collect();

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

    println!("{{");
    println!("  \"note\": \"NARK zk prove fixtures (zorch#303 slice 6a)\",");
    println!("  \"num_inputs\": {},", input.len());
    println!("  \"num_constraints\": {},", num_constraints);
    println!("  \"nark_matrices_hash_hex\": \"{}\",", hex(&nark_matrices_hash));
    println!("  \"a\": {},", matrix_json(&matrices.a));
    println!("  \"b\": {},", matrix_json(&matrices.b));
    println!("  \"c\": {},", matrix_json(&matrices.c));
    println!("  \"input\": {},", fr_list_json(&input));
    println!("  \"witness\": {},", fr_list_json(&witness));
    println!("  \"generators\": [{}],", gens_json.join(","));
    println!("  \"hiding\": {},", point_json(&hiding));
    println!("  \"r\": {},", fr_list_json(&r));
    println!("  \"a_blinder\": \"{}\",", fr_hex(&a_blinder));
    println!("  \"b_blinder\": \"{}\",", fr_hex(&b_blinder));
    println!("  \"c_blinder\": \"{}\",", fr_hex(&c_blinder));
    println!("  \"r_a_blinder\": \"{}\",", fr_hex(&r_a_blinder));
    println!("  \"r_b_blinder\": \"{}\",", fr_hex(&r_b_blinder));
    println!("  \"r_c_blinder\": \"{}\",", fr_hex(&r_c_blinder));
    println!("  \"blinder_1\": \"{}\",", fr_hex(&blinder_1));
    println!("  \"blinder_2\": \"{}\",", fr_hex(&blinder_2));
    println!("  \"proof_hex\": \"{}\"", proof_hex);
    println!("}}");
}
