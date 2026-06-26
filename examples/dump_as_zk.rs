//! Slice-6c R1CS-NARK-AS prove (zk) end-to-end fixtures for the jax port
//! (zorch#303) — the zk acceptance criterion. Mirrors `oracle.rs`'s
//! `prove_byte_identical_to_arkworks_zk` flow (seeds {0, 42}, num_inputs=5,
//! num_constraints=10) so the golden bytes are the oracle's.
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
//! Run: `cargo run --example dump_as_zk > python/testdata/as_zk_fixtures.json`

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
use ark_accumulation::r1cs_nark_as::{ASForR1CSNark, InputInstance};
use ark_accumulation::{AccumulationScheme, Accumulator, Input, MakeZK};

type G = Affine;
type CF = ark_pallas::Fq;
type Sponge = ark_sponge::poseidon::PoseidonSponge<CF>;
type AS = ASForR1CSNark<G, Sponge>;

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

/// One seeded zk accumulation step (mirrors `oracle.rs`'s `prove_bytes!`), plus
/// the replayed randomness. Returns a JSON object for the seed.
fn run_seed(seed: u64) -> String {
    let mut rng = StdRng::seed_from_u64(seed);

    let nark_pp = R1CSNark::<G, Sponge>::setup();
    let index_circuit = DummyCircuit {
        a: Some(Fr::rand(&mut rng)),
        b: Some(Fr::rand(&mut rng)),
        num_inputs: NUM_INPUTS,
        num_constraints: cfg_num_constraints(),
    };
    let (ipk, ivk) = R1CSNark::<G, Sponge>::index(&nark_pp, index_circuit).unwrap();

    let as_pp = AS::setup(&mut rng).unwrap();
    let (pk, _vk, _dk) = AS::index(&as_pp, &(), &(ipk.clone(), ivk.clone())).unwrap();

    let circuit = DummyCircuit {
        a: Some(Fr::rand(&mut rng)),
        b: Some(Fr::rand(&mut rng)),
        num_inputs: NUM_INPUTS,
        num_constraints: cfg_num_constraints(),
    };
    let _bench_t0 = std::time::Instant::now();
    let nark_sponge = ASForR1CSNark::<G, Sponge>::nark_sponge(&Sponge::new());
    let nark_proof = R1CSNark::<G, Sponge>::prove(
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

    let input = Input::<CF, Sponge, AS> {
        instance: InputInstance {
            r1cs_input: r1cs_input.clone(),
            first_round_message: nark_proof.first_msg.clone(),
        },
        witness: nark_proof.second_msg,
    };
    let inputs = vec![input];
    let no_accumulators: Vec<Accumulator<CF, Sponge, AS>> = Vec::new();

    let (accumulator, proof) = AS::prove(
        &pk,
        Input::<CF, Sponge, AS>::map_to_refs(&inputs),
        Accumulator::<CF, Sponge, AS>::map_to_refs(&no_accumulators),
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

    // Replay the full draw schedule on a fresh same-seed rng: 4 harness draws
    // (index/circuit a,b — discarded), then NARK, AS-gen, HP-gen.
    let mut rep = StdRng::seed_from_u64(seed);
    for _ in 0..4 {
        let _ = Fr::rand(&mut rep);
    }
    let r: Vec<Fr> = (0..num_witness).map(|_| Fr::rand(&mut rep)).collect();
    let nark_blinders: Vec<Fr> = (0..8).map(|_| Fr::rand(&mut rep)).collect();
    let as_r1cs_r_input = Fr::rand(&mut rep);
    let as_r1cs_r_witness = Fr::rand(&mut rep);
    let as_rand: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rep)).collect();
    let hp_hiding_a = Fr::rand(&mut rep);
    let hp_hiding_b = Fr::rand(&mut rep);
    let hp_rand: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rep)).collect();

    format!(
        "{{\"seed\":{},\"r1cs_input\":{},\"witness\":{},\
         \"r\":{},\"a_blinder\":\"{}\",\"b_blinder\":\"{}\",\"c_blinder\":\"{}\",\
         \"r_a_blinder\":\"{}\",\"r_b_blinder\":\"{}\",\"r_c_blinder\":\"{}\",\
         \"blinder_1\":\"{}\",\"blinder_2\":\"{}\",\
         \"as_r1cs_r_input\":\"{}\",\"as_r1cs_r_witness\":\"{}\",\
         \"as_rand_1\":\"{}\",\"as_rand_2\":\"{}\",\"as_rand_3\":\"{}\",\
         \"hp_hiding_a\":\"{}\",\"hp_hiding_b\":\"{}\",\
         \"hp_rand_1\":\"{}\",\"hp_rand_2\":\"{}\",\"hp_rand_3\":\"{}\",\
         \"acc_instance_hex\":\"{}\",\"acc_witness_hex\":\"{}\",\"proof_hex\":\"{}\"}}",
        seed,
        fr_list_json(&r1cs_input),
        fr_list_json(&witness),
        fr_list_json(&r),
        fr_hex(&nark_blinders[0]), fr_hex(&nark_blinders[1]), fr_hex(&nark_blinders[2]),
        fr_hex(&nark_blinders[3]), fr_hex(&nark_blinders[4]), fr_hex(&nark_blinders[5]),
        fr_hex(&nark_blinders[6]), fr_hex(&nark_blinders[7]),
        fr_hex(&as_r1cs_r_input), fr_hex(&as_r1cs_r_witness),
        fr_hex(&as_rand[0]), fr_hex(&as_rand[1]), fr_hex(&as_rand[2]),
        fr_hex(&hp_hiding_a), fr_hex(&hp_hiding_b),
        fr_hex(&hp_rand[0]), fr_hex(&hp_rand[1]), fr_hex(&hp_rand[2]),
        ser_hex(&accumulator.instance), ser_hex(&accumulator.witness), ser_hex(&proof),
    )
}

fn main() {
    // Seed-independent structural inputs (matrices, committer key, matrix hashes).
    let shape_circuit = DummyCircuit {
        a: None,
        b: None,
        num_inputs: NUM_INPUTS,
        num_constraints: cfg_num_constraints(),
    };
    let mcs = ConstraintSystem::<Fr>::new_ref();
    mcs.set_optimization_goal(OptimizationGoal::Constraints);
    mcs.set_mode(SynthesisMode::Setup);
    shape_circuit.generate_constraints(mcs.clone()).unwrap();
    mcs.finalize();
    let matrices = mcs.to_matrices().unwrap();
    let num_constraints = mcs.num_constraints();
    let nark_matrices_hash = hash_matrices(NARK_PROTOCOL_NAME, &matrices.a, &matrices.b, &matrices.c);
    let as_matrices_hash = hash_matrices(AS_PROTOCOL_NAME, &matrices.a, &matrices.b, &matrices.c);

    let cpp = PedersenCommitment::<Affine>::setup(num_constraints);
    let ck = PedersenCommitment::<Affine>::trim(&cpp, num_constraints);
    let supported_num_elems = ck.supported_num_elems();
    let (generators, hiding) = {
        let mut b = Vec::new();
        ck.serialize_uncompressed(&mut b).unwrap();
        let mut r = &b[..];
        let g = Vec::<Affine>::deserialize_uncompressed(&mut r).unwrap();
        let h = Affine::deserialize_uncompressed(&mut r).unwrap();
        (g, h)
    };
    let gens_json: Vec<String> = generators.iter().map(point_json).collect();
    let seeds_json: Vec<String> = SEEDS.iter().map(|&s| run_seed(s)).collect();

    println!("{{");
    println!("  \"note\": \"R1CS-NARK-AS zk prove fixtures (zorch#303 slice 6c)\",");
    println!("  \"num_inputs\": {},", NUM_INPUTS);
    println!("  \"num_constraints\": {},", num_constraints);
    println!("  \"supported_num_elems\": {},", supported_num_elems);
    println!("  \"nark_matrices_hash_hex\": \"{}\",", hex(&nark_matrices_hash));
    println!("  \"as_matrices_hash_hex\": \"{}\",", hex(&as_matrices_hash));
    println!("  \"a\": {},", matrix_json(&matrices.a));
    println!("  \"b\": {},", matrix_json(&matrices.b));
    println!("  \"c\": {},", matrix_json(&matrices.c));
    println!("  \"generators\": [{}],", gens_json.join(","));
    println!("  \"hiding\": {},", point_json(&hiding));
    println!("  \"seeds\": [{}]", seeds_json.join(","));
    println!("}}");
}
