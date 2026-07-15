//! Multi-addend fold fixtures for the frx port: the full IVC
//! fold — one input folded INTO one prior accumulator — the `num_addends = 3`
//! path (`beta = [1, c₁, c₂]`) the current single-input frx (`r1cs_nark_as.
//! _build_zk_core`, num_addends=2) does not yet cover.
//!
//! Per seed it runs TWO proves on a fixed toy circuit (num_inputs=5,
//! num_constraints=10), each with its own seeded rng so the replay is local:
//!   1. **acc_prev** — init-accumulate input₁ (no old accumulators); its instance
//!      + witness are dumped as serialized bytes (the frx fold parses them; their
//!      `pub(crate)` components aren't reachable from this external example).
//!   2. **fold** — `AS::prove([input₂], [acc_prev])` → the golden folded
//!      accumulator + proof. The golden is checked with `AS::verify` here, so a
//!      green dump is itself a validated num_addends=3 fold.
//!
//! Replayed randomness (the frx re-derives input₂'s NARK + the fold's AS/HP
//! commitments from these, not from arkworks' rng):
//!   * input₂ NARK (seed^0x5ec2): `r` × num_witness, then the 8 blinders.
//!   * fold AS `generate_prover_randomness` (seed^0xf01d): `r1cs_r_input`,
//!     `r1cs_r_witness` (`vec![rand;n]`, one draw each), `rand_1/2/3`; then HP
//!     `generate_prover_randomness`: `hiding_a`, `hiding_b`, `rand_1/2/3`.
//! input₁/acc_prev's own randomness is internal (acc_prev is dumped as output),
//! so it is not replayed. The two inputs use distinct fixed assignments (3·5 vs
//! 7·11) so the fold is non-degenerate.
//!
//! Run: `cargo run --example dump_as_fold_zk > python/testdata/as_fold_zk_fixtures.json`

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
const NUM_CONSTRAINTS: usize = 10;
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

/// Parse a serialized `AccumulatorInstance` into its components and render them
/// as structured JSON (points uncompressed as `{x_le_hex,y_le_hex}`, so the frx
/// side needs no point decompression). The serialization is the derived
/// field-order one: `r1cs_input: Vec<Fr>`, `comm_a/comm_b/comm_c: G`, then
/// `hp_instance: HPInputInstance` = `comm_1/comm_2/comm_3: G`.
fn acc_instance_json(bytes: &[u8]) -> String {
    let mut cur = bytes;
    let r1cs_input = Vec::<Fr>::deserialize(&mut cur).unwrap();
    let comm_a = Affine::deserialize(&mut cur).unwrap();
    let comm_b = Affine::deserialize(&mut cur).unwrap();
    let comm_c = Affine::deserialize(&mut cur).unwrap();
    let hp_comm_1 = Affine::deserialize(&mut cur).unwrap();
    let hp_comm_2 = Affine::deserialize(&mut cur).unwrap();
    let hp_comm_3 = Affine::deserialize(&mut cur).unwrap();
    format!(
        "{{\"r1cs_input\":{},\"comm_a\":{},\"comm_b\":{},\"comm_c\":{},\
         \"hp_comm_1\":{},\"hp_comm_2\":{},\"hp_comm_3\":{}}}",
        fr_list_json(&r1cs_input),
        point_json(&comm_a),
        point_json(&comm_b),
        point_json(&comm_c),
        point_json(&hp_comm_1),
        point_json(&hp_comm_2),
        point_json(&hp_comm_3),
    )
}

/// Parse a serialized `AccumulatorWitness` into its components. Derived
/// field-order serialization: `r1cs_blinded_witness: Vec<Fr>`, then `hp_witness`
/// (`hp_as::InputWitness` = `a_vec: Vec<Fr>`, `b_vec: Vec<Fr>`, `randomness:
/// Option<{rand_1,rand_2,rand_3}>`), then `randomness: Option<{sigma_a,sigma_b,
/// sigma_c}>`. An `Option` is a `u8` flag (1=Some) then the value; zk forces both
/// `Some`. The HP fields (`a_vec`/`b_vec`/`hp_rand`) feed the HP-level fold; the
/// `r1cs_blinded_witness` + sigmas feed the AS-level witness combine.
fn acc_witness_json(bytes: &[u8]) -> String {
    let mut cur = bytes;
    let r1cs_blinded_witness = Vec::<Fr>::deserialize(&mut cur).unwrap();
    let hp_a_vec = Vec::<Fr>::deserialize(&mut cur).unwrap();
    let hp_b_vec = Vec::<Fr>::deserialize(&mut cur).unwrap();
    let read_opt3 = |cur: &mut &[u8]| -> (Fr, Fr, Fr) {
        let flag = u8::deserialize(&mut *cur).unwrap();
        if flag == 1 {
            (
                Fr::deserialize(&mut *cur).unwrap(),
                Fr::deserialize(&mut *cur).unwrap(),
                Fr::deserialize(&mut *cur).unwrap(),
            )
        } else {
            (Fr::from(0u64), Fr::from(0u64), Fr::from(0u64))
        }
    };
    let (hp_rand_1, hp_rand_2, hp_rand_3) = read_opt3(&mut cur);
    let (sigma_a, sigma_b, sigma_c) = read_opt3(&mut cur);
    format!(
        "{{\"r1cs_blinded_witness\":{},\"hp_a_vec\":{},\"hp_b_vec\":{},\
         \"hp_rand_1\":\"{}\",\"hp_rand_2\":\"{}\",\"hp_rand_3\":\"{}\",\
         \"sigma_a\":\"{}\",\"sigma_b\":\"{}\",\"sigma_c\":\"{}\"}}",
        fr_list_json(&r1cs_blinded_witness),
        fr_list_json(&hp_a_vec),
        fr_list_json(&hp_b_vec),
        fr_hex(&hp_rand_1),
        fr_hex(&hp_rand_2),
        fr_hex(&hp_rand_3),
        fr_hex(&sigma_a),
        fr_hex(&sigma_b),
        fr_hex(&sigma_c),
    )
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

/// The instance + witness assignment of one fixed `DummyCircuit`.
fn assignment(a: u64, b: u64) -> (Vec<Fr>, Vec<Fr>) {
    let circuit = DummyCircuit {
        a: Some(Fr::from(a)),
        b: Some(Fr::from(b)),
        num_inputs: NUM_INPUTS,
        num_constraints: NUM_CONSTRAINTS,
    };
    let pcs = ConstraintSystem::<Fr>::new_ref();
    pcs.set_optimization_goal(OptimizationGoal::Constraints);
    pcs.set_mode(SynthesisMode::Prove { construct_matrices: false });
    circuit.generate_constraints(pcs.clone()).unwrap();
    pcs.finalize();
    let cs = pcs.borrow().unwrap();
    (cs.instance_assignment.clone(), cs.witness_assignment.clone())
}

/// One seeded zk fold step: build `acc_prev` (init-accumulate input₁), then fold
/// input₂ into it (`num_addends = 3`). Returns the JSON object for the seed.
fn run_fold_seed(seed: u64) -> String {
    let nark_pp = R1CSNark::<G, Sponge>::setup();
    let index_circuit = DummyCircuit {
        a: Some(Fr::from(2u64)),
        b: Some(Fr::from(2u64)),
        num_inputs: NUM_INPUTS,
        num_constraints: NUM_CONSTRAINTS,
    };
    let (ipk, ivk) = R1CSNark::<G, Sponge>::index(&nark_pp, index_circuit).unwrap();
    let mut setup_rng = StdRng::seed_from_u64(seed ^ 0x5e7);
    let as_pp = AS::setup(&mut setup_rng).unwrap();
    let (pk, vk, _dk) = AS::index(&as_pp, &(), &(ipk.clone(), ivk.clone())).unwrap();

    // --- acc_prev: init-accumulate input₁ (a=3,b=5), no old accumulators.
    let mut rng1 = StdRng::seed_from_u64(seed ^ 0xacc0);
    let circuit1 = DummyCircuit {
        a: Some(Fr::from(3u64)),
        b: Some(Fr::from(5u64)),
        num_inputs: NUM_INPUTS,
        num_constraints: NUM_CONSTRAINTS,
    };
    let nark_sponge1 = ASForR1CSNark::<G, Sponge>::nark_sponge(&Sponge::new());
    let nark_proof1 =
        R1CSNark::<G, Sponge>::prove(&ipk, circuit1, true, Some(nark_sponge1), Some(&mut rng1))
            .unwrap();
    let (r1cs_input1, _w1) = assignment(3, 5);
    let input1 = Input::<CF, Sponge, AS> {
        instance: InputInstance {
            r1cs_input: r1cs_input1,
            first_round_message: nark_proof1.first_msg,
        },
        witness: nark_proof1.second_msg,
    };
    let inputs1 = vec![input1];
    let no_acc: Vec<Accumulator<CF, Sponge, AS>> = Vec::new();
    let (acc_prev, _proof1) = AS::prove(
        &pk,
        Input::<CF, Sponge, AS>::map_to_refs(&inputs1),
        Accumulator::<CF, Sponge, AS>::map_to_refs(&no_acc),
        MakeZK::Enabled(&mut rng1),
        None,
    )
    .unwrap();

    // --- input₂ (a=7,b=11): the input folded into acc_prev.
    let mut rng2 = StdRng::seed_from_u64(seed ^ 0x5ec2);
    let circuit2 = DummyCircuit {
        a: Some(Fr::from(7u64)),
        b: Some(Fr::from(11u64)),
        num_inputs: NUM_INPUTS,
        num_constraints: NUM_CONSTRAINTS,
    };
    let nark_sponge2 = ASForR1CSNark::<G, Sponge>::nark_sponge(&Sponge::new());
    let nark_proof2 =
        R1CSNark::<G, Sponge>::prove(&ipk, circuit2, true, Some(nark_sponge2), Some(&mut rng2))
            .unwrap();
    let (r1cs_input2, witness2) = assignment(7, 11);
    let num_witness = witness2.len();
    let input2 = Input::<CF, Sponge, AS> {
        instance: InputInstance {
            r1cs_input: r1cs_input2.clone(),
            first_round_message: nark_proof2.first_msg.clone(),
        },
        witness: nark_proof2.second_msg,
    };

    // Capture acc_prev before it is moved into the fold call. Its `pub(crate)`
    // components aren't reachable here, so parse the serialized instance into
    // structured fields (`acc_instance_json`); the witness stays serialized hex
    // (parsed when the witness combine lands).
    let acc_prev_instance_bytes = {
        let mut b = Vec::new();
        acc_prev.instance.serialize(&mut b).unwrap();
        b
    };
    let acc_prev_instance_json = acc_instance_json(&acc_prev_instance_bytes);
    let acc_prev_witness_bytes = {
        let mut b = Vec::new();
        acc_prev.witness.serialize(&mut b).unwrap();
        b
    };
    let acc_prev_witness_json = acc_witness_json(&acc_prev_witness_bytes);

    // --- fold input₂ + acc_prev → golden (num_addends = 3).
    let mut rng_fold = StdRng::seed_from_u64(seed ^ 0xf01d);
    let inputs2 = vec![input2];
    let accs_prev = vec![acc_prev];
    let (golden_acc, golden_proof) = AS::prove(
        &pk,
        Input::<CF, Sponge, AS>::map_to_refs(&inputs2),
        Accumulator::<CF, Sponge, AS>::map_to_refs(&accs_prev),
        MakeZK::Enabled(&mut rng_fold),
        None,
    )
    .unwrap();

    let golden_instance_bytes = {
        let mut b = Vec::new();
        golden_acc.instance.serialize(&mut b).unwrap();
        b
    };
    let golden_instance_json = acc_instance_json(&golden_instance_bytes);
    let golden_witness_bytes = {
        let mut b = Vec::new();
        golden_acc.witness.serialize(&mut b).unwrap();
        b
    };
    let golden_witness_json = acc_witness_json(&golden_witness_bytes);

    // Validate the golden IS a correct fold (a green dump is a verified fixture).
    let verified = AS::verify(
        &vk,
        std::iter::once(&inputs2[0].instance),
        std::iter::once(&accs_prev[0].instance),
        &golden_acc.instance,
        &golden_proof,
        None,
    )
    .unwrap();
    assert!(verified, "seed {}: golden fold failed to verify", seed);

    // --- replay input₂ NARK randomness (fresh seed^0x5ec2): r × num_witness, 8
    // blinders (circuit₂'s assignment is fixed, so no harness draws precede them).
    let mut rep2 = StdRng::seed_from_u64(seed ^ 0x5ec2);
    let r2: Vec<Fr> = (0..num_witness).map(|_| Fr::rand(&mut rep2)).collect();
    let nark_blinders: Vec<Fr> = (0..8).map(|_| Fr::rand(&mut rep2)).collect();

    // --- replay the fold's AS + HP randomness (fresh seed^0xf01d).
    let mut rep_fold = StdRng::seed_from_u64(seed ^ 0xf01d);
    let as_r1cs_r_input = Fr::rand(&mut rep_fold);
    let as_r1cs_r_witness = Fr::rand(&mut rep_fold);
    let as_rand: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rep_fold)).collect();
    let hp_hiding_a = Fr::rand(&mut rep_fold);
    let hp_hiding_b = Fr::rand(&mut rep_fold);
    let hp_rand: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rep_fold)).collect();

    format!(
        "{{\"seed\":{},\
         \"input2_r1cs_input\":{},\"input2_witness\":{},\
         \"r\":{},\"a_blinder\":\"{}\",\"b_blinder\":\"{}\",\"c_blinder\":\"{}\",\
         \"r_a_blinder\":\"{}\",\"r_b_blinder\":\"{}\",\"r_c_blinder\":\"{}\",\
         \"blinder_1\":\"{}\",\"blinder_2\":\"{}\",\
         \"as_r1cs_r_input\":\"{}\",\"as_r1cs_r_witness\":\"{}\",\
         \"as_rand_1\":\"{}\",\"as_rand_2\":\"{}\",\"as_rand_3\":\"{}\",\
         \"hp_hiding_a\":\"{}\",\"hp_hiding_b\":\"{}\",\
         \"hp_rand_1\":\"{}\",\"hp_rand_2\":\"{}\",\"hp_rand_3\":\"{}\",\
         \"acc_prev_instance\":{},\"acc_prev_witness\":{},\
         \"golden_instance\":{},\"golden_witness\":{},\
         \"golden_instance_hex\":\"{}\",\"golden_witness_hex\":\"{}\",\
         \"golden_proof_hex\":\"{}\"}}",
        seed,
        fr_list_json(&r1cs_input2),
        fr_list_json(&witness2),
        fr_list_json(&r2),
        fr_hex(&nark_blinders[0]), fr_hex(&nark_blinders[1]), fr_hex(&nark_blinders[2]),
        fr_hex(&nark_blinders[3]), fr_hex(&nark_blinders[4]), fr_hex(&nark_blinders[5]),
        fr_hex(&nark_blinders[6]), fr_hex(&nark_blinders[7]),
        fr_hex(&as_r1cs_r_input), fr_hex(&as_r1cs_r_witness),
        fr_hex(&as_rand[0]), fr_hex(&as_rand[1]), fr_hex(&as_rand[2]),
        fr_hex(&hp_hiding_a), fr_hex(&hp_hiding_b),
        fr_hex(&hp_rand[0]), fr_hex(&hp_rand[1]), fr_hex(&hp_rand[2]),
        acc_prev_instance_json, acc_prev_witness_json,
        golden_instance_json, golden_witness_json,
        hex(&golden_instance_bytes), hex(&golden_witness_bytes), ser_hex(&golden_proof),
    )
}

fn main() {
    // Seed-independent structural inputs (matrices, committer key, matrix hashes).
    let shape_circuit = DummyCircuit {
        a: None,
        b: None,
        num_inputs: NUM_INPUTS,
        num_constraints: NUM_CONSTRAINTS,
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
    let seeds_json: Vec<String> = SEEDS.iter().map(|&s| run_fold_seed(s)).collect();

    println!("{{");
    println!("  \"note\": \"R1CS-NARK-AS multi-addend fold (num_addends=3) fixtures\",");
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
