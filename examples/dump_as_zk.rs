//! R1CS-NARK-AS prove (zk) end-to-end fixtures for the frx port, over either
//! Pasta cycle curve (Pallas or Vesta) — the zk acceptance criterion. Mirrors
//! `oracle.rs`'s `prove_byte_identical_to_arkworks_zk` flow (seeds {0, 42},
//! num_inputs=5, num_constraints=10) so the golden bytes are the oracle's.
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
use ark_ff::{BigInteger, Field, PrimeField, UniformRand, Zero};
use ark_poly_commit::trivial_pc::PedersenCommitment;
use ark_relations::lc;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, Matrix, OptimizationGoal,
    SynthesisError, SynthesisMode,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_sponge::{Absorbable, CryptographicSponge};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use blake2::VarBlake2b;
use digest::{Update, VariableOutput};

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

#[derive(Clone)]
struct DummyCircuit<F: Field> {
    a: Option<F>,
    b: Option<F>,
    num_inputs: usize,
    num_constraints: usize,
}

impl<F: Field> ConstraintSynthesizer<F> for DummyCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
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

fn fe_hex<F: PrimeField>(f: &F) -> String {
    hex(&f.into_repr().to_bytes_le())
}

fn ser_hex<T: CanonicalSerialize>(v: &T) -> String {
    let mut b = Vec::new();
    v.serialize(&mut b).unwrap();
    hex(&b)
}

fn fr_list_json<F: PrimeField>(xs: &[F]) -> String {
    let v: Vec<String> = xs.iter().map(|f| format!("\"{}\"", fe_hex(f))).collect();
    format!("[{}]", v.join(","))
}

fn matrix_json<F: PrimeField>(m: &Matrix<F>) -> String {
    let rows: Vec<String> = m
        .iter()
        .map(|row| {
            let entries: Vec<String> = row
                .iter()
                .map(|(coeff, idx)| format!("[\"{}\",{}]", fe_hex(coeff), idx))
                .collect();
            format!("[{}]", entries.join(","))
        })
        .collect();
    format!("[{}]", rows.join(","))
}

fn point_json<P: SWModelParameters>(p: &GroupAffine<P>) -> String
where
    P::BaseField: PrimeField,
{
    let (x, y) = if p.is_zero() {
        (hex(&[0u8; 32]), hex(&[0u8; 32]))
    } else {
        (hex(&p.x.into_repr().to_bytes_le()), hex(&p.y.into_repr().to_bytes_le()))
    };
    format!("{{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}}", x, y)
}

fn hash_matrices<F: PrimeField>(domain: &[u8], a: &Matrix<F>, b: &Matrix<F>, c: &Matrix<F>) -> [u8; 32] {
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

/// One seeded zk accumulation step (mirrors `oracle.rs`'s `prove_bytes!`), the
/// replayed randomness, and the arkworks decider verdict on the produced
/// accumulator (asserted `true`). Returns a JSON object for the seed.
fn run_seed<P>(seed: u64) -> String
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

    format!(
        "{{\"seed\":{},\"r1cs_input\":{},\"witness\":{},\
         \"r\":{},\"a_blinder\":\"{}\",\"b_blinder\":\"{}\",\"c_blinder\":\"{}\",\
         \"r_a_blinder\":\"{}\",\"r_b_blinder\":\"{}\",\"r_c_blinder\":\"{}\",\
         \"blinder_1\":\"{}\",\"blinder_2\":\"{}\",\
         \"as_r1cs_r_input\":\"{}\",\"as_r1cs_r_witness\":\"{}\",\
         \"as_rand_1\":\"{}\",\"as_rand_2\":\"{}\",\"as_rand_3\":\"{}\",\
         \"hp_hiding_a\":\"{}\",\"hp_hiding_b\":\"{}\",\
         \"hp_rand_1\":\"{}\",\"hp_rand_2\":\"{}\",\"hp_rand_3\":\"{}\",\
         \"acc_instance_hex\":\"{}\",\"acc_witness_hex\":\"{}\",\"proof_hex\":\"{}\",\
         \"decide\":{}}}",
        seed,
        fr_list_json(&r1cs_input),
        fr_list_json(&witness),
        fr_list_json(&r),
        fe_hex(&nark_blinders[0]), fe_hex(&nark_blinders[1]), fe_hex(&nark_blinders[2]),
        fe_hex(&nark_blinders[3]), fe_hex(&nark_blinders[4]), fe_hex(&nark_blinders[5]),
        fe_hex(&nark_blinders[6]), fe_hex(&nark_blinders[7]),
        fe_hex(&as_r1cs_r_input), fe_hex(&as_r1cs_r_witness),
        fe_hex(&as_rand[0]), fe_hex(&as_rand[1]), fe_hex(&as_rand[2]),
        fe_hex(&hp_hiding_a), fe_hex(&hp_hiding_b),
        fe_hex(&hp_rand[0]), fe_hex(&hp_rand[1]), fe_hex(&hp_rand[2]),
        ser_hex(&accumulator.instance), ser_hex(&accumulator.witness), ser_hex(&proof),
        decided,
    )
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
    let gens_json: Vec<String> = generators.iter().map(point_json::<P>).collect();
    let seeds_json: Vec<String> = SEEDS.iter().map(|&s| run_seed::<P>(s)).collect();

    println!("{{");
    println!("  \"note\": \"R1CS-NARK-AS zk prove + decide fixtures ({} curve)\",", curve);
    println!("  \"curve\": \"{}\",", curve);
    println!("  \"num_inputs\": {},", NUM_INPUTS);
    println!("  \"num_constraints\": {},", num_constraints);
    println!("  \"supported_num_elems\": {},", supported_num_elems);
    println!("  \"nark_matrices_hash_hex\": \"{}\",", hex(&nark_matrices_hash));
    println!("  \"as_matrices_hash_hex\": \"{}\",", hex(&as_matrices_hash));
    println!("  \"a\": {},", matrix_json(&matrices.a));
    println!("  \"b\": {},", matrix_json(&matrices.b));
    println!("  \"c\": {},", matrix_json(&matrices.c));
    println!("  \"generators\": [{}],", gens_json.join(","));
    println!("  \"hiding\": {},", point_json::<P>(&hiding));
    println!("  \"seeds\": [{}]", seeds_json.join(","));
    println!("}}");
}

fn main() {
    match std::env::args().nth(1).as_deref().unwrap_or("pallas") {
        "pallas" => dump::<ark_pallas::PallasParameters>("pallas"),
        "vesta" => dump::<ark_vesta::VestaParameters>("vesta"),
        other => panic!("unknown curve {} (expected pallas|vesta)", other),
    }
}
