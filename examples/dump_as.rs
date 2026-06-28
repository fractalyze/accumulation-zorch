//! R1CS-NARK-AS prove (no-zk) end-to-end fixtures for the jax port, over either
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
//! decider recomputes `comm_{a,b,c}` (the size-`n` MSMs the jax decide port
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

/// `a · b = c`, padded to `NUM_INPUTS` public inputs and `NUM_CONSTRAINTS`
/// repeats of the multiplication constraint — the circuit `oracle.rs` drives.
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

/// Canonical-LE 32-byte hex of a field element.
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

/// `[[coeff_le_hex, var_index], ...]` per row — the sparse `Matrix<Fr>` layout.
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

/// One seeded no-zk accumulation step, mirroring `oracle.rs`'s `prove_bytes!`,
/// followed by the arkworks decider on the produced accumulator. Returns the
/// replay inputs, the golden serialized accumulator + proof, and the decider's
/// verdict (asserted `true` by the caller).
fn run_seed<P>(seed: u64) -> (Vec<P::ScalarField>, Vec<P::ScalarField>, String, String, String, bool)
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
    // they equal the accumulator's stored commitments. The jax decide port
    // reproduces this; the produced fixture is valid only if arkworks accepts.
    let decided = AS::<P>::decide(&dk, accumulator.as_ref(), None::<Sponge<P>>).unwrap();
    assert!(decided, "arkworks decider must accept the produced accumulator (seed {})", seed);

    (
        r1cs_input,
        blinded_witness,
        ser_hex(&accumulator.instance),
        ser_hex(&accumulator.witness),
        ser_hex(&proof),
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
    let gens_json: Vec<String> = generators.iter().map(point_json::<P>).collect();

    let seeds_json: Vec<String> = SEEDS
        .iter()
        .map(|&seed| {
            let (r1cs_input, blinded_witness, acc_inst, acc_wit, proof, decided) =
                run_seed::<P>(seed);
            format!(
                "{{\"seed\":{},\"r1cs_input\":{},\"blinded_witness\":{},\
                 \"acc_instance_hex\":\"{}\",\"acc_witness_hex\":\"{}\",\"proof_hex\":\"{}\",\
                 \"decide\":{}}}",
                seed,
                fr_list_json(&r1cs_input),
                fr_list_json(&blinded_witness),
                acc_inst,
                acc_wit,
                proof,
                decided,
            )
        })
        .collect();

    println!("{{");
    println!("  \"note\": \"R1CS-NARK-AS no-zk prove + decide fixtures ({} curve)\",", curve);
    println!("  \"curve\": \"{}\",", curve);
    println!("  \"num_inputs\": {},", NUM_INPUTS);
    println!("  \"num_constraints\": {},", num_constraints);
    println!("  \"supported_num_elems\": {},", supported_num_elems);
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
