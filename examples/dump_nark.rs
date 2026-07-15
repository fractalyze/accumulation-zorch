//! NARK prove (no-zk) fixtures for the frx port, over either
//! Pasta cycle curve (Pallas or Vesta).
//!
//! Drives the crate's real `R1CSNark::prove` (no-zk) over a fixed `DummyCircuit`
//! and dumps the golden serialized `Proof`, plus the inputs the Python port
//! replays: the R1CS matrices (a/b/c), the instance + witness assignments, and
//! the Pedersen committer-key generators. The no-zk proof is a deterministic
//! function of the circuit — it draws no randomness, and `gamma`/`matrices_hash`
//! do not enter the proof bytes (they only matter for the zk path and the AS
//! challenges), so the proof reduces to `commit(z_a/z_b/z_c)` + the raw witness.
//! The sponge type is never instantiated (no-zk passes `None`), so no Poseidon
//! params are needed for either curve.
//!
//! Generic over the curve (`PastaCurve`-style, no per-curve copy) — the curve is
//! a CLI arg, defaulting to Pallas:
//!
//!   cargo run --example dump_nark -- pallas > python/testdata/nark_fixtures.json
//!   cargo run --example dump_nark -- vesta  > python/testdata/nark_vesta_fixtures.json

use ark_ec::models::ModelParameters;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ec::SWModelParameters;
use ark_ff::{BigInteger, Field, PrimeField, Zero};
use ark_poly_commit::trivial_pc::PedersenCommitment;
use ark_relations::lc;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, Matrix, OptimizationGoal,
    SynthesisError, SynthesisMode,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_sponge::Absorbable;

use ark_accumulation::r1cs_nark_as::r1cs_nark::R1CSNark;

/// `ConstraintF<G>` (the sponge / constraint field), re-derived: it is
/// `pub(crate)` upstream. For the Pasta curves the base field is already prime,
/// so this is just the base field.
type CF<P> = <<P as ModelParameters>::BaseField as Field>::BasePrimeField;
type Sponge<P> = ark_sponge::poseidon::PoseidonSponge<CF<P>>;

const NUM_INPUTS: usize = 5;
const NUM_CONSTRAINTS: usize = 10;

/// `a · b = c`, padded to `NUM_INPUTS` public inputs and `NUM_CONSTRAINTS`
/// repeats of the multiplication constraint — the circuit the arkworks
/// `r1cs_nark_as` tests (and `oracle.rs`) drive.
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

fn fr_list_json<F: PrimeField>(xs: &[F]) -> String {
    let v: Vec<String> = xs.iter().map(|f| format!("\"{}\"", fe_hex(f))).collect();
    format!("[{}]", v.join(","))
}

/// x-coordinate of an affine point as canonical-LE 32B hex (identity → zeros).
fn coord_x_hex<P: SWModelParameters>(p: &GroupAffine<P>) -> String
where
    P::BaseField: PrimeField,
{
    if p.is_zero() {
        hex(&[0u8; 32])
    } else {
        hex(&p.x.into_repr().to_bytes_le())
    }
}

fn coord_y_hex<P: SWModelParameters>(p: &GroupAffine<P>) -> String
where
    P::BaseField: PrimeField,
{
    if p.is_zero() {
        hex(&[0u8; 32])
    } else {
        hex(&p.y.into_repr().to_bytes_le())
    }
}

fn dump<P>(curve: &str)
where
    P: SWModelParameters,
    P::BaseField: PrimeField,
    GroupAffine<P>: Absorbable<CF<P>>,
    CF<P>: Absorbable<CF<P>>,
{
    let circuit = DummyCircuit::<P::ScalarField> {
        a: Some(P::ScalarField::from(3u64)),
        b: Some(P::ScalarField::from(5u64)),
        num_inputs: NUM_INPUTS,
        num_constraints: NUM_CONSTRAINTS,
    };

    // Golden proof from the crate's real no-zk prove (sponge irrelevant for
    // no-zk: gamma is computed but unused, so pass None).
    let pp = R1CSNark::<GroupAffine<P>, Sponge<P>>::setup();
    let (ipk, _ivk) = R1CSNark::<GroupAffine<P>, Sponge<P>>::index(&pp, circuit.clone()).unwrap();
    let proof =
        R1CSNark::<GroupAffine<P>, Sponge<P>>::prove(&ipk, circuit.clone(), false, None, None)
            .unwrap();
    let proof_hex = ser_hex(&proof);

    // Matrices a/b/c — extracted exactly as `index` does (Constraints + Setup).
    let mcs = ConstraintSystem::<P::ScalarField>::new_ref();
    mcs.set_optimization_goal(OptimizationGoal::Constraints);
    mcs.set_mode(SynthesisMode::Setup);
    circuit.clone().generate_constraints(mcs.clone()).unwrap();
    mcs.finalize();
    let matrices = mcs.to_matrices().unwrap();

    // Instance + witness assignments — exactly as `prove` extracts them
    // (Constraints + Prove, no matrix construction).
    let pcs = ConstraintSystem::<P::ScalarField>::new_ref();
    pcs.set_optimization_goal(OptimizationGoal::Constraints);
    pcs.set_mode(SynthesisMode::Prove {
        construct_matrices: false,
    });
    circuit.clone().generate_constraints(pcs.clone()).unwrap();
    pcs.finalize();
    let (input, witness, num_constraints) = {
        let cs = pcs.borrow().unwrap();
        (
            cs.instance_assignment.clone(),
            cs.witness_assignment.clone(),
            cs.num_constraints,
        )
    };

    // Committer-key generators (+ hiding) — recovered from the key's
    // uncompressed form, the same way `dump_fixtures.rs` does.
    let cpp = PedersenCommitment::<GroupAffine<P>>::setup(num_constraints);
    let ck = PedersenCommitment::<GroupAffine<P>>::trim(&cpp, num_constraints);
    let generators = {
        let mut b = Vec::new();
        ck.serialize_uncompressed(&mut b).unwrap();
        let mut r = &b[..];
        Vec::<GroupAffine<P>>::deserialize_uncompressed(&mut r).unwrap()
    };
    let gens_json: Vec<String> = generators
        .iter()
        .map(|p| format!("{{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}}", coord_x_hex(p), coord_y_hex(p)))
        .collect();

    // The first-round commitments are the leading 3×33B of `proof_hex`; the
    // Python test slices them out for per-commitment anchoring.

    println!("{{");
    println!("  \"note\": \"NARK no-zk prove fixtures ({} curve)\",", curve);
    println!("  \"curve\": \"{}\",", curve);
    println!("  \"num_inputs\": {},", input.len());
    println!("  \"num_constraints\": {},", num_constraints);
    println!("  \"a\": {},", matrix_json(&matrices.a));
    println!("  \"b\": {},", matrix_json(&matrices.b));
    println!("  \"c\": {},", matrix_json(&matrices.c));
    println!("  \"input\": {},", fr_list_json(&input));
    println!("  \"witness\": {},", fr_list_json(&witness));
    println!("  \"generators\": [{}],", gens_json.join(","));
    println!("  \"proof_hex\": \"{}\"", proof_hex);
    println!("}}");
}

fn main() {
    match std::env::args().nth(1).as_deref().unwrap_or("pallas") {
        "pallas" => dump::<ark_pallas::PallasParameters>("pallas"),
        "vesta" => dump::<ark_vesta::VestaParameters>("vesta"),
        other => panic!("unknown curve {} (expected pallas|vesta)", other),
    }
}
