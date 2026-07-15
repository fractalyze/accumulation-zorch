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
use ark_ff::{Field, PrimeField};
use ark_poly_commit::trivial_pc::PedersenCommitment;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, OptimizationGoal, SynthesisMode,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_sponge::Absorbable;
use serde::Serialize;

use ark_accumulation::r1cs_nark_as::r1cs_nark::R1CSNark;
use fixture_json::{curve_main, fe_list, matrix_json, point_list, ser_hex, DummyCircuit, MatrixJson, PointJson};

/// `ConstraintF<G>` (the sponge / constraint field), re-derived: it is
/// `pub(crate)` upstream. For the Pasta curves the base field is already prime,
/// so this is just the base field.
type CF<P> = <<P as ModelParameters>::BaseField as Field>::BasePrimeField;
type Sponge<P> = ark_sponge::poseidon::PoseidonSponge<CF<P>>;

const NUM_INPUTS: usize = 5;
const NUM_CONSTRAINTS: usize = 10;

/// The fixture schema. Field order is the emitted key order.
#[derive(Serialize)]
struct NarkFixture {
    note: String,
    curve: String,
    num_inputs: usize,
    num_constraints: usize,
    a: MatrixJson,
    b: MatrixJson,
    c: MatrixJson,
    input: Vec<String>,
    witness: Vec<String>,
    generators: Vec<PointJson>,
    proof_hex: String,
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
    // The first-round commitments are the leading 3×33B of `proof_hex`; the
    // Python test slices them out for per-commitment anchoring.

    let fixture = NarkFixture {
        note: format!("NARK no-zk prove fixtures ({} curve)", curve),
        curve: curve.to_string(),
        num_inputs: input.len(),
        num_constraints,
        a: matrix_json(&matrices.a),
        b: matrix_json(&matrices.b),
        c: matrix_json(&matrices.c),
        input: fe_list(&input),
        witness: fe_list(&witness),
        generators: point_list(&generators),
        proof_hex,
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}

curve_main!(dump);
