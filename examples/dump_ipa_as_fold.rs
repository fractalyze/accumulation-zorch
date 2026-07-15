//! IPA-PC accumulation **fold** (no-zk) fixtures for the frx port: one input
//! folded INTO a prior accumulator (`old_accumulators = [acc_prev]`), over either
//! Pasta cycle curve (Pallas or Vesta). The fold twin of `dump_ipa_as.rs` (which
//! accumulates inputs with NO old accumulators).
//!
//! Two rounds, each driving the unmodified arkworks
//! `AtomicASForInnerProductArgPC::prove` (mirrors `dump_as_fold_zk.rs`):
//!   1. **acc_prev** — `AS::prove([in0, in1], [])`: the existing no-fold prove over
//!      `NUM_PREV_INPUTS` freshly committed+opened inputs. Its instance is the
//!      prior accumulator the fold extends.
//!   2. **fold** — `AS::prove([in_new], [acc_prev])` → the golden folded
//!      accumulator. The fold's succinct-check list is `[in_new, acc_prev]`
//!      (inputs first, then accumulators, per `succinct_check_inputs_and_accumulators`).
//!
//! The crux of the fold: an accumulator IS an `InputInstance` (commitment, point,
//! evaluation, IPA proof) of the same shape as an input, so it is succinct-checked
//! and combined exactly like one — the only new structure over `dump_ipa_as.rs` is
//! the prior accumulator appended as a second addend. The frx port replays it with
//! the same combine + `IpaPC::open` machinery, fed `[in_new, acc_prev]`.
//!
//! No-zk only (`MakeZK::Disabled`): both `acc_prev` and the golden fold are
//! non-hiding (`hiding_comm`/`rand` = `None`), so every succinct check is the no-zk
//! path. The Poseidon ARK constants are loaded by the test from the per-curve
//! sponge fixture, the same as the other byte-match tests.
//!
//! Generic over the curve (CLI arg, defaulting to Pallas):
//!
//!   cargo run --example dump_ipa_as_fold -- pallas > python/testdata/ipa_as_fold_fixtures.json
//!   cargo run --example dump_ipa_as_fold -- vesta  > python/testdata/ipa_as_fold_vesta_fixtures.json

use ark_ec::models::ModelParameters;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ec::SWModelParameters;
use ark_ff::{Field, One, PrimeField, UniformRand};
use ark_poly::univariate::DensePolynomial;
use ark_poly::UVPolynomial;
use ark_poly_commit::ipa_pc::InnerProductArgPC;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment, PolynomialLabel};
use ark_sponge::domain_separated::DomainSeparatedSponge;
use ark_sponge::poseidon::PoseidonSponge;
use ark_sponge::Absorbable;
use ark_std::test_rng;
use serde::Serialize;

use fixture_json::{curve_main, fe_hex, fe_list, point_list, PointJson};

use ark_accumulation::ipa_pc_as::{
    AtomicASForInnerProductArgPC, InputInstance, IpaPCDomain, PredicateIndex,
};
use ark_accumulation::{AccumulationScheme, Accumulator, InstanceWitnessPair, MakeZK};

/// `ConstraintF<G>` (the sponge / constraint field), re-derived (it is
/// `pub(crate)` upstream). For the Pasta curves the base field is prime.
type CF<P> = <<P as ModelParameters>::BaseField as Field>::BasePrimeField;

type S<P> = PoseidonSponge<CF<P>>;
type AS<P> = AtomicASForInnerProductArgPC<GroupAffine<P>, S<P>>;

/// The exact IPA-PC instantiation `ipa_pc_as` builds on (matches `dump_ipa_as.rs`).
type IpaPC<P> = InnerProductArgPC<
    GroupAffine<P>,
    blake2::Blake2s,
    DensePolynomial<<P as ModelParameters>::ScalarField>,
    CF<P>,
    DomainSeparatedSponge<CF<P>, S<P>, IpaPCDomain>,
>;

/// Degree of each committed input polynomial (`d + 1 = 8` ⇒ `log_d = 3` IPA
/// rounds), and how many inputs `acc_prev` accumulates (two, so its own combine is
/// non-trivial). The fold then folds ONE new input into `acc_prev`.
const DEGREE: usize = 7;
const NUM_PREV_INPUTS: usize = 2;

/// An `InputInstance` (an input or the prior/new accumulator): the IPA
/// commitment, opening point + evaluation, and the proof's fold commitments,
/// `final_comm_key`, and final coefficient `c`. No-zk, so the proof is non-hiding.
/// Field order is the fixture's key order.
#[derive(Serialize)]
struct InstanceJson {
    commitment: PointJson,
    point: String,
    evaluation: String,
    l_vec: Vec<PointJson>,
    r_vec: Vec<PointJson>,
    final_comm_key: PointJson,
    c: String,
}

impl InstanceJson {
    fn from_instance<P: SWModelParameters>(inst: &InputInstance<GroupAffine<P>>) -> Self
    where
        P::BaseField: PrimeField,
    {
        InstanceJson {
            commitment: PointJson::from_affine(&inst.ipa_commitment.commitment().comm),
            point: fe_hex(&inst.point),
            evaluation: fe_hex(&inst.evaluation),
            l_vec: point_list(&inst.ipa_proof.l_vec),
            r_vec: point_list(&inst.ipa_proof.r_vec),
            final_comm_key: PointJson::from_affine(&inst.ipa_proof.final_comm_key),
            c: fe_hex(&inst.ipa_proof.c),
        }
    }
}

/// The whole fixture. Field order is the fixture's key order.
#[derive(Serialize)]
struct FoldFixture {
    note: String,
    curve: String,
    supported_degree: usize,
    num_prev_inputs: usize,
    h: PointJson,
    s: PointJson,
    generators: Vec<PointJson>,
    input: InstanceJson,
    acc_prev: InstanceJson,
    accumulator: InstanceJson,
    decider_coeffs: Vec<String>,
}

/// Commit a fresh random degree-`DEGREE` polynomial and open it at a random point
/// — one no-zk IPA input, exactly as `ipa_pc_as`'s own `generate_inputs` does.
fn make_input<P>(
    ck: &ark_poly_commit::ipa_pc::CommitterKey<GroupAffine<P>>,
    rng: &mut impl ark_std::rand::RngCore,
) -> InstanceWitnessPair<InputInstance<GroupAffine<P>>, ()>
where
    P: SWModelParameters,
    P::BaseField: PrimeField,
    GroupAffine<P>: Absorbable<CF<P>>,
    CF<P>: PrimeField + Absorbable<CF<P>>,
{
    let labeled_poly = LabeledPolynomial::new(
        PolynomialLabel::new(),
        DensePolynomial::<P::ScalarField>::rand(DEGREE, rng),
        None,
        None,
    );
    let (labeled_commitments, randoms) =
        IpaPC::<P>::commit(ck, &[labeled_poly.clone()], None).unwrap();
    let labeled_commitment = labeled_commitments.into_iter().next().unwrap();
    let point = P::ScalarField::rand(rng);
    let evaluation = labeled_poly.evaluate(&point);
    let ipa_proof = IpaPC::<P>::open_individual_opening_challenges(
        ck,
        vec![&labeled_poly],
        vec![&labeled_commitment],
        &point,
        &|_| P::ScalarField::one(),
        vec![&randoms[0]],
        None,
    )
    .unwrap();
    InstanceWitnessPair {
        instance: InputInstance {
            ipa_commitment: labeled_commitment,
            point,
            evaluation,
            ipa_proof,
        },
        witness: (),
    }
}

fn dump<P>(curve: &str)
where
    P: SWModelParameters,
    P::BaseField: PrimeField,
    GroupAffine<P>: Absorbable<CF<P>>,
    CF<P>: PrimeField + Absorbable<CF<P>>,
{
    let mut rng = test_rng();

    // AS setup + index for a degree-`DEGREE` IPA predicate (no hiding).
    let public_params = AS::<P>::setup(&mut rng).unwrap();
    let predicate_params = IpaPC::<P>::setup(DEGREE, None, &mut rng).unwrap();
    let predicate_index = PredicateIndex {
        supported_degree_bound: DEGREE,
        supported_hiding_bound: 0,
    };
    let (pk, _vk, _dk) =
        AS::<P>::index(&public_params, &predicate_params, &predicate_index).unwrap();

    // The IPA committer key the inputs commit/open against (same generators the AS
    // trims its own copy from, at the same degree).
    let (ck, _ipa_vk) = IpaPC::<P>::trim(&predicate_params, DEGREE, 0, None).unwrap();
    let svk = &ck.svk;

    // --- Round 1: acc_prev = AS::prove([in0, in1], []) — the no-fold prove.
    let prev_inputs: Vec<InstanceWitnessPair<InputInstance<GroupAffine<P>>, ()>> =
        (0..NUM_PREV_INPUTS).map(|_| make_input::<P>(&ck, &mut rng)).collect();
    let no_acc: Vec<Accumulator<CF<P>, S<P>, AS<P>>> = Vec::new();
    let (acc_prev, prev_proof) = AS::<P>::prove(
        &pk,
        prev_inputs.iter().map(|inp| inp.as_ref()),
        no_acc.iter().map(|a| a.as_ref()),
        MakeZK::Disabled,
        None,
    )
    .unwrap();
    assert!(prev_proof.is_none(), "no-zk acc_prev proof must be None");

    // --- Round 2: fold one new input into acc_prev → golden.
    let new_input = make_input::<P>(&ck, &mut rng);
    let accs_prev = vec![acc_prev];
    let (accumulator, proof) = AS::<P>::prove(
        &pk,
        std::iter::once(new_input.as_ref()),
        accs_prev.iter().map(|a| a.as_ref()),
        MakeZK::Disabled,
        None,
    )
    .unwrap();
    assert!(proof.is_none(), "no-zk fold proof must be None");

    // The decider's size-`d` MSM scalars on the golden folded accumulator: run its
    // succinct check (constant-`1` opening challenges, as the AS decider does) and
    // densely expand. `.expect` also asserts the folded accumulator opening verifies.
    let acc_inst = &accumulator.instance;
    let acc_check_poly = IpaPC::<P>::succinct_check(
        svk,
        vec![&acc_inst.ipa_commitment],
        acc_inst.point,
        vec![acc_inst.evaluation],
        &acc_inst.ipa_proof,
        &|_| P::ScalarField::one(),
    )
    .expect("folded accumulator opening must verify");
    let decider_coeffs = acc_check_poly.compute_coeffs();

    let fixture = FoldFixture {
        note: format!(
            "IPA-PC accumulation fold (no-zk) fixtures ({} curve): one input folded into a prior accumulator",
            curve
        ),
        curve: curve.to_string(),
        supported_degree: svk.supported_degree,
        num_prev_inputs: NUM_PREV_INPUTS,
        h: PointJson::from_affine(&svk.h),
        s: PointJson::from_affine(&svk.s),
        generators: point_list(&ck.comm_key),
        input: InstanceJson::from_instance(&new_input.instance),
        acc_prev: InstanceJson::from_instance(&accs_prev[0].instance),
        accumulator: InstanceJson::from_instance(&accumulator.instance),
        decider_coeffs: fe_list(&decider_coeffs),
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}

curve_main!(dump);
