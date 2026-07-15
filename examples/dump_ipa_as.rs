//! IPA-PC accumulation `prove` (no-zk) fixtures for the frx port, over either
//! Pasta cycle curve (Pallas or Vesta).
//!
//! Drives the unmodified arkworks `AtomicASForInnerProductArgPC::prove` (no-zk,
//! no old accumulators) over `NUM_INPUTS` freshly committed+opened IPA inputs,
//! and dumps both the inputs the Python port replays (each input's commitment,
//! point, evaluation, and IPA proof `l_vec`/`r_vec`/`final_comm_key`/`c`) and the
//! golden output: the new accumulator instance. Plus the succinct verifier key
//! (`h`/`s`/`supported_degree`) the per-input succinct checks read and the IPA
//! committer-key generators the accumulator's IPA open folds over.
//!
//! No-zk only (`MakeZK::Disabled`): the AS proof is `None`, every IPA proof is
//! non-hiding (`hiding_comm`/`rand` = `None`), and the combined commitment is the
//! bare `Σ final_comm_key_j · lc_challenge_j` (no `s·randomness` term). The
//! Poseidon ARK constants are loaded by the test from the per-curve sponge
//! fixture, the same as the other byte-match tests.
//!
//! Generic over the curve (`PastaCurve`-style, no per-curve copy) — the curve is
//! a CLI arg, defaulting to Pallas:
//!
//!   cargo run --example dump_ipa_as -- pallas > python/testdata/ipa_as_fixtures.json
//!   cargo run --example dump_ipa_as -- vesta  > python/testdata/ipa_as_vesta_fixtures.json

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
use ark_accumulation::{AccumulationScheme, InstanceWitnessPair, MakeZK};

/// `ConstraintF<G>` (the sponge / constraint field), re-derived (it is
/// `pub(crate)` upstream). For the Pasta curves the base field is prime.
type CF<P> = <<P as ModelParameters>::BaseField as Field>::BasePrimeField;

/// The base Fiat-Shamir sponge the AS is parameterized over; the AS / IPA wrap it
/// in their own domain-separated spronges (`AS-FOR-IPA-PC-2020` / `IPA-PC-2020`).
type S<P> = PoseidonSponge<CF<P>>;
type AS<P> = AtomicASForInnerProductArgPC<GroupAffine<P>, S<P>>;

/// The exact IPA-PC instantiation `ipa_pc_as` builds on (matches `dump_ipa.rs`).
type IpaPC<P> = InnerProductArgPC<
    GroupAffine<P>,
    blake2::Blake2s,
    DensePolynomial<<P as ModelParameters>::ScalarField>,
    CF<P>,
    DomainSeparatedSponge<CF<P>, S<P>, IpaPCDomain>,
>;

/// Degree of each committed input polynomial (`d + 1 = 8` ⇒ `log_d = 3` IPA
/// rounds), and how many inputs to accumulate — two so the linear combination is
/// non-trivial (two distinct `lc_challenge_j` actually combine).
const DEGREE: usize = 7;
const NUM_INPUTS: usize = 2;

/// An `InputInstance` (an input or the new accumulator): the IPA commitment,
/// opening point + evaluation, and the proof's fold commitments,
/// `final_comm_key`, and final coefficient `c`. Field order is the fixture's key
/// order.
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
struct IpaAsFixture {
    note: String,
    curve: String,
    supported_degree: usize,
    num_inputs: usize,
    h: PointJson,
    s: PointJson,
    generators: Vec<PointJson>,
    inputs: Vec<InstanceJson>,
    accumulator: InstanceJson,
    decider_coeffs: Vec<String>,
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

    // The IPA committer key the inputs commit/open against (the AS trims its own
    // copy in `index`; this is the same generators at the same degree).
    let (ck, _ipa_vk) = IpaPC::<P>::trim(&predicate_params, DEGREE, 0, None).unwrap();
    let svk = &ck.svk;

    // `NUM_INPUTS` IPA inputs: commit a random degree-`d` polynomial, open it at a
    // random point — exactly as `ipa_pc_as`'s own `generate_inputs` does.
    // `Input<CF, S, AS>` is a transparent alias for this `InstanceWitnessPair`; we
    // build the pair directly because the alias's phantom `AS` can't be inferred
    // from a struct literal.
    let mut inputs: Vec<InstanceWitnessPair<InputInstance<GroupAffine<P>>, ()>> =
        Vec::with_capacity(NUM_INPUTS);
    for i in 0..NUM_INPUTS {
        let labeled_poly = LabeledPolynomial::new(
            PolynomialLabel::new(),
            DensePolynomial::<P::ScalarField>::rand(DEGREE, &mut rng),
            None,
            None,
        );
        let (labeled_commitments, randoms) =
            IpaPC::<P>::commit(&ck, &[labeled_poly.clone()], None).unwrap();
        let labeled_commitment = labeled_commitments.into_iter().next().unwrap();
        let point = P::ScalarField::rand(&mut rng);
        let evaluation = labeled_poly.evaluate(&point);
        let ipa_proof = IpaPC::<P>::open_individual_opening_challenges(
            &ck,
            vec![&labeled_poly],
            vec![&labeled_commitment],
            &point,
            &|_| P::ScalarField::one(),
            vec![&randoms[0]],
            None,
        )
        .unwrap();
        let _ = i;
        inputs.push(InstanceWitnessPair {
            instance: InputInstance {
                ipa_commitment: labeled_commitment,
                point,
                evaluation,
                ipa_proof,
            },
            witness: (),
        });
    }

    // The real AS prove: no old accumulators, no-zk, no external sponge.
    let (accumulator, proof) = AS::<P>::prove(
        &pk,
        inputs.iter().map(|inp| inp.as_ref()),
        Vec::new().iter().map(
            |a: &ark_accumulation::Accumulator<CF<P>, S<P>, AS<P>>| a.as_ref(),
        ),
        MakeZK::Disabled,
        None,
    )
    .unwrap();
    assert!(proof.is_none(), "no-zk AS proof must be None");

    // The decider's size-`d` MSM scalars: run the IPA succinct check on the new
    // accumulator (a single commitment, constant-`1` opening challenges, exactly as
    // the AS decider does) and densely expand its `SuccinctCheckPolynomial`. The
    // decider accepts iff `MSM(ck.comm_key, decider_coeffs) == accumulator
    // .final_comm_key` — these coeffs are the fused GPU core's scalar input (Slice
    // 4), the generators its bases. `.unwrap()` also asserts the accumulator opening
    // verifies.
    let acc_inst = &accumulator.instance;
    let acc_check_poly = IpaPC::<P>::succinct_check(
        svk,
        vec![&acc_inst.ipa_commitment],
        acc_inst.point,
        vec![acc_inst.evaluation],
        &acc_inst.ipa_proof,
        &|_| P::ScalarField::one(),
    )
    .expect("accumulator opening must verify");
    let decider_coeffs = acc_check_poly.compute_coeffs();

    let fixture = IpaAsFixture {
        note: format!("IPA-PC accumulation prove (no-zk) fixtures ({} curve)", curve),
        curve: curve.to_string(),
        supported_degree: svk.supported_degree,
        num_inputs: NUM_INPUTS,
        h: PointJson::from_affine(&svk.h),
        s: PointJson::from_affine(&svk.s),
        generators: point_list(&ck.comm_key),
        inputs: inputs
            .iter()
            .map(|inp| InstanceJson::from_instance(&inp.instance))
            .collect(),
        accumulator: InstanceJson::from_instance(&accumulator.instance),
        decider_coeffs: fe_list(&decider_coeffs),
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}

curve_main!(dump);
