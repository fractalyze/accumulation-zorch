//! IPA-PC accumulation `prove` (no-zk) fixtures for the jax port, over either
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
use ark_ff::{BigInteger, Field, One, PrimeField, UniformRand, Zero};
use ark_poly::univariate::DensePolynomial;
use ark_poly::UVPolynomial;
use ark_poly_commit::ipa_pc::InnerProductArgPC;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment, PolynomialLabel};
use ark_sponge::domain_separated::DomainSeparatedSponge;
use ark_sponge::poseidon::PoseidonSponge;
use ark_sponge::Absorbable;
use ark_std::test_rng;

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

fn point_json<P: SWModelParameters>(p: &GroupAffine<P>) -> String
where
    P::BaseField: PrimeField,
{
    format!(
        "{{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}}",
        coord_x_hex(p),
        coord_y_hex(p)
    )
}

fn points_json<P: SWModelParameters>(ps: &[GroupAffine<P>]) -> String
where
    P::BaseField: PrimeField,
{
    let v: Vec<String> = ps.iter().map(point_json).collect();
    format!("[{}]", v.join(","))
}

/// An `InputInstance` (an input or the new accumulator) as JSON: the IPA
/// commitment, opening point + evaluation, and the proof's fold commitments,
/// `final_comm_key`, and final coefficient `c`.
fn instance_json<P: SWModelParameters>(inst: &InputInstance<GroupAffine<P>>) -> String
where
    P::BaseField: PrimeField,
{
    format!(
        "{{\"commitment\":{},\"point\":\"{}\",\"evaluation\":\"{}\",\"l_vec\":{},\"r_vec\":{},\"final_comm_key\":{},\"c\":\"{}\"}}",
        point_json(&inst.ipa_commitment.commitment().comm),
        fe_hex(&inst.point),
        fe_hex(&inst.evaluation),
        points_json(&inst.ipa_proof.l_vec),
        points_json(&inst.ipa_proof.r_vec),
        point_json(&inst.ipa_proof.final_comm_key),
        fe_hex(&inst.ipa_proof.c),
    )
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

    let inputs_json: Vec<String> = inputs.iter().map(|inp| instance_json(&inp.instance)).collect();
    let gens_json = points_json(&ck.comm_key);

    println!("{{");
    println!("  \"note\": \"IPA-PC accumulation prove (no-zk) fixtures ({} curve)\",", curve);
    println!("  \"curve\": \"{}\",", curve);
    println!("  \"supported_degree\": {},", svk.supported_degree);
    println!("  \"num_inputs\": {},", NUM_INPUTS);
    println!("  \"h\": {},", point_json(&svk.h));
    println!("  \"s\": {},", point_json(&svk.s));
    println!("  \"generators\": {},", gens_json);
    println!("  \"inputs\": [{}],", inputs_json.join(","));
    println!("  \"accumulator\": {}", instance_json(&accumulator.instance));
    println!("}}");
}

fn main() {
    match std::env::args().nth(1).as_deref().unwrap_or("pallas") {
        "pallas" => dump::<ark_pallas::PallasParameters>("pallas"),
        "vesta" => dump::<ark_vesta::VestaParameters>("vesta"),
        other => panic!("unknown curve {} (expected pallas|vesta)", other),
    }
}
