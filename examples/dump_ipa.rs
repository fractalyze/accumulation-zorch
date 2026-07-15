//! IPA-PC `succinct_check` (no-zk) fixtures for the frx port, over either
//! Pasta cycle curve (Pallas or Vesta).
//!
//! Drives the unmodified arkworks `InnerProductArgPC` (the IPA polynomial
//! commitment the `ipa_pc_as` accumulation scheme is built on): commit a random
//! degree-`d` polynomial, open it at a random point, then run the crate's real
//! `IpaPC::succinct_check` on the opening — exactly as
//! `ipa_pc_as::succinct_check_inputs` invokes it (a single commitment, the
//! constant-`1` opening-challenge closure). Dumps the succinct-check inputs the
//! Python port replays (the commitment, point, evaluation, the proof's
//! `l_vec`/`r_vec`/`final_comm_key`/`c`, and the succinct verifier key `h`/`s`/
//! `supported_degree`) plus the golden outputs the port byte-matches: the
//! `SuccinctCheckPolynomial` round challenges, its dense `compute_coeffs`
//! expansion `h(X) = ∏(1 + ξ_i·X^{2^(log d − i)})`, and its evaluation at the
//! point.
//!
//! No-zk only (`hiding_bound = 0`): `proof.hiding_comm` / `proof.rand` are
//! `None`, so `succinct_check` skips the hiding sponge (the zk/hiding variant is
//! a later slice). The Poseidon ARK constants are NOT dumped here — the test
//! loads them from the per-curve sponge fixture (`sponge_fixtures.json` /
//! `sponge_vesta_fixtures.json`), the same way the AS byte-match tests do.
//!
//! Generic over the curve (`PastaCurve`-style, no per-curve copy) — the curve is
//! a CLI arg, defaulting to Pallas:
//!
//!   cargo run --example dump_ipa -- pallas > python/testdata/ipa_fixtures.json
//!   cargo run --example dump_ipa -- vesta  > python/testdata/ipa_vesta_fixtures.json

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

use ark_accumulation::ipa_pc_as::IpaPCDomain;

/// `ConstraintF<G>` (the sponge / constraint field), re-derived (it is
/// `pub(crate)` upstream). For the Pasta curves the base field is already prime,
/// so this is just the base field.
type CF<P> = <<P as ModelParameters>::BaseField as Field>::BasePrimeField;

/// The exact IPA-PC instantiation `ipa_pc_as` builds on: Blake2s generator
/// hashing, dense univariate polynomials, and the domain-separated Poseidon
/// sponge (`"IPA-PC-2020"`) over the constraint field.
type Sponge<P> = DomainSeparatedSponge<CF<P>, PoseidonSponge<CF<P>>, IpaPCDomain>;
type IpaPC<P> = InnerProductArgPC<
    GroupAffine<P>,
    blake2::Blake2s,
    DensePolynomial<<P as ModelParameters>::ScalarField>,
    CF<P>,
    Sponge<P>,
>;

/// Degree of the committed polynomial. `d + 1 = 8` is a power of two, so the IPA
/// runs `log_d = 3` rounds — three `(L_i, R_i)` folds, three round challenges,
/// and `2^3 = 8` `h(X)` coefficients — enough to exercise the seed challenge,
/// the per-round challenge recurrence, and the dense coefficient expansion.
const DEGREE: usize = 7;

/// The whole fixture. Field order is the fixture's key order.
#[derive(Serialize)]
struct IpaFixture {
    note: String,
    curve: String,
    supported_degree: usize,
    h: PointJson,
    s: PointJson,
    commitment: PointJson,
    point: String,
    evaluation: String,
    l_vec: Vec<PointJson>,
    r_vec: Vec<PointJson>,
    final_comm_key: PointJson,
    c: String,
    round_challenges: Vec<String>,
    coeffs: Vec<String>,
    eval_at_point: String,
}

fn dump<P>(curve: &str)
where
    P: SWModelParameters,
    P::BaseField: PrimeField,
    GroupAffine<P>: Absorbable<CF<P>>,
    CF<P>: PrimeField + Absorbable<CF<P>>,
{
    let mut rng = test_rng();

    // IPA committer key for degree `DEGREE`; no-zk ⇒ hiding bound 0. `ck.svk` is
    // the succinct verifier key (`h`, `s`, `supported_degree`) `succinct_check`
    // reads.
    let pp = IpaPC::<P>::setup(DEGREE, None, &mut rng).unwrap();
    let (ck, _vk) = IpaPC::<P>::trim(&pp, DEGREE, 0, None).unwrap();
    let svk = &ck.svk;

    // Commit a random degree-`d` polynomial, open it at a random point — the same
    // (commit, point, eval, open) an `ipa_pc_as` input is built from.
    let labeled_polynomials = vec![LabeledPolynomial::new(
        PolynomialLabel::new(),
        DensePolynomial::<P::ScalarField>::rand(DEGREE, &mut rng),
        None,
        None, // no hiding bound (no-zk)
    )];
    let (labeled_commitments, randoms) =
        IpaPC::<P>::commit(&ck, &labeled_polynomials, Some(&mut rng)).unwrap();
    let labeled_polynomial = &labeled_polynomials[0];
    let labeled_commitment = &labeled_commitments[0];
    let randomness = &randoms[0];

    let point = P::ScalarField::rand(&mut rng);
    let evaluation = labeled_polynomial.evaluate(&point);
    let proof = IpaPC::<P>::open_individual_opening_challenges(
        &ck,
        vec![labeled_polynomial],
        vec![labeled_commitment],
        &point,
        &|_| P::ScalarField::one(),
        vec![randomness],
        Some(&mut rng),
    )
    .unwrap();

    // The crate's real succinct check — single commitment, constant-`1` opening
    // challenges, exactly as `ipa_pc_as::succinct_check_inputs` calls it. Returns
    // the `SuccinctCheckPolynomial` (the round challenges); a `None` would mean
    // the opening failed to verify, so `.unwrap()` also asserts the fixture is a
    // valid opening.
    let check_poly = IpaPC::<P>::succinct_check(
        svk,
        vec![labeled_commitment],
        point,
        vec![evaluation],
        &proof,
        &|_| P::ScalarField::one(),
    )
    .unwrap();
    let round_challenges = &check_poly.0;
    let coeffs = check_poly.compute_coeffs();
    let eval_at_point = check_poly.evaluate(point);

    let commitment = labeled_commitment.commitment().comm;

    let fixture = IpaFixture {
        note: format!("IPA-PC succinct_check no-zk fixtures ({} curve)", curve),
        curve: curve.to_string(),
        supported_degree: svk.supported_degree,
        h: PointJson::from_affine(&svk.h),
        s: PointJson::from_affine(&svk.s),
        commitment: PointJson::from_affine(&commitment),
        point: fe_hex(&point),
        evaluation: fe_hex(&evaluation),
        l_vec: point_list(&proof.l_vec),
        r_vec: point_list(&proof.r_vec),
        final_comm_key: PointJson::from_affine(&proof.final_comm_key),
        c: fe_hex(&proof.c),
        round_challenges: fe_list(round_challenges),
        coeffs: fe_list(&coeffs),
        eval_at_point: fe_hex(&eval_at_point),
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}

curve_main!(dump);
