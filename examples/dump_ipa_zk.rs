//! IPA-PC `succinct_check` **zk/hiding** fixtures for the frx port, over either
//! Pasta cycle curve (Pallas or Vesta) — the zk twin of `dump_ipa.rs`.
//!
//! Same flow as the no-zk dump (commit a random degree-`d` polynomial, open it at
//! a random point, run the crate's real `IpaPC::succinct_check`), but with hiding
//! enabled: the polynomial carries `hiding_bound = Some(d)`, so `commit` /
//! `open` sample a hiding polynomial + blinder and the proof gains
//! `hiding_comm = Some(Σ comm_key·hiding_coeffs + s·hiding_rand)` and
//! `rand = Some(combined_rand)`. `succinct_check` then runs its hiding block: a
//! fresh `"IPA-PC-2020"` sponge absorbs `combined_commitment, hiding_comm,
//! to_bytes![point, combined_v]` → `hiding_challenge`, folds the commitment to
//! `combined_commitment + hiding_comm·hiding_challenge − s·rand`, and derives the
//! round challenges from THAT (so the zk round challenges differ from no-zk's).
//!
//! Dumps everything the no-zk fixture does PLUS the proof's `hiding_comm` / `rand`
//! — the two extra inputs the port's zk succinct check reads (`s` is already in
//! the verifier key). The Poseidon ARK constants are loaded by the test from the
//! per-curve sponge fixture, as in the other byte-match tests.
//!
//! Generic over the curve (`PastaCurve`-style, no per-curve copy) — the curve is
//! a CLI arg, defaulting to Pallas:
//!
//!   cargo run --example dump_ipa_zk -- pallas > python/testdata/ipa_zk_fixtures.json
//!   cargo run --example dump_ipa_zk -- vesta  > python/testdata/ipa_zk_vesta_fixtures.json

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
/// `pub(crate)` upstream). For the Pasta curves the base field is already prime.
type CF<P> = <<P as ModelParameters>::BaseField as Field>::BasePrimeField;

/// The exact IPA-PC instantiation `ipa_pc_as` builds on (matches `dump_ipa.rs`).
type Sponge<P> = DomainSeparatedSponge<CF<P>, PoseidonSponge<CF<P>>, IpaPCDomain>;
type IpaPC<P> = InnerProductArgPC<
    GroupAffine<P>,
    blake2::Blake2s,
    DensePolynomial<<P as ModelParameters>::ScalarField>,
    CF<P>,
    Sponge<P>,
>;

/// Degree of the committed polynomial (`d + 1 = 8` ⇒ `log_d = 3` IPA rounds), as
/// in `dump_ipa.rs`. With hiding the polynomial is masked to degree `d`.
const DEGREE: usize = 7;

/// The whole fixture. Field order is the fixture's key order.
#[derive(Serialize)]
struct IpaZkFixture {
    note: String,
    curve: String,
    supported_degree: usize,
    h: PointJson,
    s: PointJson,
    commitment: PointJson,
    point: String,
    evaluation: String,
    polynomial: Vec<String>,
    commitment_randomness: String,
    hiding_polynomial: Vec<String>,
    hiding_rand: String,
    l_vec: Vec<PointJson>,
    r_vec: Vec<PointJson>,
    final_comm_key: PointJson,
    c: String,
    hiding_comm: PointJson,
    rand: String,
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

    // IPA committer key for degree `DEGREE`, hiding bound `DEGREE` (zk). `ck.svk`
    // is the succinct verifier key (`h`, `s`, `supported_degree`) the hiding +
    // round-challenge sponges read (`s` is the hiding generator).
    let pp = IpaPC::<P>::setup(DEGREE, None, &mut rng).unwrap();
    let (ck, _vk) = IpaPC::<P>::trim(&pp, DEGREE, DEGREE, None).unwrap();
    let svk = &ck.svk;

    // Commit a random degree-`d` polynomial WITH hiding, open it at a random point
    // — the same (commit, point, eval, open) a zk `ipa_pc_as` input is built from.
    let labeled_polynomials = vec![LabeledPolynomial::new(
        PolynomialLabel::new(),
        DensePolynomial::<P::ScalarField>::rand(DEGREE, &mut rng),
        None,
        Some(DEGREE), // hiding bound (zk)
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
    assert!(
        proof.hiding_comm.is_some() && proof.rand.is_some(),
        "zk proof must carry hiding"
    );

    // The crate's real succinct check — single commitment, constant-`1` opening
    // challenges. With hiding present it runs the hiding block before the round
    // challenges; `.unwrap()` asserts the hiding opening verifies.
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

    // Prover-gate inputs: the committed polynomial + the open's hiding blinders, so a
    // consumer can re-run zorch's zk open and byte-match this golden proof. The
    // polynomial and the commit blinder are in hand; the hiding polynomial + its
    // blinder are drawn INSIDE `IpaPC::open` and never returned, so recover them by
    // replaying the deterministic `test_rng()` draw schedule up to `open` (the same
    // technique as `dump_ipa_as_zk.rs`). The `point` self-check asserts the replay
    // reached the exact state the prover entered `open` with, so the recovered
    // `hiding_polynomial` / `hiding_rand` are the ones the golden proof used.
    let polynomial = labeled_polynomial.polynomial().coeffs.clone();
    let commitment_randomness = randomness.rand;

    let mut rep = test_rng();
    let rep_pp = IpaPC::<P>::setup(DEGREE, None, &mut rep).unwrap();
    let (rep_ck, _) = IpaPC::<P>::trim(&rep_pp, DEGREE, DEGREE, None).unwrap();
    let rep_polys = vec![LabeledPolynomial::new(
        PolynomialLabel::new(),
        DensePolynomial::<P::ScalarField>::rand(DEGREE, &mut rep),
        None,
        Some(DEGREE),
    )];
    let _ = IpaPC::<P>::commit(&rep_ck, &rep_polys, Some(&mut rep)).unwrap();
    let rep_point = P::ScalarField::rand(&mut rep);
    assert_eq!(rep_point, point, "replay schedule mismatch (point)");
    // `IpaPC::open` (hiding) draws the raw degree-`d` hiding polynomial then its
    // blinder; the port applies the `−eval(point)` vanish shift before committing.
    let hiding_polynomial = DensePolynomial::<P::ScalarField>::rand(DEGREE, &mut rep);
    let hiding_rand = P::ScalarField::rand(&mut rep);

    let fixture = IpaZkFixture {
        note: format!("IPA-PC succinct_check zk/hiding fixtures ({} curve)", curve),
        curve: curve.to_string(),
        supported_degree: svk.supported_degree,
        h: PointJson::from_affine(&svk.h),
        s: PointJson::from_affine(&svk.s),
        commitment: PointJson::from_affine(&commitment),
        point: fe_hex(&point),
        evaluation: fe_hex(&evaluation),
        polynomial: fe_list(&polynomial),
        commitment_randomness: fe_hex(&commitment_randomness),
        hiding_polynomial: fe_list(&hiding_polynomial.coeffs),
        hiding_rand: fe_hex(&hiding_rand),
        l_vec: point_list(&proof.l_vec),
        r_vec: point_list(&proof.r_vec),
        final_comm_key: PointJson::from_affine(&proof.final_comm_key),
        c: fe_hex(&proof.c),
        hiding_comm: PointJson::from_affine(&proof.hiding_comm.unwrap()),
        rand: fe_hex(&proof.rand.unwrap()),
        round_challenges: fe_list(round_challenges),
        coeffs: fe_list(&coeffs),
        eval_at_point: fe_hex(&eval_at_point),
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}

curve_main!(dump);
