//! IPA-PC `succinct_check` (no-zk) fixtures for the jax port, over either
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
use ark_ff::{BigInteger, Field, One, PrimeField, UniformRand, Zero};
use ark_poly::univariate::DensePolynomial;
use ark_poly::{Polynomial, UVPolynomial};
use ark_poly_commit::ipa_pc::InnerProductArgPC;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment, PolynomialLabel};
use ark_serialize::CanonicalSerialize;
use ark_sponge::domain_separated::DomainSeparatedSponge;
use ark_sponge::poseidon::PoseidonSponge;
use ark_sponge::Absorbable;
use ark_std::test_rng;

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

/// `{"x_le_hex":..,"y_le_hex":..}` for an affine point.
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

    println!("{{");
    println!("  \"note\": \"IPA-PC succinct_check no-zk fixtures ({} curve)\",", curve);
    println!("  \"curve\": \"{}\",", curve);
    println!("  \"supported_degree\": {},", svk.supported_degree);
    println!("  \"h\": {},", point_json(&svk.h));
    println!("  \"s\": {},", point_json(&svk.s));
    println!("  \"commitment\": {},", point_json(&commitment));
    println!("  \"point\": \"{}\",", fe_hex(&point));
    println!("  \"evaluation\": \"{}\",", fe_hex(&evaluation));
    println!("  \"l_vec\": {},", points_json(&proof.l_vec));
    println!("  \"r_vec\": {},", points_json(&proof.r_vec));
    println!("  \"final_comm_key\": {},", point_json(&proof.final_comm_key));
    println!("  \"c\": \"{}\",", fe_hex(&proof.c));
    println!("  \"round_challenges\": {},", fr_list_json(round_challenges));
    println!("  \"coeffs\": {},", fr_list_json(&coeffs));
    println!("  \"eval_at_point\": \"{}\"", fe_hex(&eval_at_point));
    println!("}}");
}

fn main() {
    match std::env::args().nth(1).as_deref().unwrap_or("pallas") {
        "pallas" => dump::<ark_pallas::PallasParameters>("pallas"),
        "vesta" => dump::<ark_vesta::VestaParameters>("vesta"),
        other => panic!("unknown curve {} (expected pallas|vesta)", other),
    }
}
