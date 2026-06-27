//! IPA-PC accumulation `prove` **zk/hiding** fixtures for the jax port, over
//! either Pasta cycle curve — the zk twin of `dump_ipa_as.rs`.
//!
//! Drives the unmodified arkworks `AtomicASForInnerProductArgPC::prove` with
//! `MakeZK::Enabled`: the AS samples a degree-1 `random_linear_polynomial`, commits
//! it (`random_linear_polynomial_commitment`), and blinds the combined commitment
//! with `s·commitment_randomness`; the new accumulator's IPA opening is hiding. The
//! inputs themselves stay no-zk (their succinct check is the no-zk path), so this
//! fixture isolates the AS-level zk additions: the random-linear-polynomial absorb,
//! the randomized combined commitment, and the random poly's contribution to the
//! new point + evaluation.
//!
//! The AS `Randomness` (`random_linear_polynomial`, its commitment,
//! `commitment_randomness`) is the `prove` return value's `proof`, but its fields
//! are `pub(crate)`, so it is recovered through its derived `CanonicalSerialize`
//! and re-parsed here (no RNG replay) — exactly the three values the port's zk
//! combine reads. The new accumulator's own IPA opening proof (hiding) is dumped
//! by the no-zk `instance_json`; its `hiding_comm`/`rand` and the open's replayed
//! hiding polynomial are a later sub-step.
//!
//!   cargo run --example dump_ipa_as_zk -- pallas > python/testdata/ipa_as_zk_fixtures.json
//!   cargo run --example dump_ipa_as_zk -- vesta  > python/testdata/ipa_as_zk_vesta_fixtures.json

use ark_ec::models::ModelParameters;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ec::SWModelParameters;
use ark_ff::{BigInteger, Field, One, PrimeField, UniformRand, Zero};
use ark_poly::univariate::DensePolynomial;
use ark_poly::UVPolynomial;
use ark_poly_commit::ipa_pc::InnerProductArgPC;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment, PolynomialLabel};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_sponge::domain_separated::DomainSeparatedSponge;
use ark_sponge::poseidon::PoseidonSponge;
use ark_sponge::Absorbable;
use ark_std::test_rng;

use ark_accumulation::ipa_pc_as::{
    AtomicASForInnerProductArgPC, InputInstance, IpaPCDomain, PredicateIndex,
};
use ark_accumulation::{AccumulationScheme, InstanceWitnessPair, MakeZK};

type CF<P> = <<P as ModelParameters>::BaseField as Field>::BasePrimeField;
type S<P> = PoseidonSponge<CF<P>>;
type AS<P> = AtomicASForInnerProductArgPC<GroupAffine<P>, S<P>>;
type IpaPC<P> = InnerProductArgPC<
    GroupAffine<P>,
    blake2::Blake2s,
    DensePolynomial<<P as ModelParameters>::ScalarField>,
    CF<P>,
    DomainSeparatedSponge<CF<P>, S<P>, IpaPCDomain>,
>;

const DEGREE: usize = 7;
const NUM_INPUTS: usize = 2;

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

fn fr_list_json<F: PrimeField>(xs: &[F]) -> String {
    let v: Vec<String> = xs.iter().map(|f| format!("\"{}\"", fe_hex(f))).collect();
    format!("[{}]", v.join(","))
}

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
    format!("{{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}}", coord_x_hex(p), coord_y_hex(p))
}

fn points_json<P: SWModelParameters>(ps: &[GroupAffine<P>]) -> String
where
    P::BaseField: PrimeField,
{
    let v: Vec<String> = ps.iter().map(point_json).collect();
    format!("[{}]", v.join(","))
}

/// A no-zk `InputInstance` (an AS input) as JSON.
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

/// The new accumulator (zk) as JSON: the `instance_json` fields PLUS its hiding
/// IPA opening's `hiding_comm` / `rand` — the two extra inputs the port's zk
/// succinct check (and thus the zk decider MSM) reads off the accumulator.
fn accumulator_zk_json<P: SWModelParameters>(inst: &InputInstance<GroupAffine<P>>) -> String
where
    P::BaseField: PrimeField,
{
    let base = instance_json(inst);
    let hiding_comm = inst.ipa_proof.hiding_comm.expect("zk accumulator has hiding_comm");
    let rand = inst.ipa_proof.rand.expect("zk accumulator has rand");
    format!(
        "{},\"hiding_comm\":{},\"rand\":\"{}\"}}",
        &base[..base.len() - 1], // drop the closing brace to append the hiding fields
        point_json(&hiding_comm),
        fe_hex(&rand),
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

    let public_params = AS::<P>::setup(&mut rng).unwrap();
    let predicate_params = IpaPC::<P>::setup(DEGREE, None, &mut rng).unwrap();
    let predicate_index = PredicateIndex {
        supported_degree_bound: DEGREE,
        supported_hiding_bound: DEGREE,
    };
    let (pk, _vk, _dk) =
        AS::<P>::index(&public_params, &predicate_params, &predicate_index).unwrap();

    let (ck, _ipa_vk) = IpaPC::<P>::trim(&predicate_params, DEGREE, 0, None).unwrap();
    let svk = &ck.svk;

    // `NUM_INPUTS` no-zk IPA inputs (no hiding bound on the input polynomials), as
    // in the no-zk dump; the zk-ness here is the AS layer, not the inputs.
    let mut inputs: Vec<InstanceWitnessPair<InputInstance<GroupAffine<P>>, ()>> =
        Vec::with_capacity(NUM_INPUTS);
    for _ in 0..NUM_INPUTS {
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
        inputs.push(InstanceWitnessPair {
            instance: InputInstance { ipa_commitment: labeled_commitment, point, evaluation, ipa_proof },
            witness: (),
        });
    }

    // The real AS prove with hiding enabled. `proof` is `Some(Randomness)`.
    let (accumulator, proof) = AS::<P>::prove(
        &pk,
        inputs.iter().map(|inp| inp.as_ref()),
        Vec::new().iter().map(
            |a: &ark_accumulation::Accumulator<CF<P>, S<P>, AS<P>>| a.as_ref(),
        ),
        MakeZK::Enabled(&mut rng),
        None,
    )
    .unwrap();
    let proof = proof.expect("zk AS proof must be Some");

    // `Randomness` fields are `pub(crate)`; recover them through the derived
    // `CanonicalSerialize` (random_linear_polynomial: Vec<Fr>; its commitment: G;
    // commitment_randomness: Fr) — no RNG replay.
    let mut buf = Vec::new();
    proof.serialize(&mut buf).unwrap();
    let mut cur = buf.as_slice();
    let rlp_coeffs = Vec::<P::ScalarField>::deserialize(&mut cur).unwrap();
    let rlp_commitment = GroupAffine::<P>::deserialize(&mut cur).unwrap();
    let commitment_randomness = P::ScalarField::deserialize(&mut cur).unwrap();

    // The zk decider's size-`d` MSM scalars: the dense `compute_coeffs` of the
    // accumulator's (hiding) succinct check — the decider accepts iff
    // `MSM(ck.comm_key, decider_coeffs) == accumulator.final_comm_key`. These are
    // the fused zk GPU core's scalar input (Slice 5e).
    let acc_inst = &accumulator.instance;
    let acc_check_poly = IpaPC::<P>::succinct_check(
        svk,
        vec![&acc_inst.ipa_commitment],
        acc_inst.point,
        vec![acc_inst.evaluation],
        &acc_inst.ipa_proof,
        &|_| P::ScalarField::one(),
    )
    .expect("accumulator hiding opening must verify");
    let decider_coeffs = acc_check_poly.compute_coeffs();

    let inputs_json: Vec<String> = inputs.iter().map(|inp| instance_json(&inp.instance)).collect();

    println!("{{");
    println!("  \"note\": \"IPA-PC accumulation prove (zk) fixtures ({} curve)\",", curve);
    println!("  \"curve\": \"{}\",", curve);
    println!("  \"supported_degree\": {},", svk.supported_degree);
    println!("  \"num_inputs\": {},", NUM_INPUTS);
    println!("  \"h\": {},", point_json(&svk.h));
    println!("  \"s\": {},", point_json(&svk.s));
    println!("  \"generators\": {},", points_json(&ck.comm_key));
    println!("  \"random_linear_polynomial\": {},", fr_list_json(&rlp_coeffs));
    println!("  \"random_linear_polynomial_commitment\": {},", point_json(&rlp_commitment));
    println!("  \"commitment_randomness\": \"{}\",", fe_hex(&commitment_randomness));
    println!("  \"inputs\": [{}],", inputs_json.join(","));
    println!("  \"accumulator\": {},", accumulator_zk_json(&accumulator.instance));
    println!("  \"decider_coeffs\": {}", fr_list_json(&decider_coeffs));
    println!("}}");
}

fn main() {
    match std::env::args().nth(1).as_deref().unwrap_or("pallas") {
        "pallas" => dump::<ark_pallas::PallasParameters>("pallas"),
        "vesta" => dump::<ark_vesta::VestaParameters>("vesta"),
        other => panic!("unknown curve {} (expected pallas|vesta)", other),
    }
}
