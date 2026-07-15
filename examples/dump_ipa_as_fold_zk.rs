//! IPA-PC accumulation **fold** (zk/hiding) fixtures for the frx port: one input
//! folded INTO a prior *hiding* accumulator (`old_accumulators = [acc_prev]`), over
//! either Pasta cycle curve. The fold twin of `dump_ipa_as_zk.rs`, and the zk twin
//! of `dump_ipa_as_fold.rs`.
//!
//! Two rounds, each driving the unmodified arkworks
//! `AtomicASForInnerProductArgPC::prove` with `MakeZK::Enabled`:
//!   1. **acc_prev** — `AS::prove([in0, in1], [])`: a zk prove over `NUM_PREV_INPUTS`
//!      no-zk inputs. Its accumulator carries a **hiding** IPA opening
//!      (`hiding_comm`/`rand`).
//!   2. **fold** — `AS::prove([in_new], [acc_prev])` → the golden folded
//!      accumulator. succinct-check list `[in_new, acc_prev]` (inputs first).
//!
//! The new wrinkle over the no-zk fold: `acc_prev` is hiding, so the fold's
//! succinct check on it is the **zk** path (folding the hiding seed with `svk.s`
//! and the proof's `hiding_comm`/`rand`), while the new input stays no-zk. The frx
//! port mirrors this: no-zk succinct check for the input, zk succinct check for the
//! prior accumulator, then the same `combine_zk` + hiding `IpaPC::open`.
//!
//! Per-phase seeded RNGs (mirrors `dump_as_fold_zk.rs`) so each round's randomness
//! replay is local. The fold's AS `Randomness` (`random_linear_polynomial`, its
//! commitment, `commitment_randomness`) is recovered through the proof's derived
//! `CanonicalSerialize`; the fold's hiding `IpaPC::open` randomness
//! (`hiding_polynomial`/`hiding_rand`, drawn inside `open` and never returned) is
//! recovered by replaying round 2's RNG from its seed — the self-check asserts the
//! replayed AS randomness equals the serialize-recovered values, so the draw
//! schedule (and thus the recovered hiding poly/rand) is correct. `acc_prev`'s own
//! hiding randomness is internal (acc_prev is dumped as output), so it is not
//! replayed.
//!
//!   cargo run --example dump_ipa_as_fold_zk -- pallas > python/testdata/ipa_as_fold_zk_fixtures.json
//!   cargo run --example dump_ipa_as_fold_zk -- vesta  > python/testdata/ipa_as_fold_zk_vesta_fixtures.json

use ark_ec::models::ModelParameters;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ec::SWModelParameters;
use ark_ff::{Field, One, PrimeField, UniformRand};
use ark_poly::univariate::DensePolynomial;
use ark_poly::UVPolynomial;
use ark_poly_commit::ipa_pc::InnerProductArgPC;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment, PolynomialLabel};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_sponge::domain_separated::DomainSeparatedSponge;
use ark_sponge::poseidon::PoseidonSponge;
use ark_sponge::Absorbable;
use ark_std::rand::{rngs::StdRng, RngCore, SeedableRng};
use serde::Serialize;

use fixture_json::{curve_main, fe_hex, fe_list, point_list, PointJson};

use ark_accumulation::ipa_pc_as::{
    AtomicASForInnerProductArgPC, InputInstance, IpaPCDomain, PredicateIndex,
};
use ark_accumulation::{AccumulationScheme, Accumulator, InstanceWitnessPair, MakeZK};

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
const NUM_PREV_INPUTS: usize = 2;

// Per-phase RNG seeds (mirrors dump_as_fold_zk.rs's seed^const scheme): setup +
// generators, the acc_prev prove, and the fold prove get independent streams so
// round 2's randomness replays from its own seed.
const SEED_SETUP: u64 = 0x5e7;
const SEED_ACC_PREV: u64 = 0xacc0;
const SEED_FOLD: u64 = 0xf01d;

/// An `InputInstance` as JSON. The no-zk new input passes `hiding = None`; the
/// hiding accumulators (acc_prev and the golden fold) pass
/// `Some((hiding_comm, rand))` — the two extra fields their zk succinct check reads.
/// The pair is skipped entirely when absent (not emitted as `null`), so the no-zk
/// input keeps the seven-key shape. Field order is the fixture's key order.
#[derive(Serialize)]
struct InstanceJson {
    commitment: PointJson,
    point: String,
    evaluation: String,
    l_vec: Vec<PointJson>,
    r_vec: Vec<PointJson>,
    final_comm_key: PointJson,
    c: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    hiding_comm: Option<PointJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rand: Option<String>,
}

impl InstanceJson {
    fn from_instance<P: SWModelParameters>(
        inst: &InputInstance<GroupAffine<P>>,
        hiding: Option<(GroupAffine<P>, P::ScalarField)>,
    ) -> Self
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
            hiding_comm: hiding.map(|(comm, _)| PointJson::from_affine(&comm)),
            rand: hiding.map(|(_, rand)| fe_hex(&rand)),
        }
    }
}

/// The whole fixture. Field order is the fixture's key order.
#[derive(Serialize)]
struct FoldZkFixture {
    note: String,
    curve: String,
    supported_degree: usize,
    num_prev_inputs: usize,
    h: PointJson,
    s: PointJson,
    generators: Vec<PointJson>,
    random_linear_polynomial: Vec<String>,
    random_linear_polynomial_commitment: PointJson,
    commitment_randomness: String,
    hiding_polynomial: Vec<String>,
    hiding_rand: String,
    input: InstanceJson,
    acc_prev: InstanceJson,
    accumulator: InstanceJson,
    decider_coeffs: Vec<String>,
}

/// Commit a fresh random degree-`DEGREE` polynomial and open it at a random point
/// — one **no-zk** IPA input (no hiding bound; the zk-ness here is the AS layer).
/// Draw order on `rng`: polynomial (`DEGREE + 1` coeffs) then point — commit/open
/// take no rng, so the replay schedule is exactly `[poly, point]`.
fn make_input<P>(
    ck: &ark_poly_commit::ipa_pc::CommitterKey<GroupAffine<P>>,
    rng: &mut impl RngCore,
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
        instance: InputInstance { ipa_commitment: labeled_commitment, point, evaluation, ipa_proof },
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
    let mut setup_rng = StdRng::seed_from_u64(SEED_SETUP);
    let public_params = AS::<P>::setup(&mut setup_rng).unwrap();
    let predicate_params = IpaPC::<P>::setup(DEGREE, None, &mut setup_rng).unwrap();
    let predicate_index = PredicateIndex {
        supported_degree_bound: DEGREE,
        supported_hiding_bound: DEGREE,
    };
    let (pk, _vk, _dk) =
        AS::<P>::index(&public_params, &predicate_params, &predicate_index).unwrap();
    let (ck, _ipa_vk) = IpaPC::<P>::trim(&predicate_params, DEGREE, 0, None).unwrap();
    let svk = &ck.svk;

    // --- Round 1: acc_prev = AS::prove([in0, in1], []) with hiding. acc_prev's
    // accumulator carries a hiding IPA opening.
    let mut rng1 = StdRng::seed_from_u64(SEED_ACC_PREV);
    let prev_inputs: Vec<InstanceWitnessPair<InputInstance<GroupAffine<P>>, ()>> =
        (0..NUM_PREV_INPUTS).map(|_| make_input::<P>(&ck, &mut rng1)).collect();
    let no_acc: Vec<Accumulator<CF<P>, S<P>, AS<P>>> = Vec::new();
    let (acc_prev, prev_proof) = AS::<P>::prove(
        &pk,
        prev_inputs.iter().map(|inp| inp.as_ref()),
        no_acc.iter().map(|a| a.as_ref()),
        MakeZK::Enabled(&mut rng1),
        None,
    )
    .unwrap();
    prev_proof.expect("zk acc_prev proof must be Some");
    let accs_prev = vec![acc_prev];

    // --- Round 2: fold one no-zk new input into the hiding acc_prev → golden.
    let mut rng2 = StdRng::seed_from_u64(SEED_FOLD);
    let new_input = make_input::<P>(&ck, &mut rng2);
    let (accumulator, proof) = AS::<P>::prove(
        &pk,
        std::iter::once(new_input.as_ref()),
        accs_prev.iter().map(|a| a.as_ref()),
        MakeZK::Enabled(&mut rng2),
        None,
    )
    .unwrap();
    let proof = proof.expect("zk fold proof must be Some");

    // Recover the FOLD's AS `Randomness` (`pub(crate)`) through its derived
    // `CanonicalSerialize`: random_linear_polynomial coeffs, its commitment,
    // commitment_randomness — the three values the port's zk combine reads.
    let mut buf = Vec::new();
    proof.serialize(&mut buf).unwrap();
    let mut cur = buf.as_slice();
    let rlp_coeffs = Vec::<P::ScalarField>::deserialize(&mut cur).unwrap();
    let rlp_commitment = GroupAffine::<P>::deserialize(&mut cur).unwrap();
    let commitment_randomness = P::ScalarField::deserialize(&mut cur).unwrap();

    // Replay round 2's RNG from its seed to recover the fold's hiding `IpaPC::open`
    // randomness (drawn inside `open`, never returned). rng2's schedule is
    // [new_input poly (DEGREE+1), new_input point, rlp0, rlp1, commitment_randomness,
    //  hiding_polynomial (DEGREE+1), hiding_rand]; the self-check on rlp/cr confirms
    // the schedule, so the hiding poly/rand drawn after it are correct.
    let mut rep = StdRng::seed_from_u64(SEED_FOLD);
    let _ = DensePolynomial::<P::ScalarField>::rand(DEGREE, &mut rep); // new_input poly
    let _ = P::ScalarField::rand(&mut rep); // new_input point
    let rep_rlp_0 = P::ScalarField::rand(&mut rep);
    let rep_rlp_1 = P::ScalarField::rand(&mut rep);
    let rep_cr = P::ScalarField::rand(&mut rep);
    assert_eq!(rep_rlp_0, rlp_coeffs[0], "replay schedule: fold rlp coeff0 mismatch");
    assert_eq!(rep_rlp_1, rlp_coeffs[1], "replay schedule: fold rlp coeff1 mismatch");
    assert_eq!(rep_cr, commitment_randomness, "replay schedule: fold commitment_randomness mismatch");
    let hiding_polynomial = DensePolynomial::<P::ScalarField>::rand(DEGREE, &mut rep);
    let hiding_rand = P::ScalarField::rand(&mut rep);

    // The zk decider's size-`d` MSM scalars: the dense `compute_coeffs` of the
    // folded accumulator's (hiding) succinct check.
    let acc_inst = &accumulator.instance;
    let acc_check_poly = IpaPC::<P>::succinct_check(
        svk,
        vec![&acc_inst.ipa_commitment],
        acc_inst.point,
        vec![acc_inst.evaluation],
        &acc_inst.ipa_proof,
        &|_| P::ScalarField::one(),
    )
    .expect("folded accumulator hiding opening must verify");
    let decider_coeffs = acc_check_poly.compute_coeffs();

    let acc_prev_inst = &accs_prev[0].instance;
    let acc_prev_hiding = (
        acc_prev_inst.ipa_proof.hiding_comm.expect("zk acc_prev has hiding_comm"),
        acc_prev_inst.ipa_proof.rand.expect("zk acc_prev has rand"),
    );
    let golden_hiding = (
        acc_inst.ipa_proof.hiding_comm.expect("zk golden has hiding_comm"),
        acc_inst.ipa_proof.rand.expect("zk golden has rand"),
    );

    let fixture = FoldZkFixture {
        note: format!(
            "IPA-PC accumulation fold (zk) fixtures ({} curve): one no-zk input folded into a hiding prior accumulator",
            curve
        ),
        curve: curve.to_string(),
        supported_degree: svk.supported_degree,
        num_prev_inputs: NUM_PREV_INPUTS,
        h: PointJson::from_affine(&svk.h),
        s: PointJson::from_affine(&svk.s),
        generators: point_list(&ck.comm_key),
        random_linear_polynomial: fe_list(&rlp_coeffs),
        random_linear_polynomial_commitment: PointJson::from_affine(&rlp_commitment),
        commitment_randomness: fe_hex(&commitment_randomness),
        hiding_polynomial: fe_list(hiding_polynomial.coeffs()),
        hiding_rand: fe_hex(&hiding_rand),
        input: InstanceJson::from_instance(&new_input.instance, None),
        acc_prev: InstanceJson::from_instance(acc_prev_inst, Some(acc_prev_hiding)),
        accumulator: InstanceJson::from_instance(acc_inst, Some(golden_hiding)),
        decider_coeffs: fe_list(&decider_coeffs),
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}

curve_main!(dump);
