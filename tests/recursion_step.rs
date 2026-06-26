//! Pasta-cycle recursion: the in-circuit verifier of an accumulation step,
//! proven (and accumulated) on the *other* curve of the cycle.
//!
//! [`RecursionCircuit<VC>`] builds the **reused** arkworks
//! `ASForR1CSNarkVerifierGadget` (zero MSM ops → no GPU seam, so it is used
//! as-is, not copied) as a `ConstraintSynthesizer<ConstraintF<VC::G>>` for the
//! verified curve `VC`. The [`VerifiedCurve`] trait carries the only things that
//! vary — the curve `G` and its in-circuit var `GVar`; the sponge is always
//! Poseidon over `ConstraintF<G>`. `RecursionCircuit` lives here (test-local):
//! it verifies an arkworks-AS accumulation, so it depends on the
//! `ark-accumulation` gadget (a dev-dep); productionizing it is a later slice.
//!
//! The hinge is the type-level Pasta cycle: `ConstraintF<VC::G>` (the verifier
//! circuit's field) IS the *other* curve's scalar field, so the circuit proves
//! on that curve directly — no isomorphism bridge. Forward: a
//! `RecursionCircuit<Pallas>` proves on Vesta (`ark_pallas::Fq == ark_vesta::Fr`),
//! in the [`vesta`] module.
//!
//! Each proving module mirrors the same shape: prove the verifier circuit as a
//! NARK on its curve (the **half-step**), then *accumulate* that NARK proof
//! by folding it into a prior accumulator (the **full IVC step**).
//!
//! CPU gate (no GPU, `--features recursion`): `recursion_circuit_satisfiable`
//! (both verifier circuits synthesize a satisfied constraint system). The `dump`
//! submodules drive the pristine arkworks prover to write the off-tree recursion
//! fixtures the **fused** GPU gates (`gpu_fused_{nark,fold}_*`) byte-match against,
//! and `arkworks_fold_timing` is the CPU baseline for the fold benchmark. The
//! fused jax-exported fold core (`tests/gpu_fused_fold_zk_byte_match.rs`) is the
//! recursion fold's GPU path.
#![cfg(feature = "recursion")]

use ark_accumulation::constraints::ASVerifierGadget;
use ark_accumulation::r1cs_nark_as::constraints::ASForR1CSNarkVerifierGadget;
use ark_accumulation::r1cs_nark_as::r1cs_nark::R1CSNark;
use ark_accumulation::r1cs_nark_as::{ASForR1CSNark, InputInstance};
use ark_accumulation::{AccumulationScheme, Accumulator, Input, MakeZK};
use ark_ec::AffineCurve;
use ark_ff::UniformRand;
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::bits::boolean::Boolean;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::groups::CurveVar;
use ark_relations::lc;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, OptimizationGoal, SynthesisError,
    SynthesisMode,
};
use ark_sponge::constraints::AbsorbableGadget;
use ark_sponge::poseidon::constraints::PoseidonSpongeVar;
use ark_sponge::poseidon::PoseidonSponge;
use ark_sponge::Absorbable;
use ark_sponge::CryptographicSponge;
use ark_std::rand::{rngs::StdRng, SeedableRng};

/// A curve whose accumulation step is verified *inside* a [`RecursionCircuit`].
///
/// The verifier circuit runs over `ConstraintF<G>` — the verified curve's base
/// field, which by the type-level Pasta cycle IS the *other* curve's scalar
/// field, so the circuit proves on that other curve directly (no isomorphism
/// bridge). Only the curve and its in-circuit var vary; the sponge is always
/// Poseidon over `ConstraintF<G>`. The two associated-type bounds are copied
/// verbatim from the reused `ASForR1CSNarkVerifierGadget`, so `VC: VerifiedCurve`
/// carries everything the gadget (and the native `ASForR1CSNark` it verifies)
/// needs — bar `ConstraintF<G>: Absorbable<ConstraintF<G>>`, restated where used.
trait VerifiedCurve {
    type G: AffineCurve + Absorbable<ConstraintF<Self::G>>;
    type GVar: CurveVar<<Self::G as AffineCurve>::Projective, ConstraintF<Self::G>>
        + AbsorbableGadget<ConstraintF<Self::G>>;
    const NAME: &'static str;
}

struct Pallas;
impl VerifiedCurve for Pallas {
    type G = ark_pallas::Affine;
    type GVar = ark_pallas::constraints::GVar;
    const NAME: &'static str = "pallas";
}

struct Vesta;
impl VerifiedCurve for Vesta {
    type G = ark_vesta::Affine;
    type GVar = ark_vesta::constraints::GVar;
    const NAME: &'static str = "vesta";
}

// `ConstraintF<G>` is `pub(crate)` upstream; re-derive it. It is the verifier
// circuit's field = the verified curve's base field = the proving curve's scalar
// field (the Pasta cycle, type-level).
type ConstraintF<G> = <<G as AffineCurve>::BaseField as ark_ff::Field>::BasePrimeField;
type CF<VC> = ConstraintF<<VC as VerifiedCurve>::G>;
type Fr<VC> = <<VC as VerifiedCurve>::G as AffineCurve>::ScalarField;
type Sponge<VC> = PoseidonSponge<CF<VC>>;
type SpongeVar<VC> = PoseidonSpongeVar<CF<VC>>;
type AS<VC> = ASForR1CSNark<<VC as VerifiedCurve>::G, Sponge<VC>>;
type ASV<VC> = ASForR1CSNarkVerifierGadget<
    <VC as VerifiedCurve>::G,
    <VC as VerifiedCurve>::GVar,
    Sponge<VC>,
    SpongeVar<VC>,
>;

type VerifierKey<VC> = <AS<VC> as AccumulationScheme<CF<VC>, Sponge<VC>>>::VerifierKey;
type AccInstance<VC> = <AS<VC> as AccumulationScheme<CF<VC>, Sponge<VC>>>::AccumulatorInstance;
type AsInputInstance<VC> = <AS<VC> as AccumulationScheme<CF<VC>, Sponge<VC>>>::InputInstance;
type Proof<VC> = <AS<VC> as AccumulationScheme<CF<VC>, Sponge<VC>>>::Proof;

/// `a · b = c`, padded — the circuit arkworks' `r1cs_nark_as` tests use.
#[derive(Clone)]
struct DummyCircuit<F: ark_ff::Field> {
    a: Option<F>,
    b: Option<F>,
    num_inputs: usize,
    num_constraints: usize,
}

impl<F: ark_ff::Field> ConstraintSynthesizer<F> for DummyCircuit<F> {
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

/// The recursion predicate: "the previous accumulation step (on `VC`'s curve)
/// verified." Allocates the step's `(vk, input instances, old-acc instances,
/// new-acc instance, proof)` and enforces the reused AS verifier gadget returns
/// `true`.
struct RecursionCircuit<VC: VerifiedCurve>
where
    ConstraintF<VC::G>: Absorbable<ConstraintF<VC::G>>,
{
    vk: VerifierKey<VC>,
    input_instances: Vec<AsInputInstance<VC>>,
    old_acc_instances: Vec<AccInstance<VC>>,
    new_acc_instance: AccInstance<VC>,
    proof: Proof<VC>,
}

impl<VC: VerifiedCurve> ConstraintSynthesizer<CF<VC>> for RecursionCircuit<VC>
where
    ConstraintF<VC::G>: Absorbable<ConstraintF<VC::G>>,
{
    fn generate_constraints(self, cs: ConstraintSystemRef<CF<VC>>) -> Result<(), SynthesisError> {
        let RecursionCircuit {
            vk,
            input_instances,
            old_acc_instances,
            new_acc_instance,
            proof,
        } = self;
        let vk_var = <ASV<VC> as ASVerifierGadget<CF<VC>, Sponge<VC>, SpongeVar<VC>, AS<VC>>>::VerifierKey::new_constant(
            cs.clone(),
            vk,
        )?;
        let input_vars = input_instances
            .iter()
            .map(|i| {
                <ASV<VC> as ASVerifierGadget<CF<VC>, Sponge<VC>, SpongeVar<VC>, AS<VC>>>::InputInstance::new_witness(
                    cs.clone(),
                    || Ok(i.clone()),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let old_acc_vars = old_acc_instances
            .iter()
            .map(|a| {
                <ASV<VC> as ASVerifierGadget<CF<VC>, Sponge<VC>, SpongeVar<VC>, AS<VC>>>::AccumulatorInstance::new_witness(
                    cs.clone(),
                    || Ok(a.clone()),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let new_acc_var =
            <ASV<VC> as ASVerifierGadget<CF<VC>, Sponge<VC>, SpongeVar<VC>, AS<VC>>>::AccumulatorInstance::new_input(
                cs.clone(),
                || Ok(new_acc_instance),
            )?;
        let proof_var = <ASV<VC> as ASVerifierGadget<CF<VC>, Sponge<VC>, SpongeVar<VC>, AS<VC>>>::Proof::new_witness(
            cs.clone(),
            || Ok(proof),
        )?;

        let verified = ASV::<VC>::verify(
            cs,
            &vk_var,
            &input_vars,
            &old_acc_vars,
            &new_acc_var,
            &proof_var,
            None::<SpongeVar<VC>>,
        )?;
        verified.enforce_equal(&Boolean::TRUE)
    }
}

/// Runs one arkworks accumulation step on `VC`'s curve (one input, no old
/// accumulators) and captures the data its in-circuit verifier needs.
fn build_step<VC: VerifiedCurve>(make_zk: bool, seed: u64) -> RecursionCircuit<VC>
where
    ConstraintF<VC::G>: Absorbable<ConstraintF<VC::G>>,
{
    let num_inputs = 5usize;
    let num_constraints = 10usize;
    let mut rng = StdRng::seed_from_u64(seed);

    let nark_pp = R1CSNark::<VC::G, Sponge<VC>>::setup();
    let index_circuit = DummyCircuit {
        a: Some(Fr::<VC>::rand(&mut rng)),
        b: Some(Fr::<VC>::rand(&mut rng)),
        num_inputs,
        num_constraints,
    };
    let (ipk, ivk) = R1CSNark::<VC::G, Sponge<VC>>::index(&nark_pp, index_circuit).unwrap();

    let as_pp = AS::<VC>::setup(&mut rng).unwrap();
    let (pk, vk, _dk) = AS::<VC>::index(&as_pp, &(), &(ipk.clone(), ivk.clone())).unwrap();

    let circuit = DummyCircuit {
        a: Some(Fr::<VC>::rand(&mut rng)),
        b: Some(Fr::<VC>::rand(&mut rng)),
        num_inputs,
        num_constraints,
    };
    let nark_sponge = ASForR1CSNark::<VC::G, Sponge<VC>>::nark_sponge(&Sponge::<VC>::new());
    let nark_proof = R1CSNark::<VC::G, Sponge<VC>>::prove(
        &ipk,
        circuit.clone(),
        make_zk,
        Some(nark_sponge),
        Some(&mut rng),
    )
    .unwrap();

    let pcs = ConstraintSystem::<Fr<VC>>::new_ref();
    pcs.set_optimization_goal(OptimizationGoal::Weight);
    pcs.set_mode(SynthesisMode::Prove {
        construct_matrices: false,
    });
    circuit.generate_constraints(pcs.clone()).unwrap();
    pcs.finalize();
    let r1cs_input = pcs.borrow().unwrap().instance_assignment.clone();

    let input = Input::<CF<VC>, Sponge<VC>, AS<VC>> {
        instance: InputInstance {
            r1cs_input,
            first_round_message: nark_proof.first_msg.clone(),
        },
        witness: nark_proof.second_msg,
    };
    let inputs = vec![input];
    let no_accumulators: Vec<Accumulator<CF<VC>, Sponge<VC>, AS<VC>>> = Vec::new();

    let make_zk_arg = if make_zk {
        MakeZK::Enabled(&mut rng)
    } else {
        MakeZK::Disabled
    };
    let (accumulator, proof) = AS::<VC>::prove(
        &pk,
        Input::<CF<VC>, Sponge<VC>, AS<VC>>::map_to_refs(&inputs),
        Accumulator::<CF<VC>, Sponge<VC>, AS<VC>>::map_to_refs(&no_accumulators),
        make_zk_arg,
        None,
    )
    .unwrap();

    RecursionCircuit {
        vk,
        input_instances: inputs.iter().map(|i| i.instance.clone()).collect(),
        old_acc_instances: Vec::new(),
        new_acc_instance: accumulator.instance,
        proof,
    }
}

/// CPU-only: the reused verifier gadget, fed a real accumulation step on each
/// curve of the cycle, synthesizes a satisfied constraint system over the
/// matching `ConstraintF`. No GPU.
#[test]
fn recursion_circuit_satisfiable() {
    fn check<VC: VerifiedCurve>()
    where
        ConstraintF<VC::G>: Absorbable<ConstraintF<VC::G>>,
    {
        for make_zk in [false, true] {
            for seed in [0u64, 7] {
                let circuit = build_step::<VC>(make_zk, seed);
                let cs = ConstraintSystem::<CF<VC>>::new_ref();
                circuit.generate_constraints(cs.clone()).unwrap();
                assert!(
                    cs.is_satisfied().unwrap(),
                    "{} recursion verifier circuit unsatisfied (make_zk={}, seed={})",
                    VC::NAME,
                    make_zk,
                    seed
                );
            }
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}

/// Shared Vesta-side plumbing for the recursion slices (the standalone half-step
/// NARK and the full IVC fold). Curve-fixed to Vesta — the cycle curve the
/// Pallas-verifier circuit proves on — driving the arkworks prover on CPU to emit
/// the golden fixtures; the fold's GPU path is the fused core.
mod vesta {
    use super::{build_step, Pallas};
    use ark_accumulation::r1cs_nark_as::r1cs_nark::{
        IndexProverKey, IndexVerifierKey, Proof as NarkProof, R1CSNark as ZkxR1CSNark,
    };
    use ark_accumulation::r1cs_nark_as::{ASForR1CSNark, InputInstance};
    use ark_accumulation::{AccumulationScheme, Accumulator, Input, MakeZK};
    use ark_relations::r1cs::{
        ConstraintSynthesizer, ConstraintSystem, OptimizationGoal, SynthesisMode,
    };
    use ark_serialize::CanonicalSerialize;
    use ark_sponge::poseidon::PoseidonSponge;
    use ark_sponge::CryptographicSponge;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    // The verifier circuit is over ark_pallas::Fq = ark_vesta::Fr (the Vesta
    // scalar field), so it proves with a Vesta NARK whose constraint field — and
    // that of the accumulation scheme over it — is ark_vesta::Fq (`VCf`,
    // = ark_pallas::Fr). The r1cs_input the accumulation input carries is still
    // over the scalar field, i.e. `CF`.
    pub(super) type VG = ark_vesta::Affine;
    pub(super) type VCf = ark_vesta::Fq;
    pub(super) type VSponge = PoseidonSponge<VCf>;
    pub(super) type VAS = ASForR1CSNark<VG, VSponge>;

    // Forward direction: the verified curve is Pallas, so the verifier circuit's
    // field — and thus the Vesta NARK's scalar field, `CF` here — is
    // `ConstraintF<Pallas> = ark_pallas::Fq = ark_vesta::Fr`.
    type CF = super::CF<Pallas>;

    /// The verifier circuit's public input — its `instance_assignment`, captured
    /// with the same optimization goal the NARK prover uses, so `verify` sees the
    /// same vector. Over the scalar field `CF` (= the Vesta NARK's scalar field).
    fn public_input(make_zk: bool, seed: u64) -> Vec<CF> {
        let pcs = ConstraintSystem::<CF>::new_ref();
        pcs.set_optimization_goal(OptimizationGoal::Constraints);
        pcs.set_mode(SynthesisMode::Prove {
            construct_matrices: false,
        });
        build_step::<Pallas>(make_zk, seed)
            .generate_constraints(pcs.clone())
            .unwrap();
        pcs.finalize();
        let input = pcs.borrow().unwrap().instance_assignment.clone();
        input
    }

    /// Vesta NARK setup + index over the recursion circuit. Its structure is
    /// fixed by `make_zk` (seed-independent), so one index serves every
    /// same-`make_zk` recursion input.
    fn nark_index(
        make_zk: bool,
        seed: u64,
    ) -> (IndexProverKey<VG>, IndexVerifierKey<VG>) {
        let nark_pp = ZkxR1CSNark::<VG, VSponge>::setup();
        ZkxR1CSNark::<VG, VSponge>::index(&nark_pp, build_step::<Pallas>(make_zk, seed)).unwrap()
    }

    /// Prove the Pallas-verifier circuit on Vesta, seeded so the fixture replays
    /// identical randomness. `fork` selects the gamma sponge: a **standalone** NARK
    /// (the half-step, verified by `R1CSNark::verify`) uses the plain
    /// `VSponge::new()` (`fork = false`); a NARK destined to be **folded** by the
    /// AS must use the AS's forked `nark_sponge` (`fork = true`) so its gamma
    /// matches the one the AS recomputes for the blinded commitments — otherwise
    /// the input's `blinded_witness` and the folded commitments use different
    /// gammas and the accumulator, while it verifies, would not decide.
    fn nark_prove(
        ipk: &IndexProverKey<VG>,
        make_zk: bool,
        seed: u64,
        fork: bool,
    ) -> NarkProof<VG> {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5ec2);
        let nark_sponge = if fork {
            VAS::nark_sponge(&VSponge::new())
        } else {
            VSponge::new()
        };
        ZkxR1CSNark::<VG, VSponge>::prove(
            ipk,
            build_step::<Pallas>(make_zk, seed),
            make_zk,
            Some(nark_sponge),
            Some(&mut rng),
        )
        .unwrap()
    }

    /// One Vesta-AS accumulation `Input`: the recursion circuit's Vesta NARK
    /// proof packaged as `(r1cs_input, first_round_message,
    /// witness = second_round_message)` — the verifier proof to be accumulated.
    fn recursion_input(
        ipk: &IndexProverKey<VG>,
        make_zk: bool,
        seed: u64,
    ) -> Input<VCf, VSponge, VAS> {
        // The AS folds this input, so its NARK must use the forked nark_sponge.
        let proof = nark_prove(ipk, make_zk, seed, true);
        Input::<VCf, VSponge, VAS> {
            instance: InputInstance {
                r1cs_input: public_input(make_zk, seed),
                first_round_message: proof.first_msg,
            },
            witness: proof.second_msg,
        }
    }

    /// The two operands of a full IVC step, built once and reused by the
    /// fixture-dump fold so the byte-match isolates the folding accumulation.
    pub(super) struct FoldOperands {
        pub pk: <VAS as AccumulationScheme<VCf, VSponge>>::ProverKey,
        pub vk: <VAS as AccumulationScheme<VCf, VSponge>>::VerifierKey,
        pub input_cur: Input<VCf, VSponge, VAS>,
        pub acc_prev: Accumulator<VCf, VSponge, VAS>,
    }

    /// Build a full IVC step: the current step's verifier-proof input and a
    /// prior Vesta accumulator (an init accumulation of a different-seed
    /// recursion NARK over the same index — the accumulator this step folds).
    pub(super) fn fold_operands(make_zk: bool, seed: u64) -> FoldOperands {
        let (ipk, ivk) = nark_index(make_zk, seed);

        let mut rng = StdRng::seed_from_u64(seed ^ 0x1cc0);
        let as_pp = VAS::setup(&mut rng).unwrap();
        let (pk, vk, _dk) = VAS::index(&as_pp, &(), &(ipk.clone(), ivk.clone())).unwrap();

        // Current step: the verifier proof accumulated this step.
        let input_cur = recursion_input(&ipk, make_zk, seed);

        // Prior step: init-accumulate a different-seed recursion NARK to get the
        // accumulator the current step folds into.
        let input_prev = recursion_input(&ipk, make_zk, seed.wrapping_add(1));
        let prev_inputs = vec![input_prev];
        let no_acc: Vec<Accumulator<VCf, VSponge, VAS>> = Vec::new();
        let mut acc_rng = StdRng::seed_from_u64(seed ^ 0xacc0);
        let make_zk_prev = if make_zk {
            MakeZK::Enabled(&mut acc_rng)
        } else {
            MakeZK::Disabled
        };
        let (acc_prev, _proof_prev) = VAS::prove(
            &pk,
            Input::<VCf, VSponge, VAS>::map_to_refs(&prev_inputs),
            Accumulator::<VCf, VSponge, VAS>::map_to_refs(&no_acc),
            make_zk_prev,
            None,
        )
        .unwrap();

        FoldOperands {
            pk,
            vk,
            input_cur,
            acc_prev,
        }
    }

    /// Informational timing: the arkworks full IVC fold
    /// **prove** at recursion scale, `--release`, to pair with the warm fused-GPU
    /// fold timing (`tests/gpu_fused_fold_bench.rs`) for the GPU-vs-arkworks figure.
    /// Times only `VAS::prove` + serialize (the same scope
    /// as the fused consumer — no `AS::verify`); the slow `fold_operands` recursion
    /// synthesis is one-time setup, excluded. `#[ignore]`d and meant for `--release`:
    ///
    ///     cargo test --release --features recursion --test recursion_step \
    ///       vesta::arkworks_fold_timing -- --ignored --nocapture
    #[test]
    #[ignore = "--release arkworks fold-prove timing at recursion scale (slow arkworks fold setup)"]
    fn arkworks_fold_timing() {
        let ops = fold_operands(true, 0); // one-time setup, not timed
        let prove_once = || {
            let cur = vec![ops.input_cur.clone()];
            let prev = vec![ops.acc_prev.clone()];
            let mut rng = StdRng::seed_from_u64(0xf01d);
            let (acc_new, proof) = VAS::prove(
                &ops.pk,
                Input::<VCf, VSponge, VAS>::map_to_refs(&cur),
                Accumulator::<VCf, VSponge, VAS>::map_to_refs(&prev),
                MakeZK::Enabled(&mut rng),
                None,
            )
            .unwrap();
            let mut bytes = Vec::new();
            acc_new.instance.serialize(&mut bytes).unwrap();
            acc_new.witness.serialize(&mut bytes).unwrap();
            proof.serialize(&mut bytes).unwrap();
            bytes.len()
        };
        let n_bytes = prove_once(); // warm
        const N: usize = 5;
        let mut times: Vec<f64> = Vec::with_capacity(N);
        for _ in 0..N {
            let t = std::time::Instant::now();
            let _ = prove_once();
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "arkworks full IVC fold prove (Vesta, make_zk, ~77.5K constraints, {n_bytes}B) \
             --release: median {:.0} ms over {N} runs (min {:.0}, max {:.0})",
            times[N / 2],
            times[0],
            times[N - 1],
        );
    }

    /// Dump the forward recursion-circuit no-zk Vesta NARK
    /// to an off-tree JSON fixture, for the jax `nark.prove_no_zk` byte-match. The
    /// circuit is the AS verifier gadget (~22.5K constraints × ~21K vars, but
    /// sparse), so the fixture is large and written **off-tree**
    /// (`$ACCUMULATION_ZORCH_ARTIFACTS`, default `artifacts/`); the Python gate
    /// skips when it is absent, the same way the `#[ignore]` GPU gates do. Schema
    /// matches `nark_fixtures.json` so the Python loader is shared.
    mod dump {
        use super::{
            build_step, fold_operands, nark_index, nark_prove, Pallas, VCf, VSponge, VAS, VG,
        };
        use ark_accumulation::r1cs_nark_as::r1cs_nark::R1CSNark as ZkxR1CSNark;
        use ark_accumulation::{AccumulationScheme, Accumulator, Input, MakeZK};
        use ark_ff::{BigInteger, PrimeField, UniformRand, Zero};
        use ark_poly_commit::trivial_pc::PedersenCommitment;
        use ark_relations::r1cs::{
            ConstraintSynthesizer, ConstraintSystem, Matrix, OptimizationGoal, SynthesisMode,
        };
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
        use ark_std::rand::{rngs::StdRng, SeedableRng};
        use blake2::VarBlake2b;
        use digest::{Update, VariableOutput};

        // The Vesta NARK's scalar field = the verifier circuit's field
        // (ark_pallas::Fq = ark_vesta::Fr); the matrices / assignment are over it.
        type CF = super::CF;

        fn hex(bytes: &[u8]) -> String {
            let mut s = String::with_capacity(bytes.len() * 2);
            for b in bytes {
                s.push_str(&format!("{:02x}", b));
            }
            s
        }
        fn fe_hex(f: &CF) -> String {
            hex(&f.into_repr().to_bytes_le())
        }
        fn ser_hex<T: CanonicalSerialize>(v: &T) -> String {
            let mut b = Vec::new();
            v.serialize(&mut b).unwrap();
            hex(&b)
        }
        fn matrix_json(m: &Matrix<CF>) -> String {
            let rows: Vec<String> = m
                .iter()
                .map(|row| {
                    let e: Vec<String> = row
                        .iter()
                        .map(|(c, i)| format!("[\"{}\",{}]", fe_hex(c), i))
                        .collect();
                    format!("[{}]", e.join(","))
                })
                .collect();
            format!("[{}]", rows.join(","))
        }
        fn fr_list_json(xs: &[CF]) -> String {
            let v: Vec<String> = xs.iter().map(|f| format!("\"{}\"", fe_hex(f))).collect();
            format!("[{}]", v.join(","))
        }
        fn xy(p: &VG) -> (String, String) {
            if p.is_zero() {
                (hex(&[0u8; 32]), hex(&[0u8; 32]))
            } else {
                (hex(&p.x.into_repr().to_bytes_le()), hex(&p.y.into_repr().to_bytes_le()))
            }
        }
        fn point_json(p: &VG) -> String {
            let (x, y) = xy(p);
            format!("{{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}}", x, y)
        }

        // The AS-level matrices-hash domain (the fold's `beta` sponge forks it),
        // alongside the NARK domain `dump_recursion_nark_zk` already uses.
        const AS_PROTOCOL_NAME: &[u8] = b"AS-FOR-R1CS-NARK-2020";

        /// Parse a serialized `AccumulatorInstance` into structured JSON components
        /// (points uncompressed, so jax needs no decompression). Field order is the
        /// derived one: `r1cs_input: Vec<CF>`, `comm_a/comm_b/comm_c: VG`, then
        /// `hp_instance` = `hp_comm_1/2/3: VG`.
        fn acc_instance_json(bytes: &[u8]) -> String {
            let mut cur = bytes;
            let r1cs_input = Vec::<CF>::deserialize(&mut cur).unwrap();
            let comm_a = VG::deserialize(&mut cur).unwrap();
            let comm_b = VG::deserialize(&mut cur).unwrap();
            let comm_c = VG::deserialize(&mut cur).unwrap();
            let hp_comm_1 = VG::deserialize(&mut cur).unwrap();
            let hp_comm_2 = VG::deserialize(&mut cur).unwrap();
            let hp_comm_3 = VG::deserialize(&mut cur).unwrap();
            format!(
                "{{\"r1cs_input\":{},\"comm_a\":{},\"comm_b\":{},\"comm_c\":{},\
                 \"hp_comm_1\":{},\"hp_comm_2\":{},\"hp_comm_3\":{}}}",
                fr_list_json(&r1cs_input),
                point_json(&comm_a), point_json(&comm_b), point_json(&comm_c),
                point_json(&hp_comm_1), point_json(&hp_comm_2), point_json(&hp_comm_3),
            )
        }

        /// Parse a serialized `AccumulatorWitness` into structured JSON. Derived
        /// field order: `r1cs_blinded_witness: Vec<CF>`, then `hp_witness`
        /// (`a_vec: Vec<CF>`, `b_vec: Vec<CF>`, `randomness: Option<{rand_1,2,3}>`),
        /// then `randomness: Option<{sigma_a,sigma_b,sigma_c}>` (zk forces both Some).
        fn acc_witness_json(bytes: &[u8]) -> String {
            let mut cur = bytes;
            let r1cs_blinded_witness = Vec::<CF>::deserialize(&mut cur).unwrap();
            let hp_a_vec = Vec::<CF>::deserialize(&mut cur).unwrap();
            let hp_b_vec = Vec::<CF>::deserialize(&mut cur).unwrap();
            let read_opt3 = |cur: &mut &[u8]| -> (CF, CF, CF) {
                let flag = u8::deserialize(&mut *cur).unwrap();
                assert_eq!(flag, 1, "zk fold forces Some randomness");
                (
                    CF::deserialize(&mut *cur).unwrap(),
                    CF::deserialize(&mut *cur).unwrap(),
                    CF::deserialize(&mut *cur).unwrap(),
                )
            };
            let (hp_rand_1, hp_rand_2, hp_rand_3) = read_opt3(&mut cur);
            let (sigma_a, sigma_b, sigma_c) = read_opt3(&mut cur);
            format!(
                "{{\"r1cs_blinded_witness\":{},\"hp_a_vec\":{},\"hp_b_vec\":{},\
                 \"hp_rand_1\":\"{}\",\"hp_rand_2\":\"{}\",\"hp_rand_3\":\"{}\",\
                 \"sigma_a\":\"{}\",\"sigma_b\":\"{}\",\"sigma_c\":\"{}\"}}",
                fr_list_json(&r1cs_blinded_witness),
                fr_list_json(&hp_a_vec), fr_list_json(&hp_b_vec),
                fe_hex(&hp_rand_1), fe_hex(&hp_rand_2), fe_hex(&hp_rand_3),
                fe_hex(&sigma_a), fe_hex(&sigma_b), fe_hex(&sigma_c),
            )
        }

        /// `cargo test --features recursion --test recursion_step dump_recursion_nark`
        /// (writes `$ACCUMULATION_ZORCH_ARTIFACTS/recursion_nark_fixtures.json`).
        #[test]
        fn dump_recursion_nark() {
            let out_dir = std::env::var("ACCUMULATION_ZORCH_ARTIFACTS")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| std::path::PathBuf::from("artifacts"));
            std::fs::create_dir_all(&out_dir).unwrap();
            let (make_zk, seed) = (false, 0u64);

            // Golden no-zk Vesta NARK proof over the forward recursion circuit
            // (sponge irrelevant for no-zk: gamma is computed but unused → None).
            let pp = ZkxR1CSNark::<VG, VSponge>::setup();
            let (ipk, _ivk) =
                ZkxR1CSNark::<VG, VSponge>::index(&pp, build_step::<Pallas>(make_zk, seed)).unwrap();
            let proof = ZkxR1CSNark::<VG, VSponge>::prove(
                &ipk,
                build_step::<Pallas>(make_zk, seed),
                false,
                None,
                None,
            )
            .unwrap();
            let proof_hex = ser_hex(&proof);

            // Matrices a/b/c (Constraints + Setup), over CF.
            let mcs = ConstraintSystem::<CF>::new_ref();
            mcs.set_optimization_goal(OptimizationGoal::Constraints);
            mcs.set_mode(SynthesisMode::Setup);
            build_step::<Pallas>(make_zk, seed)
                .generate_constraints(mcs.clone())
                .unwrap();
            mcs.finalize();
            let matrices = mcs.to_matrices().unwrap();

            // Instance + witness assignment (Constraints + Prove, no matrices).
            let pcs = ConstraintSystem::<CF>::new_ref();
            pcs.set_optimization_goal(OptimizationGoal::Constraints);
            pcs.set_mode(SynthesisMode::Prove {
                construct_matrices: false,
            });
            build_step::<Pallas>(make_zk, seed)
                .generate_constraints(pcs.clone())
                .unwrap();
            pcs.finalize();
            let (input, witness, num_constraints) = {
                let cs = pcs.borrow().unwrap();
                (
                    cs.instance_assignment.clone(),
                    cs.witness_assignment.clone(),
                    cs.num_constraints,
                )
            };

            // Committer-key generators (recovered uncompressed, like dump_nark).
            let cpp = PedersenCommitment::<VG>::setup(num_constraints);
            let ck = PedersenCommitment::<VG>::trim(&cpp, num_constraints);
            let generators: Vec<VG> = {
                let mut b = Vec::new();
                ck.serialize_uncompressed(&mut b).unwrap();
                let mut r = &b[..];
                Vec::<VG>::deserialize_uncompressed(&mut r).unwrap()
            };
            let gens_json: Vec<String> = generators
                .iter()
                .map(|p| {
                    let (x, y) = xy(p);
                    format!("{{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}}", x, y)
                })
                .collect();

            let json = format!(
                concat!(
                    "{{\n  \"note\": \"recursion-circuit no-zk Vesta NARK\",\n",
                    "  \"curve\": \"vesta\",\n  \"num_constraints\": {},\n  \"num_vars\": {},\n",
                    "  \"a\": {},\n  \"b\": {},\n  \"c\": {},\n  \"input\": {},\n  \"witness\": {},\n",
                    "  \"generators\": [{}],\n  \"proof_hex\": \"{}\"\n}}\n"
                ),
                num_constraints,
                input.len() + witness.len(),
                matrix_json(&matrices.a),
                matrix_json(&matrices.b),
                matrix_json(&matrices.c),
                fr_list_json(&input),
                fr_list_json(&witness),
                gens_json.join(","),
                proof_hex,
            );
            let path = out_dir.join("recursion_nark_fixtures.json");
            std::fs::write(&path, json).unwrap();
            eprintln!(
                "[dump] wrote {} ({} constraints, {} generators, proof {}B)",
                path.display(),
                num_constraints,
                generators.len(),
                proof_hex.len() / 2
            );
        }

        const NARK_PROTOCOL_NAME: &[u8] = b"R1CS-NARK-2020";

        /// `r1cs_nark::hash_matrices` (the crate's is `pub(crate)`): blake2b-256
        /// over the protocol domain then the canonical-serialized a/b/c. This is
        /// the `matrices_hash` the gamma challenge absorbs.
        fn hash_matrices(domain: &[u8], a: &Matrix<CF>, b: &Matrix<CF>, c: &Matrix<CF>) -> [u8; 32] {
            let mut serialized = domain.to_vec();
            a.serialize(&mut serialized).unwrap();
            b.serialize(&mut serialized).unwrap();
            c.serialize(&mut serialized).unwrap();
            let mut hasher = VarBlake2b::new(32).unwrap();
            hasher.update(&serialized);
            let mut out = [0u8; 32];
            hasher.finalize_variable(|res| out.copy_from_slice(res));
            out
        }

        /// `cargo test --features recursion --test recursion_step dump_recursion_nark_zk`
        /// (writes `$ACCUMULATION_ZORCH_ARTIFACTS/recursion_nark_zk_fixtures.json`).
        ///
        /// The zk half-step's golden IS `nark_prove` — the exact
        /// subject the GPU gate (`recursion_step_proves_on_vesta`, make_zk=true)
        /// byte-matches — so the fused export reproduces that proof. The prover's
        /// sampled randomness (the `r` witness blinders + 8 sigma blinders) is
        /// recovered by replaying its `Fr::rand` schedule on a fresh rng with
        /// `nark_prove`'s own seed (`seed ^ 0x5ec2`). Unlike the toy
        /// `dump_nark_zk`, the gamma sponge here is the plain `VSponge::new()`
        /// `nark_prove` passes — **unforked**: a standalone NARK draws no
        /// AS-level `nark_sponge` fork.
        #[test]
        fn dump_recursion_nark_zk() {
            let out_dir = std::env::var("ACCUMULATION_ZORCH_ARTIFACTS")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| std::path::PathBuf::from("artifacts"));
            std::fs::create_dir_all(&out_dir).unwrap();
            let (make_zk, seed) = (true, 0u64);

            // Golden zk Vesta NARK proof = the GPU byte-match subject itself.
            let (ipk, _ivk) = nark_index(make_zk, seed);
            let proof = nark_prove(&ipk, make_zk, seed, false);
            let proof_hex = ser_hex(&proof);

            // Matrices a/b/c (Constraints + Setup) over CF — the make_zk=true
            // circuit (its structure differs from no-zk: extra blinding vars).
            let mcs = ConstraintSystem::<CF>::new_ref();
            mcs.set_optimization_goal(OptimizationGoal::Constraints);
            mcs.set_mode(SynthesisMode::Setup);
            build_step::<Pallas>(make_zk, seed)
                .generate_constraints(mcs.clone())
                .unwrap();
            mcs.finalize();
            let matrices = mcs.to_matrices().unwrap();
            let nark_matrices_hash =
                hash_matrices(NARK_PROTOCOL_NAME, &matrices.a, &matrices.b, &matrices.c);

            // Instance + witness assignment (Constraints + Prove, no matrices).
            let pcs = ConstraintSystem::<CF>::new_ref();
            pcs.set_optimization_goal(OptimizationGoal::Constraints);
            pcs.set_mode(SynthesisMode::Prove {
                construct_matrices: false,
            });
            build_step::<Pallas>(make_zk, seed)
                .generate_constraints(pcs.clone())
                .unwrap();
            pcs.finalize();
            let (input, witness, num_constraints) = {
                let cs = pcs.borrow().unwrap();
                (
                    cs.instance_assignment.clone(),
                    cs.witness_assignment.clone(),
                    cs.num_constraints,
                )
            };
            let num_witness = witness.len();

            // Committer key: generators + the hiding generator (zk commits with a
            // blinder appended on the hiding base).
            let cpp = PedersenCommitment::<VG>::setup(num_constraints);
            let ck = PedersenCommitment::<VG>::trim(&cpp, num_constraints);
            let (generators, hiding): (Vec<VG>, VG) = {
                let mut b = Vec::new();
                ck.serialize_uncompressed(&mut b).unwrap();
                let mut r = &b[..];
                let g = Vec::<VG>::deserialize_uncompressed(&mut r).unwrap();
                let h = VG::deserialize_uncompressed(&mut r).unwrap();
                (g, h)
            };
            let gens_json: Vec<String> = generators
                .iter()
                .map(|p| {
                    let (x, y) = xy(p);
                    format!("{{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}}", x, y)
                })
                .collect();
            let (hx, hy) = xy(&hiding);

            // Replay nark_prove's make_zk draw schedule on a fresh same-seed rng
            // (nark_prove seeds with `seed ^ 0x5ec2`): `r` (one per witness var),
            // then a/b/c_blinder, r_a/r_b/r_c_blinder, blinder_1, blinder_2.
            let mut rep = StdRng::seed_from_u64(seed ^ 0x5ec2);
            let r: Vec<CF> = (0..num_witness).map(|_| CF::rand(&mut rep)).collect();
            let a_blinder = CF::rand(&mut rep);
            let b_blinder = CF::rand(&mut rep);
            let c_blinder = CF::rand(&mut rep);
            let r_a_blinder = CF::rand(&mut rep);
            let r_b_blinder = CF::rand(&mut rep);
            let r_c_blinder = CF::rand(&mut rep);
            let blinder_1 = CF::rand(&mut rep);
            let blinder_2 = CF::rand(&mut rep);

            let json = format!(
                concat!(
                    "{{\n  \"note\": \"recursion-circuit zk Vesta NARK (unforked sponge)\",\n",
                    "  \"curve\": \"vesta\",\n  \"num_constraints\": {},\n  \"num_vars\": {},\n",
                    "  \"nark_matrices_hash_hex\": \"{}\",\n",
                    "  \"a\": {},\n  \"b\": {},\n  \"c\": {},\n  \"input\": {},\n  \"witness\": {},\n",
                    "  \"generators\": [{}],\n  \"hiding\": {{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}},\n",
                    "  \"r\": {},\n",
                    "  \"a_blinder\": \"{}\",\n  \"b_blinder\": \"{}\",\n  \"c_blinder\": \"{}\",\n",
                    "  \"r_a_blinder\": \"{}\",\n  \"r_b_blinder\": \"{}\",\n  \"r_c_blinder\": \"{}\",\n",
                    "  \"blinder_1\": \"{}\",\n  \"blinder_2\": \"{}\",\n",
                    "  \"proof_hex\": \"{}\"\n}}\n"
                ),
                num_constraints,
                input.len() + witness.len(),
                hex(&nark_matrices_hash),
                matrix_json(&matrices.a),
                matrix_json(&matrices.b),
                matrix_json(&matrices.c),
                fr_list_json(&input),
                fr_list_json(&witness),
                gens_json.join(","),
                hx,
                hy,
                fr_list_json(&r),
                fe_hex(&a_blinder),
                fe_hex(&b_blinder),
                fe_hex(&c_blinder),
                fe_hex(&r_a_blinder),
                fe_hex(&r_b_blinder),
                fe_hex(&r_c_blinder),
                fe_hex(&blinder_1),
                fe_hex(&blinder_2),
                proof_hex,
            );
            let path = out_dir.join("recursion_nark_zk_fixtures.json");
            std::fs::write(&path, json).unwrap();
            eprintln!(
                "[dump] wrote {} ({} constraints, {} vars, {} generators, proof {}B)",
                path.display(),
                num_constraints,
                input.len() + witness.len(),
                generators.len(),
                proof_hex.len() / 2
            );
        }

        /// Dump the forward recursion-circuit zk **fold** (one
        /// input folded INTO one prior Vesta accumulator, num_addends=3) for the
        /// jax `r1cs_nark_as.prove_zk_fold(VESTA, …)` end-to-end byte-match — the
        /// recursion-scale analog of the toy `dump_as_fold_zk`. Replays the input's
        /// NARK randomness (`seed ^ 0x5ec2`) and the fold's AS + HP randomness
        /// (`seed ^ 0xf01d`); `acc_prev` is fed as its parsed instance/witness
        /// components (a materialized prior accumulator). Off-tree (large), gated
        /// `--features recursion`; the Python side skips when absent.
        ///
        /// `cargo test --features recursion --test recursion_step \
        ///   vesta::dump::dump_recursion_fold_zk` (writes
        /// `$ACCUMULATION_ZORCH_ARTIFACTS/recursion_fold_zk_fixtures.json`).
        #[test]
        fn dump_recursion_fold_zk() {
            let out_dir = std::env::var("ACCUMULATION_ZORCH_ARTIFACTS")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| std::path::PathBuf::from("artifacts"));
            std::fs::create_dir_all(&out_dir).unwrap();
            let (make_zk, seed) = (true, 0u64);

            // Forward recursion circuit: matrices (Setup) + the current input's raw
            // assignment (Prove), over the Vesta NARK scalar field CF.
            let mcs = ConstraintSystem::<CF>::new_ref();
            mcs.set_optimization_goal(OptimizationGoal::Constraints);
            mcs.set_mode(SynthesisMode::Setup);
            build_step::<Pallas>(make_zk, seed).generate_constraints(mcs.clone()).unwrap();
            mcs.finalize();
            let matrices = mcs.to_matrices().unwrap();
            let nark_matrices_hash =
                hash_matrices(NARK_PROTOCOL_NAME, &matrices.a, &matrices.b, &matrices.c);
            let as_matrices_hash =
                hash_matrices(AS_PROTOCOL_NAME, &matrices.a, &matrices.b, &matrices.c);

            let pcs = ConstraintSystem::<CF>::new_ref();
            pcs.set_optimization_goal(OptimizationGoal::Constraints);
            pcs.set_mode(SynthesisMode::Prove { construct_matrices: false });
            build_step::<Pallas>(make_zk, seed).generate_constraints(pcs.clone()).unwrap();
            pcs.finalize();
            let (input2, witness2, num_constraints) = {
                let cs = pcs.borrow().unwrap();
                (cs.instance_assignment.clone(), cs.witness_assignment.clone(), cs.num_constraints)
            };
            let num_witness = witness2.len();

            // Committer key (generators + hiding base).
            let cpp = PedersenCommitment::<VG>::setup(num_constraints);
            let ck = PedersenCommitment::<VG>::trim(&cpp, num_constraints);
            let supported_num_elems = ck.supported_num_elems();
            let (generators, hiding): (Vec<VG>, VG) = {
                let mut b = Vec::new();
                ck.serialize_uncompressed(&mut b).unwrap();
                let mut r = &b[..];
                let g = Vec::<VG>::deserialize_uncompressed(&mut r).unwrap();
                let h = VG::deserialize_uncompressed(&mut r).unwrap();
                (g, h)
            };
            let gens_json: Vec<String> = generators.iter().map(point_json).collect();
            let (hx, hy) = xy(&hiding);

            // The fold operands (input_cur + the prior Vesta accumulator) and the
            // golden fold (seed ^ 0xf01d, matching `fold_step`).
            let ops = fold_operands(make_zk, seed);
            let acc_prev_instance_json = {
                let mut b = Vec::new();
                ops.acc_prev.instance.serialize(&mut b).unwrap();
                acc_instance_json(&b)
            };
            let acc_prev_witness_json = {
                let mut b = Vec::new();
                ops.acc_prev.witness.serialize(&mut b).unwrap();
                acc_witness_json(&b)
            };
            let cur_inputs = vec![ops.input_cur.clone()];
            let prev_accs = vec![ops.acc_prev.clone()];
            let mut rng_fold = StdRng::seed_from_u64(seed ^ 0xf01d);
            let (golden_acc, golden_proof) = VAS::prove(
                &ops.pk,
                Input::<VCf, VSponge, VAS>::map_to_refs(&cur_inputs),
                Accumulator::<VCf, VSponge, VAS>::map_to_refs(&prev_accs),
                MakeZK::Enabled(&mut rng_fold),
                None,
            )
            .unwrap();
            let verified = VAS::verify(
                &ops.vk,
                vec![&ops.input_cur.instance],
                vec![&ops.acc_prev.instance],
                &golden_acc.instance,
                &golden_proof,
                None,
            )
            .unwrap();
            assert!(verified, "golden recursion fold failed to verify");

            let golden_instance_hex = ser_hex(&golden_acc.instance);
            let golden_witness_hex = ser_hex(&golden_acc.witness);
            let golden_proof_hex = ser_hex(&golden_proof);

            // Replay input_cur's NARK randomness (seed ^ 0x5ec2): r × num_witness,
            // then the 8 blinders.
            let mut rep2 = StdRng::seed_from_u64(seed ^ 0x5ec2);
            let r: Vec<CF> = (0..num_witness).map(|_| CF::rand(&mut rep2)).collect();
            let nark_blinders: Vec<CF> = (0..8).map(|_| CF::rand(&mut rep2)).collect();

            // Replay the fold's AS + HP randomness (seed ^ 0xf01d): the AS
            // proof-randomness, then the HP hiding randomness.
            let mut rep_fold = StdRng::seed_from_u64(seed ^ 0xf01d);
            let as_r1cs_r_input = CF::rand(&mut rep_fold);
            let as_r1cs_r_witness = CF::rand(&mut rep_fold);
            let as_rand: Vec<CF> = (0..3).map(|_| CF::rand(&mut rep_fold)).collect();
            let hp_hiding_a = CF::rand(&mut rep_fold);
            let hp_hiding_b = CF::rand(&mut rep_fold);
            let hp_rand: Vec<CF> = (0..3).map(|_| CF::rand(&mut rep_fold)).collect();

            let json = format!(
                concat!(
                    "{{\n  \"note\": \"recursion-circuit zk Vesta fold (num_addends=3)\",\n",
                    "  \"curve\": \"vesta\",\n  \"num_constraints\": {},\n  \"num_vars\": {},\n",
                    "  \"supported_num_elems\": {},\n",
                    "  \"nark_matrices_hash_hex\": \"{}\",\n  \"as_matrices_hash_hex\": \"{}\",\n",
                    "  \"a\": {},\n  \"b\": {},\n  \"c\": {},\n",
                    "  \"generators\": [{}],\n  \"hiding\": {{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}},\n",
                    "  \"input2_r1cs_input\": {},\n  \"input2_witness\": {},\n",
                    "  \"r\": {},\n",
                    "  \"a_blinder\": \"{}\",\n  \"b_blinder\": \"{}\",\n  \"c_blinder\": \"{}\",\n",
                    "  \"r_a_blinder\": \"{}\",\n  \"r_b_blinder\": \"{}\",\n  \"r_c_blinder\": \"{}\",\n",
                    "  \"blinder_1\": \"{}\",\n  \"blinder_2\": \"{}\",\n",
                    "  \"as_r1cs_r_input\": \"{}\",\n  \"as_r1cs_r_witness\": \"{}\",\n",
                    "  \"as_rand_1\": \"{}\",\n  \"as_rand_2\": \"{}\",\n  \"as_rand_3\": \"{}\",\n",
                    "  \"hp_hiding_a\": \"{}\",\n  \"hp_hiding_b\": \"{}\",\n",
                    "  \"hp_rand_1\": \"{}\",\n  \"hp_rand_2\": \"{}\",\n  \"hp_rand_3\": \"{}\",\n",
                    "  \"acc_prev_instance\": {},\n  \"acc_prev_witness\": {},\n",
                    "  \"golden_instance_hex\": \"{}\",\n  \"golden_witness_hex\": \"{}\",\n",
                    "  \"golden_proof_hex\": \"{}\"\n}}\n"
                ),
                num_constraints,
                input2.len() + witness2.len(),
                supported_num_elems,
                hex(&nark_matrices_hash),
                hex(&as_matrices_hash),
                matrix_json(&matrices.a),
                matrix_json(&matrices.b),
                matrix_json(&matrices.c),
                gens_json.join(","),
                hx,
                hy,
                fr_list_json(&input2),
                fr_list_json(&witness2),
                fr_list_json(&r),
                fe_hex(&nark_blinders[0]), fe_hex(&nark_blinders[1]), fe_hex(&nark_blinders[2]),
                fe_hex(&nark_blinders[3]), fe_hex(&nark_blinders[4]), fe_hex(&nark_blinders[5]),
                fe_hex(&nark_blinders[6]), fe_hex(&nark_blinders[7]),
                fe_hex(&as_r1cs_r_input), fe_hex(&as_r1cs_r_witness),
                fe_hex(&as_rand[0]), fe_hex(&as_rand[1]), fe_hex(&as_rand[2]),
                fe_hex(&hp_hiding_a), fe_hex(&hp_hiding_b),
                fe_hex(&hp_rand[0]), fe_hex(&hp_rand[1]), fe_hex(&hp_rand[2]),
                acc_prev_instance_json, acc_prev_witness_json,
                golden_instance_hex, golden_witness_hex, golden_proof_hex,
            );
            let path = out_dir.join("recursion_fold_zk_fixtures.json");
            std::fs::write(&path, json).unwrap();
            eprintln!(
                "[dump] wrote {} ({} constraints, {} vars, fold acc.instance {}B / witness {}B / proof {}B)",
                path.display(),
                num_constraints,
                input2.len() + witness2.len(),
                golden_instance_hex.len() / 2,
                golden_witness_hex.len() / 2,
                golden_proof_hex.len() / 2,
            );
        }
    }
}

/// Shared Pallas-side plumbing for the reverse full IVC step. Curve-fixed to
/// Pallas — the cycle curve the Vesta-verifier circuit proves on — driving the
/// arkworks prover on CPU to emit the golden fixtures. Mirror of [`vesta`] with
/// the cycle curves swapped.
mod pallas {
    use super::{build_step, Vesta};
    use ark_accumulation::r1cs_nark_as::r1cs_nark::{
        IndexProverKey, IndexVerifierKey, Proof as NarkProof, R1CSNark as ZkxR1CSNark,
    };
    use ark_accumulation::r1cs_nark_as::{ASForR1CSNark, InputInstance};
    use ark_accumulation::{AccumulationScheme, Accumulator, Input, MakeZK};
    use ark_relations::r1cs::{
        ConstraintSynthesizer, ConstraintSystem, OptimizationGoal, SynthesisMode,
    };
    use ark_sponge::poseidon::PoseidonSponge;
    use ark_sponge::CryptographicSponge;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    // The verifier circuit is over ark_vesta::Fq = ark_pallas::Fr (the Pallas
    // scalar field), so it proves with a Pallas NARK whose constraint field — and
    // that of the accumulation scheme over it — is ark_pallas::Fq (`PCf`,
    // = ark_vesta::Fr). The r1cs_input the accumulation input carries is still
    // over the scalar field, i.e. `CF`.
    pub(super) type PG = ark_pallas::Affine;
    pub(super) type PCf = ark_pallas::Fq;
    pub(super) type PSponge = PoseidonSponge<PCf>;
    pub(super) type PAS = ASForR1CSNark<PG, PSponge>;

    // Reverse direction: the verified curve is Vesta, so the verifier circuit's
    // field — and thus the Pallas NARK's scalar field, `CF` here — is
    // `ConstraintF<Vesta> = ark_vesta::Fq = ark_pallas::Fr`.
    type CF = super::CF<Vesta>;

    /// The verifier circuit's public input — its `instance_assignment`, captured
    /// with the same optimization goal the NARK prover uses, so `verify` sees the
    /// same vector. Over the scalar field `CF` (= the Pallas NARK's scalar field).
    fn public_input(make_zk: bool, seed: u64) -> Vec<CF> {
        let pcs = ConstraintSystem::<CF>::new_ref();
        pcs.set_optimization_goal(OptimizationGoal::Constraints);
        pcs.set_mode(SynthesisMode::Prove {
            construct_matrices: false,
        });
        build_step::<Vesta>(make_zk, seed)
            .generate_constraints(pcs.clone())
            .unwrap();
        pcs.finalize();
        let input = pcs.borrow().unwrap().instance_assignment.clone();
        input
    }

    /// Pallas NARK setup + index over the recursion circuit. Its structure is
    /// fixed by `make_zk` (seed-independent), so one index serves every
    /// same-`make_zk` recursion input.
    fn nark_index(make_zk: bool, seed: u64) -> (IndexProverKey<PG>, IndexVerifierKey<PG>) {
        let nark_pp = ZkxR1CSNark::<PG, PSponge>::setup();
        ZkxR1CSNark::<PG, PSponge>::index(&nark_pp, build_step::<Vesta>(make_zk, seed)).unwrap()
    }

    /// Prove the Vesta-verifier circuit on Pallas, seeded so the fixture replays
    /// identical randomness. `fork` selects the gamma sponge: a NARK destined to be **folded**
    /// by the AS must use the AS's forked `nark_sponge` (`fork = true`) so its
    /// gamma matches the one the AS recomputes for the blinded commitments;
    /// a standalone NARK would use the plain `PSponge::new()`. See the `vesta`
    /// twin for the full rationale.
    fn nark_prove(
        ipk: &IndexProverKey<PG>,
        make_zk: bool,
        seed: u64,
        fork: bool,
    ) -> NarkProof<PG> {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5ec2);
        let nark_sponge = if fork {
            PAS::nark_sponge(&PSponge::new())
        } else {
            PSponge::new()
        };
        ZkxR1CSNark::<PG, PSponge>::prove(
            ipk,
            build_step::<Vesta>(make_zk, seed),
            make_zk,
            Some(nark_sponge),
            Some(&mut rng),
        )
        .unwrap()
    }

    /// One Pallas-AS accumulation `Input`: the recursion circuit's Pallas NARK
    /// proof packaged as `(r1cs_input, first_round_message,
    /// witness = second_round_message)` — the verifier proof to be accumulated.
    fn recursion_input(
        ipk: &IndexProverKey<PG>,
        make_zk: bool,
        seed: u64,
    ) -> Input<PCf, PSponge, PAS> {
        // The AS folds this input, so its NARK must use the forked nark_sponge.
        let proof = nark_prove(ipk, make_zk, seed, true);
        Input::<PCf, PSponge, PAS> {
            instance: InputInstance {
                r1cs_input: public_input(make_zk, seed),
                first_round_message: proof.first_msg,
            },
            witness: proof.second_msg,
        }
    }

    /// The two operands of a full IVC step, built once and reused by the
    /// fixture-dump fold so the byte-match isolates the folding accumulation.
    pub(super) struct FoldOperands {
        pub pk: <PAS as AccumulationScheme<PCf, PSponge>>::ProverKey,
        pub vk: <PAS as AccumulationScheme<PCf, PSponge>>::VerifierKey,
        pub input_cur: Input<PCf, PSponge, PAS>,
        pub acc_prev: Accumulator<PCf, PSponge, PAS>,
    }

    /// Build a full IVC step: the current step's verifier-proof input and a
    /// prior Pallas accumulator (an init accumulation of a different-seed
    /// recursion NARK over the same index — the accumulator this step folds).
    pub(super) fn fold_operands(make_zk: bool, seed: u64) -> FoldOperands {
        let (ipk, ivk) = nark_index(make_zk, seed);

        let mut rng = StdRng::seed_from_u64(seed ^ 0x1cc0);
        let as_pp = PAS::setup(&mut rng).unwrap();
        let (pk, vk, _dk) = PAS::index(&as_pp, &(), &(ipk.clone(), ivk.clone())).unwrap();

        // Current step: the verifier proof accumulated this step.
        let input_cur = recursion_input(&ipk, make_zk, seed);

        // Prior step: init-accumulate a different-seed recursion NARK to get the
        // accumulator the current step folds into.
        let input_prev = recursion_input(&ipk, make_zk, seed.wrapping_add(1));
        let prev_inputs = vec![input_prev];
        let no_acc: Vec<Accumulator<PCf, PSponge, PAS>> = Vec::new();
        let mut acc_rng = StdRng::seed_from_u64(seed ^ 0xacc0);
        let make_zk_prev = if make_zk {
            MakeZK::Enabled(&mut acc_rng)
        } else {
            MakeZK::Disabled
        };
        let (acc_prev, _proof_prev) = PAS::prove(
            &pk,
            Input::<PCf, PSponge, PAS>::map_to_refs(&prev_inputs),
            Accumulator::<PCf, PSponge, PAS>::map_to_refs(&no_acc),
            make_zk_prev,
            None,
        )
        .unwrap();

        FoldOperands {
            pk,
            vk,
            input_cur,
            acc_prev,
        }
    }

    /// Off-tree fixture dumps for the reverse (Pallas) direction — the
    /// curve-swapped twin of `vesta::dump`. Only the zk **fold** dump is mirrored
    /// here (the standalone half-step byte-match is Vesta-only); the helpers are
    /// the same JSON encoders, typed for `PG`/`CF`.
    mod dump {
        use super::{build_step, fold_operands, Vesta, PAS, PCf, PG, PSponge};
        use ark_accumulation::{AccumulationScheme, Accumulator, Input, MakeZK};
        use ark_ff::{BigInteger, PrimeField, UniformRand, Zero};
        use ark_poly_commit::trivial_pc::PedersenCommitment;
        use ark_relations::r1cs::{
            ConstraintSynthesizer, ConstraintSystem, Matrix, OptimizationGoal, SynthesisMode,
        };
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
        use ark_std::rand::{rngs::StdRng, SeedableRng};
        use blake2::VarBlake2b;
        use digest::{Update, VariableOutput};

        // The Pallas NARK's scalar field = the verifier circuit's field
        // (ark_vesta::Fq = ark_pallas::Fr); the matrices / assignment are over it.
        type CF = super::CF;

        fn hex(bytes: &[u8]) -> String {
            let mut s = String::with_capacity(bytes.len() * 2);
            for b in bytes {
                s.push_str(&format!("{:02x}", b));
            }
            s
        }
        fn fe_hex(f: &CF) -> String {
            hex(&f.into_repr().to_bytes_le())
        }
        fn ser_hex<T: CanonicalSerialize>(v: &T) -> String {
            let mut b = Vec::new();
            v.serialize(&mut b).unwrap();
            hex(&b)
        }
        fn matrix_json(m: &Matrix<CF>) -> String {
            let rows: Vec<String> = m
                .iter()
                .map(|row| {
                    let e: Vec<String> = row
                        .iter()
                        .map(|(c, i)| format!("[\"{}\",{}]", fe_hex(c), i))
                        .collect();
                    format!("[{}]", e.join(","))
                })
                .collect();
            format!("[{}]", rows.join(","))
        }
        fn fr_list_json(xs: &[CF]) -> String {
            let v: Vec<String> = xs.iter().map(|f| format!("\"{}\"", fe_hex(f))).collect();
            format!("[{}]", v.join(","))
        }
        fn xy(p: &PG) -> (String, String) {
            if p.is_zero() {
                (hex(&[0u8; 32]), hex(&[0u8; 32]))
            } else {
                (hex(&p.x.into_repr().to_bytes_le()), hex(&p.y.into_repr().to_bytes_le()))
            }
        }
        fn point_json(p: &PG) -> String {
            let (x, y) = xy(p);
            format!("{{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}}", x, y)
        }

        const NARK_PROTOCOL_NAME: &[u8] = b"R1CS-NARK-2020";
        const AS_PROTOCOL_NAME: &[u8] = b"AS-FOR-R1CS-NARK-2020";

        fn hash_matrices(domain: &[u8], a: &Matrix<CF>, b: &Matrix<CF>, c: &Matrix<CF>) -> [u8; 32] {
            let mut serialized = domain.to_vec();
            a.serialize(&mut serialized).unwrap();
            b.serialize(&mut serialized).unwrap();
            c.serialize(&mut serialized).unwrap();
            let mut hasher = VarBlake2b::new(32).unwrap();
            hasher.update(&serialized);
            let mut out = [0u8; 32];
            hasher.finalize_variable(|res| out.copy_from_slice(res));
            out
        }

        /// Parse a serialized `AccumulatorInstance` into structured JSON (see the
        /// `vesta::dump` twin for the field-order rationale).
        fn acc_instance_json(bytes: &[u8]) -> String {
            let mut cur = bytes;
            let r1cs_input = Vec::<CF>::deserialize(&mut cur).unwrap();
            let comm_a = PG::deserialize(&mut cur).unwrap();
            let comm_b = PG::deserialize(&mut cur).unwrap();
            let comm_c = PG::deserialize(&mut cur).unwrap();
            let hp_comm_1 = PG::deserialize(&mut cur).unwrap();
            let hp_comm_2 = PG::deserialize(&mut cur).unwrap();
            let hp_comm_3 = PG::deserialize(&mut cur).unwrap();
            format!(
                "{{\"r1cs_input\":{},\"comm_a\":{},\"comm_b\":{},\"comm_c\":{},\
                 \"hp_comm_1\":{},\"hp_comm_2\":{},\"hp_comm_3\":{}}}",
                fr_list_json(&r1cs_input),
                point_json(&comm_a), point_json(&comm_b), point_json(&comm_c),
                point_json(&hp_comm_1), point_json(&hp_comm_2), point_json(&hp_comm_3),
            )
        }

        /// Parse a serialized `AccumulatorWitness` into structured JSON.
        fn acc_witness_json(bytes: &[u8]) -> String {
            let mut cur = bytes;
            let r1cs_blinded_witness = Vec::<CF>::deserialize(&mut cur).unwrap();
            let hp_a_vec = Vec::<CF>::deserialize(&mut cur).unwrap();
            let hp_b_vec = Vec::<CF>::deserialize(&mut cur).unwrap();
            let read_opt3 = |cur: &mut &[u8]| -> (CF, CF, CF) {
                let flag = u8::deserialize(&mut *cur).unwrap();
                assert_eq!(flag, 1, "zk fold forces Some randomness");
                (
                    CF::deserialize(&mut *cur).unwrap(),
                    CF::deserialize(&mut *cur).unwrap(),
                    CF::deserialize(&mut *cur).unwrap(),
                )
            };
            let (hp_rand_1, hp_rand_2, hp_rand_3) = read_opt3(&mut cur);
            let (sigma_a, sigma_b, sigma_c) = read_opt3(&mut cur);
            format!(
                "{{\"r1cs_blinded_witness\":{},\"hp_a_vec\":{},\"hp_b_vec\":{},\
                 \"hp_rand_1\":\"{}\",\"hp_rand_2\":\"{}\",\"hp_rand_3\":\"{}\",\
                 \"sigma_a\":\"{}\",\"sigma_b\":\"{}\",\"sigma_c\":\"{}\"}}",
                fr_list_json(&r1cs_blinded_witness),
                fr_list_json(&hp_a_vec), fr_list_json(&hp_b_vec),
                fe_hex(&hp_rand_1), fe_hex(&hp_rand_2), fe_hex(&hp_rand_3),
                fe_hex(&sigma_a), fe_hex(&sigma_b), fe_hex(&sigma_c),
            )
        }

        /// Dump the **reverse** recursion-circuit zk fold (one
        /// input folded into a prior Pallas accumulator, num_addends=3) for the jax
        /// `prove_zk_fold(PALLAS, …)` byte-match — the Pallas twin of
        /// `vesta::dump::dump_recursion_fold_zk`.
        ///
        /// `cargo test --features recursion --test recursion_step \
        ///   pallas::dump::dump_recursion_fold_zk` (writes
        /// `$ACCUMULATION_ZORCH_ARTIFACTS/recursion_fold_zk_pallas_fixtures.json`).
        #[test]
        fn dump_recursion_fold_zk() {
            let out_dir = std::env::var("ACCUMULATION_ZORCH_ARTIFACTS")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| std::path::PathBuf::from("artifacts"));
            std::fs::create_dir_all(&out_dir).unwrap();
            let (make_zk, seed) = (true, 0u64);

            let mcs = ConstraintSystem::<CF>::new_ref();
            mcs.set_optimization_goal(OptimizationGoal::Constraints);
            mcs.set_mode(SynthesisMode::Setup);
            build_step::<Vesta>(make_zk, seed).generate_constraints(mcs.clone()).unwrap();
            mcs.finalize();
            let matrices = mcs.to_matrices().unwrap();
            let nark_matrices_hash =
                hash_matrices(NARK_PROTOCOL_NAME, &matrices.a, &matrices.b, &matrices.c);
            let as_matrices_hash =
                hash_matrices(AS_PROTOCOL_NAME, &matrices.a, &matrices.b, &matrices.c);

            let pcs = ConstraintSystem::<CF>::new_ref();
            pcs.set_optimization_goal(OptimizationGoal::Constraints);
            pcs.set_mode(SynthesisMode::Prove { construct_matrices: false });
            build_step::<Vesta>(make_zk, seed).generate_constraints(pcs.clone()).unwrap();
            pcs.finalize();
            let (input2, witness2, num_constraints) = {
                let cs = pcs.borrow().unwrap();
                (cs.instance_assignment.clone(), cs.witness_assignment.clone(), cs.num_constraints)
            };
            let num_witness = witness2.len();

            let cpp = PedersenCommitment::<PG>::setup(num_constraints);
            let ck = PedersenCommitment::<PG>::trim(&cpp, num_constraints);
            let supported_num_elems = ck.supported_num_elems();
            let (generators, hiding): (Vec<PG>, PG) = {
                let mut b = Vec::new();
                ck.serialize_uncompressed(&mut b).unwrap();
                let mut r = &b[..];
                let g = Vec::<PG>::deserialize_uncompressed(&mut r).unwrap();
                let h = PG::deserialize_uncompressed(&mut r).unwrap();
                (g, h)
            };
            let gens_json: Vec<String> = generators.iter().map(point_json).collect();
            let (hx, hy) = xy(&hiding);

            let ops = fold_operands(make_zk, seed);
            let acc_prev_instance_json = {
                let mut b = Vec::new();
                ops.acc_prev.instance.serialize(&mut b).unwrap();
                acc_instance_json(&b)
            };
            let acc_prev_witness_json = {
                let mut b = Vec::new();
                ops.acc_prev.witness.serialize(&mut b).unwrap();
                acc_witness_json(&b)
            };
            let cur_inputs = vec![ops.input_cur.clone()];
            let prev_accs = vec![ops.acc_prev.clone()];
            let mut rng_fold = StdRng::seed_from_u64(seed ^ 0xf01d);
            let (golden_acc, golden_proof) = PAS::prove(
                &ops.pk,
                Input::<PCf, PSponge, PAS>::map_to_refs(&cur_inputs),
                Accumulator::<PCf, PSponge, PAS>::map_to_refs(&prev_accs),
                MakeZK::Enabled(&mut rng_fold),
                None,
            )
            .unwrap();
            let verified = PAS::verify(
                &ops.vk,
                vec![&ops.input_cur.instance],
                vec![&ops.acc_prev.instance],
                &golden_acc.instance,
                &golden_proof,
                None,
            )
            .unwrap();
            assert!(verified, "golden reverse recursion fold failed to verify");

            let golden_instance_hex = ser_hex(&golden_acc.instance);
            let golden_witness_hex = ser_hex(&golden_acc.witness);
            let golden_proof_hex = ser_hex(&golden_proof);

            let mut rep2 = StdRng::seed_from_u64(seed ^ 0x5ec2);
            let r: Vec<CF> = (0..num_witness).map(|_| CF::rand(&mut rep2)).collect();
            let nark_blinders: Vec<CF> = (0..8).map(|_| CF::rand(&mut rep2)).collect();

            let mut rep_fold = StdRng::seed_from_u64(seed ^ 0xf01d);
            let as_r1cs_r_input = CF::rand(&mut rep_fold);
            let as_r1cs_r_witness = CF::rand(&mut rep_fold);
            let as_rand: Vec<CF> = (0..3).map(|_| CF::rand(&mut rep_fold)).collect();
            let hp_hiding_a = CF::rand(&mut rep_fold);
            let hp_hiding_b = CF::rand(&mut rep_fold);
            let hp_rand: Vec<CF> = (0..3).map(|_| CF::rand(&mut rep_fold)).collect();

            let json = format!(
                concat!(
                    "{{\n  \"note\": \"recursion-circuit zk Pallas fold (num_addends=3)\",\n",
                    "  \"curve\": \"pallas\",\n  \"num_constraints\": {},\n  \"num_vars\": {},\n",
                    "  \"supported_num_elems\": {},\n",
                    "  \"nark_matrices_hash_hex\": \"{}\",\n  \"as_matrices_hash_hex\": \"{}\",\n",
                    "  \"a\": {},\n  \"b\": {},\n  \"c\": {},\n",
                    "  \"generators\": [{}],\n  \"hiding\": {{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}},\n",
                    "  \"input2_r1cs_input\": {},\n  \"input2_witness\": {},\n",
                    "  \"r\": {},\n",
                    "  \"a_blinder\": \"{}\",\n  \"b_blinder\": \"{}\",\n  \"c_blinder\": \"{}\",\n",
                    "  \"r_a_blinder\": \"{}\",\n  \"r_b_blinder\": \"{}\",\n  \"r_c_blinder\": \"{}\",\n",
                    "  \"blinder_1\": \"{}\",\n  \"blinder_2\": \"{}\",\n",
                    "  \"as_r1cs_r_input\": \"{}\",\n  \"as_r1cs_r_witness\": \"{}\",\n",
                    "  \"as_rand_1\": \"{}\",\n  \"as_rand_2\": \"{}\",\n  \"as_rand_3\": \"{}\",\n",
                    "  \"hp_hiding_a\": \"{}\",\n  \"hp_hiding_b\": \"{}\",\n",
                    "  \"hp_rand_1\": \"{}\",\n  \"hp_rand_2\": \"{}\",\n  \"hp_rand_3\": \"{}\",\n",
                    "  \"acc_prev_instance\": {},\n  \"acc_prev_witness\": {},\n",
                    "  \"golden_instance_hex\": \"{}\",\n  \"golden_witness_hex\": \"{}\",\n",
                    "  \"golden_proof_hex\": \"{}\"\n}}\n"
                ),
                num_constraints,
                input2.len() + witness2.len(),
                supported_num_elems,
                hex(&nark_matrices_hash),
                hex(&as_matrices_hash),
                matrix_json(&matrices.a),
                matrix_json(&matrices.b),
                matrix_json(&matrices.c),
                gens_json.join(","),
                hx,
                hy,
                fr_list_json(&input2),
                fr_list_json(&witness2),
                fr_list_json(&r),
                fe_hex(&nark_blinders[0]), fe_hex(&nark_blinders[1]), fe_hex(&nark_blinders[2]),
                fe_hex(&nark_blinders[3]), fe_hex(&nark_blinders[4]), fe_hex(&nark_blinders[5]),
                fe_hex(&nark_blinders[6]), fe_hex(&nark_blinders[7]),
                fe_hex(&as_r1cs_r_input), fe_hex(&as_r1cs_r_witness),
                fe_hex(&as_rand[0]), fe_hex(&as_rand[1]), fe_hex(&as_rand[2]),
                fe_hex(&hp_hiding_a), fe_hex(&hp_hiding_b),
                fe_hex(&hp_rand[0]), fe_hex(&hp_rand[1]), fe_hex(&hp_rand[2]),
                acc_prev_instance_json, acc_prev_witness_json,
                golden_instance_hex, golden_witness_hex, golden_proof_hex,
            );
            let path = out_dir.join("recursion_fold_zk_pallas_fixtures.json");
            std::fs::write(&path, json).unwrap();
            eprintln!(
                "[dump] wrote {} ({} constraints, {} vars, fold acc.instance {}B / witness {}B / proof {}B)",
                path.display(),
                num_constraints,
                input2.len() + witness2.len(),
                golden_instance_hex.len() / 2,
                golden_witness_hex.len() / 2,
                golden_proof_hex.len() / 2,
            );
        }
    }
}
