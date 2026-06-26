//! Byte-identical oracle.
//!
//! The crux de-risk for the faithful copy + the [`MsmBackend`](crate::backend)
//! seam: `accumulation_zorch`'s `prove` (specialized to `CpuBackend`) must
//! produce output bit-for-bit identical to the *unmodified* arkworks
//! `ark_accumulation` prover over Pallas, given the same seeded randomness.
//!
//! Both crates are compiled against the same `ark-*` dependencies (the
//! `accumulation-experimental` branch), so any byte difference can only come
//! from the copy itself or the seam — not a dependency-version mismatch. The
//! comparison is over serialized bytes, so the two crates' nominally-distinct
//! types never need to interoperate; only a `DummyCircuit` (pure
//! `ark_relations`) and the shared scalar field are common.

use ark_ff::UniformRand;
use ark_relations::lc;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, OptimizationGoal, SynthesisError,
    SynthesisMode,
};
use ark_serialize::CanonicalSerialize;
use ark_sponge::CryptographicSponge;
use ark_std::rand::{rngs::StdRng, SeedableRng};

type G = ark_pallas::Affine;
type CF = ark_pallas::Fq;
type Fr = ark_pallas::Fr;
type Sponge = ark_sponge::poseidon::PoseidonSponge<CF>;

/// `a · b = c`, padded with `num_inputs - 1` extra public inputs and
/// `num_constraints - 1` repeats of the multiplication constraint. A copy of the
/// circuit the arkworks `r1cs_nark_as` tests use, defined here so both crates'
/// `R1CSNark::prove` (which takes `impl ConstraintSynthesizer<Fr>`) can be driven
/// by one shared circuit type.
#[derive(Clone)]
struct DummyCircuit {
    a: Option<Fr>,
    b: Option<Fr>,
    num_inputs: usize,
    num_constraints: usize,
}

impl ConstraintSynthesizer<Fr> for DummyCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
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

/// Runs one full accumulation step (NARK setup/index/prove → build input →
/// AS setup/index/prove) under the crate rooted at `$root`, seeded by `$seed`,
/// and returns the serialized `(accumulator instance, accumulator witness,
/// proof)`. The body is identical across crates, so two expansions consume the
/// shared `rng` in lockstep; only the (nominally distinct) types differ.
macro_rules! prove_bytes {
    ($root:tt, $make_zk:expr, $seed:expr) => {{
        use $root::r1cs_nark_as::r1cs_nark::R1CSNark;
        use $root::r1cs_nark_as::{ASForR1CSNark, InputInstance};
        use $root::{AccumulationScheme, Accumulator, Input, MakeZK};

        type AS = ASForR1CSNark<G, Sponge>;

        let num_inputs = 5usize;
        let num_constraints = 10usize;
        let make_zk: bool = $make_zk;
        let mut rng = StdRng::seed_from_u64($seed);

        // NARK setup + index over a freshly sampled circuit.
        let nark_pp = R1CSNark::<G, Sponge>::setup();
        let index_circuit = DummyCircuit {
            a: Some(Fr::rand(&mut rng)),
            b: Some(Fr::rand(&mut rng)),
            num_inputs,
            num_constraints,
        };
        let (ipk, ivk) = R1CSNark::<G, Sponge>::index(&nark_pp, index_circuit).unwrap();

        // Accumulation-scheme setup + index.
        let as_pp = AS::setup(&mut rng).unwrap();
        let (pk, _vk, _dk) = AS::index(&as_pp, &(), &(ipk.clone(), ivk.clone())).unwrap();

        // Build one accumulation input by running the NARK prover.
        let circuit = DummyCircuit {
            a: Some(Fr::rand(&mut rng)),
            b: Some(Fr::rand(&mut rng)),
            num_inputs,
            num_constraints,
        };
        let nark_sponge = ASForR1CSNark::<G, Sponge>::nark_sponge(&Sponge::new());
        let nark_proof = R1CSNark::<G, Sponge>::prove(
            &ipk,
            circuit.clone(),
            make_zk,
            Some(nark_sponge),
            Some(&mut rng),
        )
        .unwrap();

        let pcs = ConstraintSystem::new_ref();
        pcs.set_optimization_goal(OptimizationGoal::Weight);
        pcs.set_mode(SynthesisMode::Prove {
            construct_matrices: false,
        });
        circuit.generate_constraints(pcs.clone()).unwrap();
        pcs.finalize();
        let r1cs_input = pcs.borrow().unwrap().instance_assignment.clone();

        let input = Input::<CF, Sponge, AS> {
            instance: InputInstance {
                r1cs_input,
                first_round_message: nark_proof.first_msg.clone(),
            },
            witness: nark_proof.second_msg,
        };
        let inputs = vec![input];
        let no_accumulators: Vec<Accumulator<CF, Sponge, AS>> = Vec::new();

        let make_zk_arg = if make_zk {
            MakeZK::Enabled(&mut rng)
        } else {
            MakeZK::Disabled
        };
        let (accumulator, proof) = AS::prove(
            &pk,
            Input::<CF, Sponge, AS>::map_to_refs(&inputs),
            Accumulator::<CF, Sponge, AS>::map_to_refs(&no_accumulators),
            make_zk_arg,
            None,
        )
        .unwrap();

        let mut bytes = Vec::new();
        accumulator.instance.serialize(&mut bytes).unwrap();
        accumulator.witness.serialize(&mut bytes).unwrap();
        proof.serialize(&mut bytes).unwrap();
        bytes
    }};
}

fn assert_byte_identical(make_zk: bool, seed: u64) {
    let ours = prove_bytes!(crate, make_zk, seed);
    let theirs = prove_bytes!(ark_accumulation, make_zk, seed);
    assert!(!ours.is_empty(), "prove produced no bytes");
    assert_eq!(
        ours, theirs,
        "accumulation-zorch CpuBackend prove diverged from arkworks (make_zk={make_zk}, seed={seed})"
    );
}

#[test]
fn prove_byte_identical_to_arkworks_no_zk() {
    assert_byte_identical(false, 0);
    assert_byte_identical(false, 42);
}

#[test]
fn prove_byte_identical_to_arkworks_zk() {
    assert_byte_identical(true, 0);
    assert_byte_identical(true, 42);
}
