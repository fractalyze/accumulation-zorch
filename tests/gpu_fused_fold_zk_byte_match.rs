//! End-to-end GPU byte-match for the **fused zk IVC fold** of the recursion
//! circuit (the full IVC step — forward / reverse,
//! `num_addends = 3`): the whole multi-addend fold run as **one** PJRT call
//! (`fused::prove_fold_zk_fused`) must serialize byte-for-byte to the golden
//! folded `acc.instance ‖ acc.witness ‖ proof` arkworks produces — the same golden
//! the CPU byte-match (`recursion_fold_zk_test.py`) already hits, but with a single
//! fused dispatch whose `M·z` is reduced on-device from the sparse COO
//! (`field.sparse_matvec` → `stablehlo.scatter`) instead of the per-MSM
//! `GpuBackend` fold dispatch.
//!
//! This is the GPU half of the IVC fold gate, and the bar acceptance-criterion
//! "the full IVC fold step runs as fused call(s) and byte-matches GpuBackend/CPU
//! (both cycle directions)" needs: a match here proves the on-device fold (the
//! `num_addends=3` AS-level combine, the HP-level old-accumulator fold, and the
//! multi-addend `beta` sponge) is correct on the GPU plugin, both directions, not
//! just under the CPU lowering. The fold core takes three runtime affine inputs —
//! the committer key `bases_h = generators[:rows] ‖ hiding`, the HP placeholder
//! identity, and the old accumulator's six commitments — vs the single-input
//! prove's two; the output pytree (and serialization) is identical, a fold being an
//! `AccumulatorInstance ‖ AccumulatorWitness ‖ Proof`.
//!
//! Fixture-driven and **off-tree** (each recursion fold fixture is ~165 MB and its
//! `.mlirbc` ~80 MB): both live under `$ACCUMULATION_ZORCH_ARTIFACTS`. Each
//! direction SKIPS (prints) when its artifacts are absent — the same on-demand
//! contract as the Python recursion gate; the test fails only if neither direction
//! ran. Generate them first:
//!
//!     ACCUMULATION_ZORCH_ARTIFACTS=<dir> cargo test --features recursion \
//!       --test recursion_step vesta::dump::dump_recursion_fold_zk   # forward
//!     ACCUMULATION_ZORCH_ARTIFACTS=<dir> cargo test --features recursion \
//!       --test recursion_step pallas::dump::dump_recursion_fold_zk  # reverse
//!     ACCUMULATION_ZORCH_ARTIFACTS=<dir> PROVE_CURVE=vesta  bazel run //export:export_fold_zk
//!     ACCUMULATION_ZORCH_ARTIFACTS=<dir> PROVE_CURVE=pallas bazel run //export:export_fold_zk
//!
//! Hardware-gated; run only when the GPU is idle:
//!
//!     XLA_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so ACCUMULATION_ZORCH_ARTIFACTS=<dir> \
//!       cargo test --features gpu --test gpu_fused_fold_zk_byte_match -- --ignored --test-threads=1
#![cfg(feature = "gpu")]

use accumulation_zorch::fused;
use accumulation_zorch::gpu::{Pallas, PastaCurve, Vesta};
use ark_ec::models::ModelParameters;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ff::{PrimeField, Zero};
use std::path::{Path, PathBuf};

type Affine<C> = GroupAffine<<C as PastaCurve>::Params>;
type Fr<C> = <<C as PastaCurve>::Params as ModelParameters>::ScalarField;
type Base<C> = <<C as PastaCurve>::Params as ModelParameters>::BaseField;

/// Decode an even-length lowercase hex string to bytes.
fn from_hex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd-length hex");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

/// Encode bytes to a lowercase hex string (matches the fixture's `*_hex`).
fn to_hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{:02x}", x));
    }
    s
}

/// A canonical-LE scalar from its fixture hex.
fn fr_from_hex<C: PastaCurve>(s: &str) -> Fr<C> {
    Fr::<C>::from_le_bytes_mod_order(&from_hex(s))
}

/// An affine point from a fixture `{x_le_hex, y_le_hex}` object (canonical-LE
/// base-field coordinates; finite — the committer-key / accumulator points are
/// never the identity).
fn point_from_json<C: PastaCurve>(v: &serde_json::Value) -> Affine<C>
where
    Base<C>: PrimeField,
{
    let x = Base::<C>::from_le_bytes_mod_order(&from_hex(v["x_le_hex"].as_str().unwrap()));
    let y = Base::<C>::from_le_bytes_mod_order(&from_hex(v["y_le_hex"].as_str().unwrap()));
    GroupAffine::new(x, y, false)
}

/// Run + byte-match one fold direction. Returns `false` (and prints SKIP) when its
/// off-tree fixture or `.mlirbc` is absent — the on-demand contract the recursion
/// gates use. Each direction is the same curve-generic `prove_fold_zk_fused`, on
/// the direction's own curve `C` and recursion fold fixture.
fn check_direction<C: PastaCurve>(
    label: &str,
    artifacts: &Path,
    fixture_name: &str,
    mlirbc_name: &str,
) -> bool
where
    Base<C>: PrimeField,
{
    let fixture_path = artifacts.join(fixture_name);
    let mlirbc_path = artifacts.join(mlirbc_name);
    if !fixture_path.exists() || !mlirbc_path.exists() {
        println!(
            "SKIP [{label}] — missing off-tree artifact(s):\n    {}\n    {}",
            fixture_path.display(),
            mlirbc_path.display(),
        );
        return false;
    }

    // --- Phase A: parse the fixture + build the runtime inputs, no PJRT client.
    let d: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture_path).expect("read fixture"))
            .expect("parse fixture json");

    // The fold core's three runtime affine inputs: `bases_h = generators[:rows] ‖
    // hiding` (rows = num_constraints), the HP placeholder identity, and the old
    // accumulator's six commitments `[comm_a, comm_b, comm_c, hp_1, hp_2, hp_3]`.
    let rows = d["num_constraints"].as_u64().unwrap() as usize;
    let generators: Vec<Affine<C>> =
        d["generators"].as_array().unwrap().iter().map(point_from_json::<C>).collect();
    let mut bases_h = generators[..rows].to_vec();
    bases_h.push(point_from_json::<C>(&d["hiding"]));
    let id_pt = Affine::<C>::zero();
    let acc = &d["acc_prev_instance"];
    let acc_comms: Vec<Affine<C>> =
        ["comm_a", "comm_b", "comm_c", "hp_comm_1", "hp_comm_2", "hp_comm_3"]
            .iter()
            .map(|k| point_from_json::<C>(&acc[*k]))
            .collect();

    // AS ProofRandomness.r1cs_input = vec![as_r1cs_r_input; input_len] — the one
    // serialized field that is a baked constant, not an output leaf; input_len is
    // the folded input's length.
    let input_len = d["input2_r1cs_input"].as_array().unwrap().len();
    let r1cs_r_input = vec![fr_from_hex::<C>(d["as_r1cs_r_input"].as_str().unwrap()); input_len];
    let mlirbc = std::fs::read(&mlirbc_path).expect("read fold .mlirbc");

    // --- Phase B: one fused PJRT call → the folded acc.instance ‖ witness ‖ proof.
    let got = fused::prove_fold_zk_fused::<C>(&mlirbc, &bases_h, &id_pt, &acc_comms, &r1cs_r_input);
    assert_eq!(
        to_hex(&got.acc_instance),
        d["golden_instance_hex"].as_str().unwrap(),
        "[{label}] folded acc.instance diverged ({}B)",
        got.acc_instance.len(),
    );
    assert_eq!(
        to_hex(&got.acc_witness),
        d["golden_witness_hex"].as_str().unwrap(),
        "[{label}] folded acc.witness diverged ({}B)",
        got.acc_witness.len(),
    );
    assert_eq!(
        to_hex(&got.proof),
        d["golden_proof_hex"].as_str().unwrap(),
        "[{label}] fold proof diverged ({}B)",
        got.proof.len(),
    );
    println!(
        "  [{label}] recursion zk fold ({rows} constraints, acc.instance {}B ‖ witness {}B ‖ proof {}B) byte-matches arkworks via ONE fused PJRT call",
        got.acc_instance.len(),
        got.acc_witness.len(),
        got.proof.len(),
    );
    true
}

#[test]
#[ignore = "needs XLA_PJRT_PLUGIN + off-tree recursion fold fixtures + fold_zk_<curve>.mlirbc + a GPU"]
fn gpu_fused_fold_zk_byte_match() {
    let artifacts = std::env::var("ACCUMULATION_ZORCH_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("artifacts"));

    println!("fused zk IVC fold GPU byte-match (full IVC step, num_addends=3):");
    // Forward folds on Vesta (constraint field ark_vesta::Fq), reverse on Pallas.
    let fwd = check_direction::<Vesta>(
        "vesta forward",
        &artifacts,
        "recursion_fold_zk_fixtures.json",
        "fold_zk_vesta.mlirbc",
    );
    let rev = check_direction::<Pallas>(
        "pallas reverse",
        &artifacts,
        "recursion_fold_zk_pallas_fixtures.json",
        "fold_zk_pallas.mlirbc",
    );
    assert!(
        fwd || rev,
        "no fold fixtures present — generate them (dump_recursion_fold_zk + export/export_fold_zk.py)",
    );
    println!("FUSED ZK FOLD GPU BYTE-MATCH PASSED");
}
