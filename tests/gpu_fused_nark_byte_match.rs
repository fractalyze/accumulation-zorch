//! End-to-end GPU byte-match for the **fused no-zk NARK** prove of the recursion
//! circuit (the recursion half-step): the whole no-zk Vesta NARK
//! prove run as **one** PJRT call (`fused::prove_nark_no_zk_fused`) must serialize
//! byte-for-byte to the golden `Proof` arkworks produces — the same proof the
//! per-MSM `GpuBackend` byte-match (`recursion_step::vesta::on_gpu::
//! recursion_step_proves_on_vesta`) hits, but with a single fused dispatch whose
//! `M·z` is reduced on-device from the sparse COO (`field.sparse_matvec` →
//! `stablehlo.scatter`) instead of three per-MSM commit dispatches.
//!
//! This is the GPU half of the half-step no-zk gate: the CPU side
//! (`recursion_nark_test.py::…_fused_…`) already byte-matches the same golden, so
//! a match here proves the on-device sparse scatter is correct on the GPU plugin,
//! not just under the CPU lowering.
//!
//! Fixture-driven and **off-tree** (the recursion fixture is ~17 MB and the
//! `.mlirbc` ~6 MB): both live under `$ACCUMULATION_ZORCH_ARTIFACTS`. The test
//! SKIPS (prints + returns) when either is absent — the same on-demand contract
//! as the Python recursion gate. Generate them first:
//!
//!     ACCUMULATION_ZORCH_ARTIFACTS=<dir> \
//!       cargo test --features recursion --test recursion_step dump_recursion_nark
//!     ACCUMULATION_ZORCH_ARTIFACTS=<dir> bazel run //export:export_nark
//!
//! Hardware-gated; run only when the GPU is idle:
//!
//!     XLA_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so \
//!       ACCUMULATION_ZORCH_ARTIFACTS=<dir> \
//!       cargo test --features gpu --test gpu_fused_nark_byte_match -- --ignored --test-threads=1
#![cfg(feature = "gpu")]

mod common;

use accumulation_zorch::fused;
use accumulation_zorch::gpu::Vesta;
use ark_vesta::{Affine, Fr};
use common::{fr_vec, point_from_json, to_hex};

#[test]
#[ignore = "needs XLA_PJRT_PLUGIN + off-tree recursion fixture + nark_no_zk_vesta.mlirbc + a GPU"]
fn gpu_fused_nark_byte_match() {
    let artifacts = fixture_json::artifacts_dir(env!("CARGO_MANIFEST_DIR"));
    let fixture_path = artifacts.join("recursion_nark_fixtures.json");
    let mlirbc_path = artifacts.join("nark_no_zk_vesta.mlirbc");
    if !fixture_path.exists() || !mlirbc_path.exists() {
        println!(
            "SKIP — missing off-tree artifact(s) under {}:\n  {}\n  {}\n  (generate: dump_recursion_nark + export/export_nark.py)",
            artifacts.display(),
            fixture_path.display(),
            mlirbc_path.display(),
        );
        return;
    }

    // --- Phase A: parse the fixture + build the runtime inputs, no PJRT client.
    let d: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture_path).expect("read fixture"))
            .expect("parse fixture json");
    // The no-zk NARK commits `M·z` over the full generator set (no hiding term),
    // so the committer key is every generator (one per constraint row).
    let bases: Vec<Affine> =
        d["generators"].as_array().unwrap().iter().map(point_from_json::<Vesta>).collect();
    // blinded_witness = the raw witness on the no-zk path (a baked host constant).
    let witness: Vec<Fr> = fr_vec::<Vesta>(&d["witness"]);
    let golden = d["proof_hex"].as_str().unwrap().to_string();
    let mlirbc = std::fs::read(&mlirbc_path).expect("read nark_no_zk_vesta.mlirbc");

    // --- Phase B: one fused PJRT call on the GPU, then byte-match the proof.
    println!("fused no-zk Vesta NARK GPU byte-match (recursion half-step):");
    let got = fused::prove_nark_no_zk_fused::<Vesta>(&mlirbc, &bases, &witness);
    assert_eq!(
        to_hex(&got),
        golden,
        "fused no-zk Vesta NARK proof diverged from arkworks (got {}B, want {}B)",
        got.len(),
        golden.len() / 2,
    );
    println!(
        "  recursion no-zk NARK proof ({} constraints, {} generators, {}B) byte-matches arkworks via ONE fused PJRT call",
        d["num_constraints"].as_u64().unwrap(),
        bases.len(),
        got.len(),
    );
    println!("FUSED NO-ZK NARK GPU BYTE-MATCH PASSED");
}
