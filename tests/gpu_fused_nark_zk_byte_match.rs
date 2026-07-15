//! End-to-end GPU byte-match for the **fused zk NARK** prove of the recursion
//! circuit (the recursion half-step, make_zk path): the whole zk
//! Vesta NARK prove run as **one** PJRT call (`fused::prove_nark_zk_fused`) must
//! serialize byte-for-byte to the golden `Proof` arkworks produces — the same
//! proof the per-MSM `GpuBackend` byte-match (`recursion_step::vesta::on_gpu::
//! recursion_step_proves_on_vesta`, make_zk=true) hits, but with a single fused
//! dispatch whose `M·z` is reduced on-device from the sparse COO
//! (`field.sparse_matvec` → `stablehlo.scatter`) instead of eight per-MSM commit
//! dispatches.
//!
//! This is the GPU half of the half-step zk gate: the CPU side
//! (`recursion_nark_zk_test.py`) already byte-matches the same golden, so a match
//! here proves the on-device scatter + the in-trace gamma sponge (the unforked
//! `Sponge::new()` the standalone half-step passes) are correct on the GPU plugin,
//! not just under the CPU lowering.
//!
//! Fixture-driven and **off-tree** (the zk recursion fixture is ~147 MB and the
//! `.mlirbc` ~73 MB): both live under `$ACCUMULATION_ZORCH_ARTIFACTS`. The test
//! SKIPS (prints + returns) when either is absent — the same on-demand contract
//! as the Python recursion gate. Generate them first:
//!
//!     ACCUMULATION_ZORCH_ARTIFACTS=<dir> \
//!       cargo test --features recursion --test recursion_step dump_recursion_nark_zk
//!     ACCUMULATION_ZORCH_ARTIFACTS=<dir> bazel run //export:export_nark_zk
//!
//! Hardware-gated; run only when the GPU is idle:
//!
//!     XLA_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so \
//!       ACCUMULATION_ZORCH_ARTIFACTS=<dir> \
//!       cargo test --features gpu --test gpu_fused_nark_zk_byte_match -- --ignored --test-threads=1
#![cfg(feature = "gpu")]

mod common;

use accumulation_zorch::fused;
use accumulation_zorch::gpu::Vesta;
use ark_vesta::Affine;
use common::{point_from_json, to_hex};

#[test]
#[ignore = "needs XLA_PJRT_PLUGIN + off-tree zk recursion fixture + nark_zk_vesta.mlirbc + a GPU"]
fn gpu_fused_nark_zk_byte_match() {
    let artifacts = fixture_json::artifacts_dir(env!("CARGO_MANIFEST_DIR"));
    let fixture_path = artifacts.join("recursion_nark_zk_fixtures.json");
    let mlirbc_path = artifacts.join("nark_zk_vesta.mlirbc");
    if !fixture_path.exists() || !mlirbc_path.exists() {
        println!(
            "SKIP — missing off-tree artifact(s) under {}:\n  {}\n  {}\n  (generate: dump_recursion_nark_zk + export/export_nark_zk.py)",
            artifacts.display(),
            fixture_path.display(),
            mlirbc_path.display(),
        );
        return;
    }

    // --- Phase A: parse the fixture + build the runtime input, no PJRT client.
    let d: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture_path).expect("read fixture"))
            .expect("parse fixture json");
    // The zk NARK commits with a blinder on the hiding base, so the committer key
    // is every generator (one per constraint row) followed by the hiding term —
    // the `bases_h` stack `nark.build_zk_core` lowered the core for.
    let mut bases_h: Vec<Affine> =
        d["generators"].as_array().unwrap().iter().map(point_from_json::<Vesta>).collect();
    bases_h.push(point_from_json::<Vesta>(&d["hiding"]));
    let golden = d["proof_hex"].as_str().unwrap().to_string();
    let mlirbc = std::fs::read(&mlirbc_path).expect("read nark_zk_vesta.mlirbc");

    // --- Phase B: one fused PJRT call on the GPU, then byte-match the proof.
    println!("fused zk Vesta NARK GPU byte-match (recursion half-step):");
    let got = fused::prove_nark_zk_fused::<Vesta>(&mlirbc, &bases_h);
    assert_eq!(
        to_hex(&got),
        golden,
        "fused zk Vesta NARK proof diverged from arkworks (got {}B, want {}B)",
        got.len(),
        golden.len() / 2,
    );
    println!(
        "  recursion zk NARK proof ({} constraints, {} bases_h, {}B) byte-matches arkworks via ONE fused PJRT call",
        d["num_constraints"].as_u64().unwrap(),
        bases_h.len(),
        got.len(),
    );
    println!("FUSED ZK NARK GPU BYTE-MATCH PASSED");
}
