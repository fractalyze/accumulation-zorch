//! End-to-end GPU byte-match for the **IPA-PC accumulation decider** fused core
//! (Slice 4) — the decider's size-`d` MSM run as **one** PJRT call
//! (`crate::fused::decide_ipa_msm_fused`) must reproduce the accumulator's
//! arkworks `final_comm_key` byte-for-byte, over **both** Pasta cycle curves.
//!
//! Fixture-driven: `python/testdata/ipa_as{,_vesta}_fixtures.json` hold the IPA
//! committer-key `generators`, the accumulator's `final_comm_key` (the golden
//! decider output), and `decider_coeffs` — the dense
//! `compute_coeffs(succinct_check(accumulator))` the arkworks oracle emits
//! (`examples/dump_ipa_as.rs`), tied to the jax port by `testing/ipa_as_test.py`.
//! The decider accepts iff `MSM(generators, decider_coeffs) == final_comm_key`
//! (`ipa_pc_as.decide_final_key` / `IpaPC::check`); this exercises that MSM on the
//! GPU. `export/export_ipa.py` lowers one general
//! `ipa_decider_msm_<curve>.mlirbc` per curve whose runtime inputs are the
//! `coeffs` (scalars) and `generators` (bases), so one core decides any
//! accumulator at that degree.
//!
//! Hardware-gated; run only when the GPU is idle (the MSM is tiny — 8 terms — so
//! both curves run in one process well under the plugin's VRAM pool):
//!
//!     ZKX_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so \
//!       cargo test --features gpu --test gpu_fused_ipa_decide_byte_match -- --ignored --test-threads=1 --nocapture
#![cfg(feature = "gpu")]

mod common;

use accumulation_zorch::fused;
use accumulation_zorch::gpu::{Pallas, PastaCurve, Vesta};
use ark_ec::models::ModelParameters;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ff::PrimeField;
use common::{fr_from_hex, point_from_json};
use serde_json::Value;
use std::path::PathBuf;

type Affine<C> = GroupAffine<<C as PastaCurve>::Params>;
type Fr<C> = <<C as PastaCurve>::Params as ModelParameters>::ScalarField;
type Fq<C> = <<C as PastaCurve>::Params as ModelParameters>::BaseField;

/// Run the decider MSM core for one curve and assert it reproduces the
/// accumulator's `final_comm_key`.
fn check_curve<C: PastaCurve>(fixture: &str, artifact: &str)
where
    Fq<C>: PrimeField,
{
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let d: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("python/testdata").join(fixture)).expect("read fixture"),
    )
    .expect("parse fixture json");
    let artifacts = std::env::var("ACCUMULATION_ZORCH_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("artifacts"));

    // The committer-key generators (bases) and the dense check-poly coeffs
    // (scalars) — the decider MSM's two runtime inputs — plus the golden
    // `final_comm_key` it must reproduce.
    let generators: Vec<Affine<C>> =
        d["generators"].as_array().unwrap().iter().map(point_from_json::<C>).collect();
    let coeffs: Vec<Fr<C>> = d["decider_coeffs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| fr_from_hex::<C>(h.as_str().unwrap()))
        .collect();
    let want = point_from_json::<C>(&d["accumulator"]["final_comm_key"]);

    let mlirbc = std::fs::read(artifacts.join(artifact))
        .unwrap_or_else(|e| panic!("read {}: {}", artifact, e));
    let got = fused::decide_ipa_msm_fused::<C>(&mlirbc, &coeffs, &generators);

    assert_eq!(got, want, "decider size-d MSM != accumulator.final_comm_key");
    println!(
        "    decider size-d MSM (= MSM(generators, h(X) coeffs), {} terms) byte-matches arkworks final_comm_key",
        coeffs.len()
    );
}

#[test]
#[ignore = "needs ZKX_PJRT_PLUGIN + artifacts/ipa_decider_msm_{pallas,vesta}.mlirbc + a GPU"]
fn gpu_fused_ipa_decide_byte_match() {
    println!("fused jax-exported IPA-PC accumulation DECIDER MSM GPU byte-match (Pallas + Vesta):");
    println!("  [pallas]");
    check_curve::<Pallas>("ipa_as_fixtures.json", "ipa_decider_msm_pallas.mlirbc");
    println!("  [vesta]");
    check_curve::<Vesta>("ipa_as_vesta_fixtures.json", "ipa_decider_msm_vesta.mlirbc");
    println!("ALL FUSED IPA-PC-AS DECIDER MSM GPU BYTE-MATCH CHECKS PASSED");
}
