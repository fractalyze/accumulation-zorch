//! End-to-end GPU byte-match for the **zk (hiding) IPA-PC accumulation fold** open
//! fused core (Slice 3) — one no-zk input folded INTO a prior *hiding* accumulator,
//! its rlp-seeded combined check polynomial opened with a hiding blind as **one**
//! PJRT call (`crate::fused::open_ipa_fold_zk_fused`) must reproduce the golden
//! folded hiding accumulator's IPA proof (`l_vec`/`r_vec`/`final_comm_key`/`c` +
//! `hiding_comm`/`rand`) byte-for-byte, over **both** Pasta cycle curves.
//!
//! The zk twin of `gpu_fused_ipa_fold_byte_match`: the core additionally runs
//! zorch's `_open_one_zk` hiding prelude on-device (the blinding Pedersen
//! commitment, the on-device `hiding_challenge`, the blinded `lax.scan` fold) and
//! recovers the blinding-folded `final_comm_key` in-trace. `export/export_ipa_fold.py
//! … zk` bakes one fixture's hiding statement into `ipa_fold_zk_<curve>.mlirbc`; the
//! committer-key `generators` is the sole runtime input. This is the GPU twin of the
//! `ipa_as_fold_zk_test.py` CPU byte-match.
//!
//! Hardware-gated; run only on an idle GPU:
//!
//!     XLA_PJRT_PLUGIN=.../xla_cuda_plugin.so \
//!       cargo test --features gpu --test gpu_fused_ipa_fold_zk_byte_match -- --ignored --test-threads=1 --nocapture
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
type Fq<C> = <<C as PastaCurve>::Params as ModelParameters>::BaseField;

/// Run the zk fold open core for one curve and assert the folded hiding
/// accumulator's IPA proof (all six fields) reproduces the golden.
fn check_curve<C: PastaCurve>(fixture: &str, artifact: &str)
where
    Fq<C>: PrimeField,
{
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let d: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("python/testdata").join(fixture)).expect("read fixture"),
    )
    .expect("parse fixture json");
    let artifacts = fixture_json::artifacts_dir(env!("CARGO_MANIFEST_DIR"));

    let generators: Vec<Affine<C>> =
        d["generators"].as_array().unwrap().iter().map(point_from_json::<C>).collect();
    let acc = &d["accumulator"];
    let want_l: Vec<Affine<C>> =
        acc["l_vec"].as_array().unwrap().iter().map(point_from_json::<C>).collect();
    let want_r: Vec<Affine<C>> =
        acc["r_vec"].as_array().unwrap().iter().map(point_from_json::<C>).collect();
    let want_final = point_from_json::<C>(&acc["final_comm_key"]);
    let want_c = fr_from_hex::<C>(acc["c"].as_str().unwrap());
    let want_hiding = point_from_json::<C>(&acc["hiding_comm"]);
    let want_rand = fr_from_hex::<C>(acc["rand"].as_str().unwrap());

    let mlirbc = std::fs::read(artifacts.join(artifact))
        .unwrap_or_else(|e| panic!("read {}: {}", artifact, e));
    let got = fused::open_ipa_fold_zk_fused::<C>(&mlirbc, &generators);

    assert_eq!(got.l_vec, want_l, "hiding accumulator ipa_proof.l_vec != arkworks");
    assert_eq!(got.r_vec, want_r, "hiding accumulator ipa_proof.r_vec != arkworks");
    assert_eq!(got.final_comm_key, want_final, "hiding accumulator ipa_proof.final_comm_key != arkworks");
    assert_eq!(got.c, want_c, "hiding accumulator ipa_proof.c != arkworks");
    assert_eq!(got.hiding_comm, want_hiding, "hiding accumulator ipa_proof.hiding_comm != arkworks");
    assert_eq!(got.rand, want_rand, "hiding accumulator ipa_proof.rand != arkworks");
    println!(
        "    zk IPA fold open ({} rounds: l_vec/r_vec/final_comm_key/c + hiding_comm/rand) byte-matches arkworks",
        got.l_vec.len()
    );
}

#[test]
#[ignore = "needs XLA_PJRT_PLUGIN + artifacts/ipa_fold_zk_{pallas,vesta}.mlirbc + a GPU"]
fn gpu_fused_ipa_fold_zk_byte_match() {
    println!("fused frx-exported zk IPA-PC accumulation FOLD open GPU byte-match (hiding, Pallas + Vesta):");
    println!("  [pallas]");
    check_curve::<Pallas>("ipa_as_fold_zk_fixtures.json", "ipa_fold_zk_pallas.mlirbc");
    println!("  [vesta]");
    check_curve::<Vesta>("ipa_as_fold_zk_vesta_fixtures.json", "ipa_fold_zk_vesta.mlirbc");
    println!("ALL FUSED ZK IPA-PC-AS FOLD OPEN GPU BYTE-MATCH CHECKS PASSED");
}
