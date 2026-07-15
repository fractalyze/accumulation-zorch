//! End-to-end GPU byte-match for the **IPA-PC accumulation fold** open fused core
//! (Slice 3) — one input folded INTO a prior accumulator, its combined check
//! polynomial opened as **one** PJRT call (`crate::fused::open_ipa_fold_fused`)
//! must reproduce the golden folded accumulator's IPA proof (`l_vec` / `r_vec` /
//! `final_comm_key` / `c`) byte-for-byte, over **both** Pasta cycle curves.
//!
//! Unlike the decider (one host-fed size-`d` MSM), the fold's `IpaPC::open` is
//! sequential — each round's Fiat-Shamir challenge is squeezed on-device from that
//! round's `L_j`/`R_j` MSM outputs — so this core runs the whole open (the
//! `lax.scan` basis fold + the `final_comm_key` MSM, Poseidon interleaved via the
//! arkworks-faithful `ipa_challenger`) on the GPU. `export/export_ipa_fold.py`
//! bakes one fixture's combined check polynomial (the host combine — cheap
//! field/sponge, already byte-matched on CPU by `ipa_as_fold_test`) into
//! `ipa_fold_<curve>.mlirbc`; the committer-key `generators` is the sole runtime
//! input. This is the GPU twin of that CPU byte-match.
//!
//! Fixture-driven: `python/testdata/ipa_as_fold{,_vesta}_fixtures.json` — the same
//! two-round fold golden `cargo run --example dump_ipa_as_fold` dumps and
//! `ipa_as_fold_test.py` byte-matches on CPU; `accumulator` holds the golden IPA
//! proof this core reproduces.
//!
//! Hardware-gated; run only when the GPU is idle (the open is tiny — 8-coeff, 3
//! rounds — so both curves run in one process well under the plugin's VRAM pool):
//!
//!     XLA_PJRT_PLUGIN=.../xla_cuda_plugin.so \
//!       cargo test --features gpu --test gpu_fused_ipa_fold_byte_match -- --ignored --test-threads=1 --nocapture
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

/// Run the fold open core for one curve and assert the folded accumulator's IPA
/// proof (`l_vec` / `r_vec` / `final_comm_key` / `c`) reproduces the golden.
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

    // The committer-key generators (the open core's sole runtime input) and the
    // golden folded accumulator's IPA proof it must reproduce.
    let generators: Vec<Affine<C>> =
        d["generators"].as_array().unwrap().iter().map(point_from_json::<C>).collect();
    let acc = &d["accumulator"];
    let want_l: Vec<Affine<C>> =
        acc["l_vec"].as_array().unwrap().iter().map(point_from_json::<C>).collect();
    let want_r: Vec<Affine<C>> =
        acc["r_vec"].as_array().unwrap().iter().map(point_from_json::<C>).collect();
    let want_final = point_from_json::<C>(&acc["final_comm_key"]);
    let want_c = fr_from_hex::<C>(acc["c"].as_str().unwrap());

    let mlirbc = std::fs::read(artifacts.join(artifact))
        .unwrap_or_else(|e| panic!("read {}: {}", artifact, e));
    let got = fused::open_ipa_fold_fused::<C>(&mlirbc, &generators);

    assert_eq!(got.l_vec, want_l, "folded accumulator ipa_proof.l_vec != arkworks");
    assert_eq!(got.r_vec, want_r, "folded accumulator ipa_proof.r_vec != arkworks");
    assert_eq!(got.final_comm_key, want_final, "folded accumulator ipa_proof.final_comm_key != arkworks");
    assert_eq!(got.c, want_c, "folded accumulator ipa_proof.c != arkworks");
    println!(
        "    IPA fold open ({} rounds: l_vec/r_vec/final_comm_key/c) byte-matches arkworks",
        got.l_vec.len()
    );
}

#[test]
#[ignore = "needs XLA_PJRT_PLUGIN + artifacts/ipa_fold_{pallas,vesta}.mlirbc + a GPU"]
fn gpu_fused_ipa_fold_byte_match() {
    println!("fused jax-exported IPA-PC accumulation FOLD open GPU byte-match (no-zk, Pallas + Vesta):");
    println!("  [pallas]");
    check_curve::<Pallas>("ipa_as_fold_fixtures.json", "ipa_fold_pallas.mlirbc");
    println!("  [vesta]");
    check_curve::<Vesta>("ipa_as_fold_vesta_fixtures.json", "ipa_fold_vesta.mlirbc");
    println!("ALL FUSED IPA-PC-AS FOLD OPEN GPU BYTE-MATCH CHECKS PASSED");
}
