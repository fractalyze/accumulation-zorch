//! Warm GPU timing for the **fused IPA-PC accumulation fold** open: compile the
//! baked fold open core once, then time many warm `run_ipa_fold_fused` calls — the
//! steady-state cost of one fused PJRT dispatch of the whole sequential
//! `IpaPC::open` (the `lax.scan` basis fold + on-device Poseidon per round + the
//! `final_comm_key` MSM), excluding the one-time compile.
//!
//! The IPA accumulate is field-heavy with small MSMs (8-coeff, 3 rounds at the
//! fixture degree), so — like the R1CS fold (host-bound, Amdahl-capped ~1.4×) — the
//! win is modest/host-bound; the point is to record it, completing the README's IPA
//! "accumulate" bench row alongside the decider.
//!
//! Fixture-driven: needs `python/testdata/ipa_as_fold_fixtures.json` +
//! `ipa_fold_pallas.mlirbc` under `$ACCUMULATION_ZORCH_ARTIFACTS`; SKIPS when the
//! `.mlirbc` is absent. Hardware-gated; run on an idle GPU:
//!
//!     XLA_PJRT_PLUGIN=.../xla_cuda_plugin.so \
//!       cargo test --release --features gpu --test gpu_fused_ipa_fold_bench -- --ignored --nocapture
#![cfg(feature = "gpu")]

mod common;

use accumulation_zorch::fused;
use accumulation_zorch::gpu::{Pallas, PastaCurve};
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use common::point_from_json;
use std::path::PathBuf;
use std::time::Instant;

type Affine<C> = GroupAffine<<C as PastaCurve>::Params>;

#[test]
#[ignore = "needs XLA_PJRT_PLUGIN + ipa_fold_pallas.mlirbc + a GPU; run --release"]
fn gpu_fused_ipa_fold_bench() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let artifacts = fixture_json::artifacts_dir(env!("CARGO_MANIFEST_DIR"));
    let mlirbc_path = artifacts.join("ipa_fold_pallas.mlirbc");
    if !mlirbc_path.exists() {
        println!("SKIP — missing {}", mlirbc_path.display());
        return;
    }
    let d: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("python/testdata/ipa_as_fold_fixtures.json"))
            .expect("read fixture"),
    )
    .unwrap();
    let generators: Vec<Affine<Pallas>> =
        d["generators"].as_array().unwrap().iter().map(point_from_json::<Pallas>).collect();
    let rounds = (generators.len() as u32).saturating_sub(1).next_power_of_two().trailing_zeros();
    let mlirbc = std::fs::read(&mlirbc_path).unwrap();

    // Compile once; the steady-state cost is the run, not the compile.
    let t_compile = Instant::now();
    let exe = fused::load_fused(&mlirbc);
    let compile = t_compile.elapsed();

    // Warm up (first run also pays JIT/allocation), then time N warm runs.
    let _ = fused::run_ipa_fold_fused::<Pallas>(exe, &generators);
    const N: usize = 20;
    let mut times: Vec<f64> = Vec::with_capacity(N);
    for _ in 0..N {
        let t = Instant::now();
        let got = fused::run_ipa_fold_fused::<Pallas>(exe, &generators);
        times.push(t.elapsed().as_secs_f64() * 1e3);
        let _ = std::hint::black_box(got.final_comm_key);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[N / 2];
    let mean = times.iter().sum::<f64>() / N as f64;
    println!(
        "fused IPA-PC-AS fold open (Pallas, {} coeffs, {rounds} rounds) warm GPU:\n  \
         compile {:.2}s (once); warm run median {median:.2} ms, mean {mean:.2} ms (min {:.2}, max {:.2}) over {N} runs",
        generators.len(),
        compile.as_secs_f64(),
        times[0],
        times[N - 1],
    );
}
