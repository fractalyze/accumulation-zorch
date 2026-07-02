//! Scale benchmark for the IPA-PC accumulation **decider MSM** — the scheme's
//! GPU-value op (`final_key = Σ generators_i·coeffs_i`, size `d`). Times the warm
//! GPU `fused::run_decide_ipa_msm` (one PJRT call) against the arkworks CPU
//! variable-base MSM at a configurable size, with a byte-match-at-scale gate
//! (GPU == CPU). Unlike the recursion fold (host-bound, Amdahl-capped at ~1.4×),
//! the decider is a pure size-`d` MSM, so it shows the GPU's MSM advantage as `d`
//! grows.
//!
//! The inputs are synthetic (random generators + random scalars at size `n`) —
//! the MSM time depends on the size, not the values, and the values stay
//! consistent CPU↔GPU so the correctness gate is real. The core is the general
//! `lax.msm` lowered at size `n`:
//!
//!     IPA_DECIDER_SIZE=65536 export/export_ipa.py            # -> ipa_decider_msm_bench.mlirbc
//!     XLA_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so \
//!     IPA_DECIDER_SIZE=65536 IPA_DECIDER_MLIRBC=.../ipa_decider_msm_bench.mlirbc \
//!       cargo test --release --features gpu --test gpu_fused_ipa_decide_bench -- --ignored --nocapture
#![cfg(feature = "gpu")]

use accumulation_zorch::fused;
use accumulation_zorch::gpu::Pallas;
use ark_ec::msm::VariableBaseMSM;
use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{PrimeField, UniformRand};
use ark_pallas::{Affine, Fr};
use ark_std::test_rng;
use std::path::PathBuf;
use std::time::Instant;

#[test]
#[ignore = "scale bench: needs XLA_PJRT_PLUGIN + IPA_DECIDER_SIZE + IPA_DECIDER_MLIRBC + a GPU"]
fn gpu_fused_ipa_decide_bench() {
    let n: usize =
        std::env::var("IPA_DECIDER_SIZE").expect("set IPA_DECIDER_SIZE").parse().expect("size");
    let mlirbc_path = PathBuf::from(std::env::var("IPA_DECIDER_MLIRBC").expect("set IPA_DECIDER_MLIRBC"));
    let iters: usize =
        std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);

    // Synthetic size-`n` committer key + check-poly coefficients (deterministic).
    // The MSM time is size-driven, not value-driven; matching values on both sides
    // keep the byte-match gate meaningful. Setup is untimed.
    let mut rng = test_rng();
    let g = Affine::prime_subgroup_generator();
    let generators: Vec<Affine> =
        (0..n).map(|_| g.mul(Fr::rand(&mut rng).into_repr()).into_affine()).collect();
    let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

    // CPU golden + timing: the arkworks variable-base MSM the decider's `cm_commit`
    // runs (`final_key = Σ generators_i·coeffs_i`).
    let scalars_repr: Vec<_> = coeffs.iter().map(|c| c.into_repr()).collect();
    let t = Instant::now();
    let cpu = VariableBaseMSM::multi_scalar_mul(&generators, &scalars_repr).into_affine();
    let cpu_ms = t.elapsed();

    // GPU: compile once, then warm runs. The first run doubles as warmup + the
    // byte-match-at-scale correctness gate.
    let mlirbc = std::fs::read(&mlirbc_path).expect("read mlirbc");
    let exe = fused::load_fused(&mlirbc);
    let got = fused::run_decide_ipa_msm::<Pallas>(exe, &coeffs, &generators);
    assert_eq!(got, cpu, "decider MSM GPU != CPU at n={n}");

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = fused::run_decide_ipa_msm::<Pallas>(exe, &coeffs, &generators);
        times.push(t.elapsed());
    }
    times.sort();
    let gpu_min = times[0];
    let gpu_median = times[times.len() / 2];
    let speedup = cpu_ms.as_secs_f64() / gpu_min.as_secs_f64();
    println!(
        "[bench] decider MSM n={n}: CPU arkworks={cpu_ms:?} | GPU fused warm min={gpu_min:?} \
         median={gpu_median:?} over {iters} iters | speedup={speedup:.2}x (byte-matches arkworks)"
    );
}
