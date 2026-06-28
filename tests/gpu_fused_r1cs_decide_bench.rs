//! Scale benchmark for the **R1CS-NARK accumulation decider** — the scheme's
//! GPU-value op (the six size-`n` Pedersen commitments `comm_{a,b,c}` +
//! `test_comm_{1,2,3}`, MSMs over `generators ‖ hiding`). Times the warm GPU
//! `fused::run_decide_r1cs` (one PJRT call computing all six MSMs +
//! the HP Hadamard product) against the arkworks CPU variable-base MSM at a
//! configurable size, with a byte-match-at-scale gate (GPU == CPU). Like the
//! IPA-PC decider (and unlike the host-bound recursion fold), the decider is pure
//! MSM, so it shows the GPU's MSM advantage as `n` grows.
//!
//! The inputs are synthetic (random generators + random `A·z`/`B·z`/`C·z` vectors
//! + random randomizers at size `n`) — the MSM time is size-driven, not
//! value-driven, and the values stay consistent CPU↔GPU so the correctness gate is
//! real. The core is the size-`n` pure-MSM decider load:
//!
//!     AS_DECIDE_SIZE=65536 export/export_as_decide.py        # -> as_decider_bench.mlirbc
//!     ZKX_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so \
//!     AS_DECIDE_SIZE=65536 AS_DECIDE_MLIRBC=.../as_decider_bench.mlirbc \
//!       cargo test --release --features gpu --test gpu_fused_r1cs_decide_bench -- --ignored --nocapture
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

/// `commit(scalars, randomizer) = Σ scalars_i·bases[i] + randomizer·bases[n]` —
/// the arkworks variable-base MSM over `generators ‖ hiding` the decider's
/// `cm_commit` runs.
fn cpu_commit(bases_h: &[Affine], scalars: &[Fr], randomizer: Fr) -> Affine {
    let mut s: Vec<Fr> = scalars.to_vec();
    s.push(randomizer);
    let repr: Vec<_> = s.iter().map(|x| x.into_repr()).collect();
    VariableBaseMSM::multi_scalar_mul(bases_h, &repr).into_affine()
}

#[test]
#[ignore = "scale bench: needs ZKX_PJRT_PLUGIN + AS_DECIDE_SIZE + AS_DECIDE_MLIRBC + a GPU"]
fn gpu_fused_r1cs_decide_bench() {
    let n: usize =
        std::env::var("AS_DECIDE_SIZE").expect("set AS_DECIDE_SIZE").parse().expect("size");
    let mlirbc_path = PathBuf::from(std::env::var("AS_DECIDE_MLIRBC").expect("set AS_DECIDE_MLIRBC"));
    let iters: usize =
        std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);

    // Synthetic size-`n` inputs (deterministic): committer key `generators ‖
    // hiding` (n+1 bases), the three reduced vectors `A·z`/`B·z`/`C·z`, and the six
    // randomizers. Setup is untimed.
    let mut rng = test_rng();
    let g = Affine::prime_subgroup_generator();
    let bases_h: Vec<Affine> =
        (0..n + 1).map(|_| g.mul(Fr::rand(&mut rng).into_repr()).into_affine()).collect();
    let av: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let bv: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let cv: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let rand6: Vec<Fr> = (0..6).map(|_| Fr::rand(&mut rng)).collect();
    let product: Vec<Fr> = av.iter().zip(&bv).map(|(a, b)| *a * b).collect();

    // CPU golden + timing: the six arkworks variable-base MSMs the decider runs
    // (`comm_a/b/c` + the HP `test_comm_1/2/3 = commit(a_vec, b_vec, a∘b)`).
    let t = Instant::now();
    let cpu = [
        cpu_commit(&bases_h, &av, rand6[0]),
        cpu_commit(&bases_h, &bv, rand6[1]),
        cpu_commit(&bases_h, &cv, rand6[2]),
        cpu_commit(&bases_h, &av, rand6[3]),
        cpu_commit(&bases_h, &bv, rand6[4]),
        cpu_commit(&bases_h, &product, rand6[5]),
    ];
    let cpu_ms = t.elapsed();

    // GPU: compile once, then warm runs. The first run doubles as warmup + the
    // byte-match-at-scale correctness gate (all six commitments).
    let mlirbc = std::fs::read(&mlirbc_path).expect("read mlirbc");
    let exe = fused::load_fused(&mlirbc);
    let got = fused::run_decide_r1cs::<Pallas>(exe, &bases_h, &av, &bv, &cv, &rand6);
    assert_eq!(got.len(), 6, "decider returned {} commitments, expected 6", got.len());
    for (i, (g, c)) in got.iter().zip(&cpu).enumerate() {
        assert_eq!(g, c, "decider commitment {i} GPU != CPU at n={n}");
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = fused::run_decide_r1cs::<Pallas>(exe, &bases_h, &av, &bv, &cv, &rand6);
        times.push(t.elapsed());
    }
    times.sort();
    let gpu_min = times[0];
    let gpu_median = times[times.len() / 2];
    let speedup = cpu_ms.as_secs_f64() / gpu_min.as_secs_f64();
    println!(
        "[bench] R1CS-NARK decider (6 size-{n} MSMs): CPU arkworks={cpu_ms:?} | GPU fused warm \
         min={gpu_min:?} median={gpu_median:?} over {iters} iters | speedup={speedup:.2}x \
         (byte-matches arkworks)"
    );
}
