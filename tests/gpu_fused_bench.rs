//! Scale benchmark for the fused jax-exported prove core: warm GPU run time of
//! `fused::run_fused` (one PJRT call) at a configurable circuit size, with a
//! byte-match-at-scale correctness gate against the fixture golden. Pairs with
//! the CPU `[bench]` timing the `dump_as_zk` example prints (the arkworks AS
//! prove), to answer "GPU vs arkworks accumulation" across sizes.
//!
//! Drive it with an off-tree large fixture + the general core's exported `.mlirbc`
//! (one seed-independent artifact; the fixture's first seed supplies
//! the runtime witness/randomness):
//!
//!     AS_ZK_NUM_CONSTRAINTS=16384 cargo run --release --example dump_as_zk > fix.json
//!     AS_ZK_FIXTURE=fix.json export/export_prove.py   # -> prove_zk_general.mlirbc
//!     ZKX_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so \
//!     AS_ZK_FIXTURE=.../fix.json FUSED_MLIRBC=.../prove_zk_general.mlirbc \
//!       cargo test --release --features gpu --test gpu_fused_bench -- --ignored --nocapture
#![cfg(feature = "gpu")]

mod common;

use accumulation_zorch::fused;
use accumulation_zorch::gpu::Pallas;
use ark_ff::Zero;
use ark_pallas::Affine;
use std::path::PathBuf;
use std::time::Instant;

#[test]
#[ignore = "scale bench: needs ZKX_PJRT_PLUGIN + AS_ZK_FIXTURE + FUSED_MLIRBC + a GPU"]
fn gpu_fused_bench() {
    let fixture = PathBuf::from(std::env::var("AS_ZK_FIXTURE").expect("set AS_ZK_FIXTURE"));
    let mlirbc_path = PathBuf::from(std::env::var("FUSED_MLIRBC").expect("set FUSED_MLIRBC"));
    let iters: usize =
        std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);

    // Phase A: parse fixture, build inputs (no PJRT client).
    let d: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("read fixture")).unwrap();
    let n = d["num_constraints"].as_u64().unwrap() as usize;
    let generators: Vec<Affine> =
        d["generators"].as_array().unwrap().iter().map(common::point_from_json::<Pallas>).collect();
    let mut bases_h = generators[..n].to_vec();
    bases_h.push(common::point_from_json::<Pallas>(&d["hiding"]));
    let id_pt = Affine::zero();

    // The general core proves any seed; benchmark the fixture's first seed.
    let s = &d["seeds"][0];
    let seed = s["seed"].as_u64().unwrap();
    let inputs = common::zk_inputs_from_seed::<Pallas>(s);
    let mlirbc = std::fs::read(&mlirbc_path).expect("read mlirbc");

    // Phase B: compile once (incl. Client_Compile), then time warm runs.
    let exe = fused::load_fused(&mlirbc);

    // First run doubles as warmup + the byte-match-at-scale correctness gate.
    let got = fused::run_fused::<Pallas>(exe, &bases_h, &id_pt, &inputs);
    assert_eq!(common::to_hex(&got.acc_instance), s["acc_instance_hex"].as_str().unwrap(), "acc.instance");
    assert_eq!(common::to_hex(&got.acc_witness), s["acc_witness_hex"].as_str().unwrap(), "acc.witness");
    assert_eq!(common::to_hex(&got.proof), s["proof_hex"].as_str().unwrap(), "proof");

    let mut times: Vec<std::time::Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = fused::run_fused::<Pallas>(exe, &bases_h, &id_pt, &inputs);
        times.push(t.elapsed());
    }
    times.sort();
    let min = times[0];
    let median = times[times.len() / 2];
    println!(
        "[bench] n={n} seed={seed} GPU fused warm run (one PJRT call): min={min:?} median={median:?} over {iters} iters \
         (byte-matches arkworks; out {}B)",
        got.concat().len()
    );
}
