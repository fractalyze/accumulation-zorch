//! Warm GPU timing for the **fused zk IVC fold** prove (zorch#326, the
//! informational acceptance item): compile the fold core once, then time many warm
//! `run_fold_fused` calls — the steady-state cost of one fused PJRT fold dispatch
//! at recursion scale (~77.5K constraints, 2¹⁷-MSM class), excluding the one-time
//! compile. Pairs with the arkworks fold-prove --release timing
//! (`recursion_step` `vesta::arkworks_fold_timing`) for the GPU-vs-arkworks
//! comparison.
//!
//! Fixture-driven and off-tree (Vesta forward direction): needs
//! `recursion_fold_zk_fixtures.json` + `fold_zk_vesta.mlirbc` under
//! `$ACCUMULATION_ZORCH_ARTIFACTS`; SKIPS when absent. Hardware-gated; run on an
//! idle GPU:
//!
//!     ZKX_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so ACCUMULATION_ZORCH_ARTIFACTS=<dir> \
//!       cargo test --release --features gpu --test gpu_fused_fold_bench -- --ignored --nocapture
#![cfg(feature = "gpu")]

use accumulation_zorch::fused;
use accumulation_zorch::gpu::{PastaCurve, Vesta};
use ark_ec::models::ModelParameters;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ff::{PrimeField, Zero};
use std::path::PathBuf;
use std::time::Instant;

type Affine<C> = GroupAffine<<C as PastaCurve>::Params>;
type Fr<C> = <<C as PastaCurve>::Params as ModelParameters>::ScalarField;
type Base<C> = <<C as PastaCurve>::Params as ModelParameters>::BaseField;

fn from_hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex")).collect()
}

fn point_from_json<C: PastaCurve>(v: &serde_json::Value) -> Affine<C>
where
    Base<C>: PrimeField,
{
    let x = Base::<C>::from_le_bytes_mod_order(&from_hex(v["x_le_hex"].as_str().unwrap()));
    let y = Base::<C>::from_le_bytes_mod_order(&from_hex(v["y_le_hex"].as_str().unwrap()));
    GroupAffine::new(x, y, false)
}

#[test]
#[ignore = "needs ZKX_PJRT_PLUGIN + off-tree Vesta fold fixture + fold_zk_vesta.mlirbc + a GPU; run --release"]
fn gpu_fused_fold_bench() {
    let artifacts = std::env::var("ACCUMULATION_ZORCH_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("artifacts"));
    let fixture = artifacts.join("recursion_fold_zk_fixtures.json");
    let mlirbc_path = artifacts.join("fold_zk_vesta.mlirbc");
    if !fixture.exists() || !mlirbc_path.exists() {
        println!("SKIP — missing {} or {}", fixture.display(), mlirbc_path.display());
        return;
    }
    let d: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("read fixture")).unwrap();

    let rows = d["num_constraints"].as_u64().unwrap() as usize;
    let generators: Vec<Affine<Vesta>> =
        d["generators"].as_array().unwrap().iter().map(point_from_json::<Vesta>).collect();
    let mut bases_h = generators[..rows].to_vec();
    bases_h.push(point_from_json::<Vesta>(&d["hiding"]));
    let id_pt = Affine::<Vesta>::zero();
    let acc = &d["acc_prev_instance"];
    let acc_comms: Vec<Affine<Vesta>> =
        ["comm_a", "comm_b", "comm_c", "hp_comm_1", "hp_comm_2", "hp_comm_3"]
            .iter()
            .map(|k| point_from_json::<Vesta>(&acc[*k]))
            .collect();
    let input_len = d["input2_r1cs_input"].as_array().unwrap().len();
    let as_r = Fr::<Vesta>::from_le_bytes_mod_order(&from_hex(d["as_r1cs_r_input"].as_str().unwrap()));
    let r1cs_r_input = vec![as_r; input_len];
    let mlirbc = std::fs::read(&mlirbc_path).unwrap();

    // Compile once; the steady-state cost is the run, not the compile.
    let t_compile = Instant::now();
    let exe = fused::load_fused(&mlirbc);
    let compile = t_compile.elapsed();

    // Warm up (first run also pays JIT/allocation), then time N warm runs.
    let _ = fused::run_fold_fused::<Vesta>(exe, &bases_h, &id_pt, &acc_comms, &r1cs_r_input);
    const N: usize = 20;
    let mut times: Vec<f64> = Vec::with_capacity(N);
    for _ in 0..N {
        let t = Instant::now();
        let got = fused::run_fold_fused::<Vesta>(exe, &bases_h, &id_pt, &acc_comms, &r1cs_r_input);
        times.push(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(got.concat());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[N / 2];
    let mean = times.iter().sum::<f64>() / N as f64;
    println!(
        "fused zk IVC fold (Vesta, {rows} constraints, 2¹⁷-MSM class) warm GPU prove:\n  \
         compile {:.2}s (once); warm run median {median:.1} ms, mean {mean:.1} ms (min {:.1}, max {:.1}) over {N} runs",
        compile.as_secs_f64(),
        times[0],
        times[N - 1],
    );
}
