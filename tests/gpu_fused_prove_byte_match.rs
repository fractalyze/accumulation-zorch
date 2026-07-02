//! End-to-end GPU byte-match for the **general** fused jax-exported zk prove core:
//! the whole zk `ASForR1CSNark` prove run as **one** PJRT call
//! (`crate::fused::run_fused`) must serialize byte-for-byte to the golden
//! `acc.instance ‖ acc.witness ‖ proof` arkworks produces — the same target the
//! per-MSM `GpuBackend` byte-match (`gpu_prove_byte_match.rs`) hits, but with a
//! single fused dispatch instead of one per commitment.
//!
//! Fixture-driven: `python/testdata/as_zk_fixtures.json` holds the committer key
//! (`generators`, `hiding`) and, per seed, the assignment + replayed randomness +
//! golden output hex. This exercises the **general** core:
//! `export/export_prove.py` lowers ONE seed-independent `prove_zk_general.mlirbc`
//! whose runtime inputs are the committer key PLUS the assignment + all replayed
//! randomness (NARK/AS/HP) — so one core, compiled once, proves every seed fed its
//! own witness/randomness at run time (no per-seed `.mlirbc`).
//!
//! Two-phase to avoid monopolizing the shared GPU (the plugin preallocates ~75%
//! VRAM on first client use): Phase A parses the fixture + builds all inputs
//! touching no PJRT client; Phase B creates the (leaked) client, compiles the one
//! core, and runs it per seed. Hardware-gated; run only when the GPU is idle:
//!
//!     XLA_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so \
//!       cargo test --features gpu --test gpu_fused_prove_byte_match -- --ignored --test-threads=1
#![cfg(feature = "gpu")]

mod common;

use accumulation_zorch::fused::{self, ZkProveInputs};
use accumulation_zorch::gpu::Pallas;
use ark_ff::Zero;
use ark_pallas::Affine;
use std::path::PathBuf;

#[test]
#[ignore = "needs XLA_PJRT_PLUGIN + artifacts/prove_zk_general.mlirbc + a GPU"]
fn gpu_fused_prove_byte_match() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("python/testdata/as_zk_fixtures.json");
    let d: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("read fixture"))
            .expect("parse fixture json");
    let artifacts = std::env::var("ACCUMULATION_ZORCH_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("artifacts"));

    // --- Phase A: build the runtime inputs + golden, touching no PJRT client.
    // The committer key is the fused core's affine input: `bases_h =
    // generators[:rows] ‖ hiding` (rows = num_constraints), `id_pt` = identity.
    let rows = d["num_constraints"].as_u64().unwrap() as usize;
    let generators: Vec<Affine> =
        d["generators"].as_array().unwrap().iter().map(common::point_from_json::<Pallas>).collect();
    let mut bases_h = generators[..rows].to_vec();
    bases_h.push(common::point_from_json::<Pallas>(&d["hiding"]));
    let id_pt = Affine::zero();

    struct Case {
        seed: u64,
        inputs: ZkProveInputs<Pallas>,
        acc_instance_hex: String,
        acc_witness_hex: String,
        proof_hex: String,
    }
    let cases: Vec<Case> = d["seeds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| Case {
            seed: s["seed"].as_u64().unwrap(),
            inputs: common::zk_inputs_from_seed::<Pallas>(s),
            acc_instance_hex: s["acc_instance_hex"].as_str().unwrap().to_string(),
            acc_witness_hex: s["acc_witness_hex"].as_str().unwrap().to_string(),
            proof_hex: s["proof_hex"].as_str().unwrap().to_string(),
        })
        .collect();
    assert!(!cases.is_empty(), "fixture has no seeds");

    // --- Phase B: compile the ONE general core once, run it per seed (each fed its
    // own witness/randomness), and byte-match per component.
    println!("general fused jax-exported prove core GPU byte-match (one core × seeds):");
    let mlirbc = std::fs::read(artifacts.join("prove_zk_general.mlirbc"))
        .unwrap_or_else(|e| panic!("read prove_zk_general.mlirbc: {}", e));
    let exe = fused::load_fused(&mlirbc);
    for c in &cases {
        let got = fused::run_fused::<Pallas>(exe, &bases_h, &id_pt, &c.inputs);
        assert_eq!(
            common::to_hex(&got.acc_instance),
            c.acc_instance_hex,
            "seed {} acc.instance diverged",
            c.seed
        );
        assert_eq!(
            common::to_hex(&got.acc_witness),
            c.acc_witness_hex,
            "seed {} acc.witness diverged",
            c.seed
        );
        assert_eq!(common::to_hex(&got.proof), c.proof_hex, "seed {} proof diverged", c.seed);
        println!(
            "  seed {}: (acc.instance {}B ‖ acc.witness {}B ‖ proof {}B) byte-matches arkworks via the one general core",
            c.seed,
            got.acc_instance.len(),
            got.acc_witness.len(),
            got.proof.len()
        );
    }
    println!("ALL GENERAL FUSED-CORE GPU BYTE-MATCH CHECKS PASSED");
}
