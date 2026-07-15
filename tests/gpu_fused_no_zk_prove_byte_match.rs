//! End-to-end GPU byte-match for the **no-zk** fused frx-exported prove core
//! — the no-zk twin of `gpu_fused_prove_byte_match.rs`. The
//! whole no-zk `ASForR1CSNark` prove runs as **one** PJRT call
//! (`crate::fused::prove_no_zk_fused`) and must serialize byte-for-byte to the
//! golden `acc.instance ‖ acc.witness ‖ proof` arkworks produces.
//!
//! Fixture-driven: `python/testdata/as_fixtures.json` holds the committer key
//! (`generators`), the per-seed `r1cs_input` / `blinded_witness`, and the golden
//! output hex. This exercises the **general** no-zk core:
//! `export/export_prove.py no-zk` lowers ONE `prove_no_zk_general.mlirbc` whose
//! runtime inputs are the committer key `bases = generators[:rows]` PLUS the
//! assignment (`r1cs_input` / `blinded_witness`) — so one core proves every seed,
//! fed each seed's assignment at run time (no per-seed `.mlirbc`).
//!
//! **Separate test file (separate binary) on purpose:** the plugin preallocates
//! ~75% VRAM on first client use, so two fused executables in one process exhaust
//! the pool (`CUDA_ERROR_OUT_OF_MEMORY`). Splitting the zk and no-zk gates into
//! distinct binaries lets `cargo test` run them in separate processes, each with a
//! fresh GPU. Hardware-gated; run only when the GPU is idle:
//!
//!     XLA_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so \
//!       cargo test --features gpu --test gpu_fused_no_zk_prove_byte_match -- --ignored --test-threads=1
#![cfg(feature = "gpu")]

use accumulation_zorch::fused;
use accumulation_zorch::gpu::Pallas;
use ark_ff::PrimeField;
use ark_pallas::{Affine, Fq, Fr};
use std::path::PathBuf;

/// Decode an even-length lowercase hex string to bytes.
fn from_hex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd-length hex");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

/// Encode bytes to a lowercase hex string (matches the fixture's `*_hex`).
fn to_hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{:02x}", x));
    }
    s
}

/// A canonical-LE field element from its fixture hex.
fn fr_from_hex(s: &str) -> Fr {
    Fr::from_le_bytes_mod_order(&from_hex(s))
}

/// An affine point from a fixture `{x_le_hex, y_le_hex}` object (finite — the
/// committer-key points are never the identity).
fn point_from_json(v: &serde_json::Value) -> Affine {
    let x = Fq::from_le_bytes_mod_order(&from_hex(v["x_le_hex"].as_str().unwrap()));
    let y = Fq::from_le_bytes_mod_order(&from_hex(v["y_le_hex"].as_str().unwrap()));
    Affine::new(x, y, false)
}

#[test]
#[ignore = "needs XLA_PJRT_PLUGIN + artifacts/prove_no_zk_general.mlirbc + a GPU"]
fn gpu_fused_no_zk_prove_byte_match() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("python/testdata/as_fixtures.json");
    let d: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("read fixture"))
            .expect("parse fixture json");
    let artifacts = std::env::var("ACCUMULATION_ZORCH_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("artifacts"));

    // --- Phase A: the general no-zk core takes the committer key
    // `bases = generators[:rows]` (no hiding base) plus the assignment
    // (`r1cs_input` / `blinded_witness`) as runtime inputs. ONE core, loaded once,
    // fed each seed's assignment below; the assignment also serializes host-side.
    let rows = d["num_constraints"].as_u64().unwrap() as usize;
    let generators: Vec<Affine> =
        d["generators"].as_array().unwrap().iter().map(point_from_json).collect();
    let bases = generators[..rows].to_vec();
    let mlirbc = std::fs::read(artifacts.join("prove_no_zk_general.mlirbc"))
        .unwrap_or_else(|e| panic!("read prove_no_zk_general.mlirbc: {}", e));

    struct Case {
        seed: u64,
        r1cs_input: Vec<Fr>,
        blinded_witness: Vec<Fr>,
        acc_instance_hex: String,
        acc_witness_hex: String,
        proof_hex: String,
    }
    let fr_vec = |v: &serde_json::Value| -> Vec<Fr> {
        v.as_array().unwrap().iter().map(|h| fr_from_hex(h.as_str().unwrap())).collect()
    };
    let cases: Vec<Case> = d["seeds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| Case {
            seed: s["seed"].as_u64().unwrap(),
            r1cs_input: fr_vec(&s["r1cs_input"]),
            blinded_witness: fr_vec(&s["blinded_witness"]),
            acc_instance_hex: s["acc_instance_hex"].as_str().unwrap().to_string(),
            acc_witness_hex: s["acc_witness_hex"].as_str().unwrap().to_string(),
            proof_hex: s["proof_hex"].as_str().unwrap().to_string(),
        })
        .collect();
    assert!(!cases.is_empty(), "fixture has no seeds");

    // --- Phase B: ONE general core, per-seed runtime assignment, byte-match each.
    println!("fused frx-exported GENERAL NO-ZK prove core GPU byte-match:");
    for c in &cases {
        let got =
            fused::prove_no_zk_fused::<Pallas>(&mlirbc, &bases, &c.r1cs_input, &c.blinded_witness);
        assert_eq!(
            to_hex(&got.acc_instance),
            c.acc_instance_hex,
            "seed {} acc.instance diverged",
            c.seed
        );
        assert_eq!(
            to_hex(&got.acc_witness),
            c.acc_witness_hex,
            "seed {} acc.witness diverged",
            c.seed
        );
        assert_eq!(to_hex(&got.proof), c.proof_hex, "seed {} proof diverged", c.seed);
        println!(
            "  seed {}: (acc.instance {}B ‖ acc.witness {}B ‖ proof {}B) byte-matches arkworks (one general core)",
            c.seed,
            got.acc_instance.len(),
            got.acc_witness.len(),
            got.proof.len()
        );
    }
    println!("ALL FUSED GENERAL NO-ZK-CORE GPU BYTE-MATCH CHECKS PASSED");
}
