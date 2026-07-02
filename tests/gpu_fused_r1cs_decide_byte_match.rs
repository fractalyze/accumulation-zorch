//! End-to-end GPU byte-match for the **R1CS-NARK accumulation decider** fused
//! core — the decider's six size-`n` MSMs run as **one** PJRT call
//! (`crate::fused::run_decide_r1cs`) must reproduce the accumulator's arkworks
//! `comm_{a,b,c}` and `hp_instance.comm_{1,2,3}` byte-for-byte, over **both** Pasta
//! cycle curves and both the no-zk and zk accumulators.
//!
//! Fixture-driven and faithful to arkworks: each fixture's accumulator is the
//! unmodified arkworks `(acc.instance ‖ acc.witness)`, emitted only for an
//! accumulator arkworks decides `true`. The golden commitments live in
//! `acc_instance_hex` (compressed points); the decider inputs — `z = r1cs_input ‖
//! r1cs_blinded_witness` and the six randomizers `(σ_a,σ_b,σ_c,ρ₁,ρ₂,ρ₃)` (0 on
//! the no-zk path) — are parsed as scalars from `acc_{instance,witness}_hex`. The
//! same zk-agnostic `as_decider_<curve>.mlirbc` decides both modes (the zk-ness
//! lives in the randomizers); `export/export_as_decide.py` lowers it (CPU, no GPU).
//!
//! Hardware-gated; run only when the GPU is idle:
//!
//!     XLA_PJRT_PLUGIN=.../pjrt_c_api_gpu_plugin.so \
//!       cargo test --features gpu --test gpu_fused_r1cs_decide_byte_match -- --ignored --test-threads=1 --nocapture
#![cfg(feature = "gpu")]

mod common;

use accumulation_zorch::fused;
use accumulation_zorch::gpu::{Pallas, PastaCurve, Vesta};
use ark_ec::models::ModelParameters;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ff::{PrimeField, Zero};
use ark_serialize::CanonicalSerialize;
use common::{point_from_json, to_hex};
use serde_json::Value;
use std::path::PathBuf;

type Affine<C> = GroupAffine<<C as PastaCurve>::Params>;
type Fr<C> = <<C as PastaCurve>::Params as ModelParameters>::ScalarField;
type Fq<C> = <<C as PastaCurve>::Params as ModelParameters>::BaseField;

const FR_BYTES: usize = 32;
const POINT_BYTES: usize = 33;

fn take_fr_vec<C: PastaCurve>(buf: &[u8], off: &mut usize) -> Vec<Fr<C>> {
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&buf[*off..*off + 8]);
    let n = u64::from_le_bytes(len_bytes) as usize;
    *off += 8;
    (0..n)
        .map(|_| {
            let f = Fr::<C>::from_le_bytes_mod_order(&buf[*off..*off + FR_BYTES]);
            *off += FR_BYTES;
            f
        })
        .collect()
}

/// A 33-byte compressed point as hex (compared byte-for-byte against the GPU
/// output's compressed serialization — no decompression).
fn take_point_hex(buf: &[u8], off: &mut usize) -> String {
    let h = to_hex(&buf[*off..*off + POINT_BYTES]);
    *off += POINT_BYTES;
    h
}

fn take_opt_triple<C: PastaCurve>(buf: &[u8], off: &mut usize) -> Option<[Fr<C>; 3]> {
    let flag = buf[*off];
    *off += 1;
    if flag == 0 {
        return None;
    }
    let mut t = [Fr::<C>::zero(); 3];
    for f in t.iter_mut() {
        *f = Fr::<C>::from_le_bytes_mod_order(&buf[*off..*off + FR_BYTES]);
        *off += FR_BYTES;
    }
    Some(t)
}

fn artifacts() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::env::var("ACCUMULATION_ZORCH_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("artifacts"))
}

/// Decide every seed of one fixture on the GPU and assert the six recomputed
/// commitments byte-match the accumulator's stored ones.
fn check_curve<C: PastaCurve>(fixture: &str, artifact: &str, zk: bool)
where
    Fq<C>: PrimeField,
{
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let d: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("python/testdata").join(fixture)).expect("read fixture"),
    )
    .expect("parse fixture json");

    let generators: Vec<Affine<C>> =
        d["generators"].as_array().unwrap().iter().map(point_from_json::<C>).collect();
    let hiding = point_from_json::<C>(&d["hiding"]);
    let mut bases_h = generators;
    bases_h.push(hiding);

    let mlirbc = std::fs::read(artifacts().join(artifact))
        .unwrap_or_else(|e| panic!("read {}: {}", artifact, e));
    let exe = fused::load_fused(&mlirbc);
    let labels = ["comm_a", "comm_b", "comm_c", "hp_comm_1", "hp_comm_2", "hp_comm_3"];

    for seed in d["seeds"].as_array().unwrap() {
        assert!(seed["decide"].as_bool().unwrap(), "fixture seed not arkworks-decided");

        // AccumulatorInstance: r1cs_input (Vec<Fr>), then the six golden commitments.
        let inst = common::from_hex(seed["acc_instance_hex"].as_str().unwrap());
        let mut o = 0usize;
        let r1cs_input = take_fr_vec::<C>(&inst, &mut o);
        let want: Vec<String> = (0..6).map(|_| take_point_hex(&inst, &mut o)).collect();

        // AccumulatorWitness: blinded_witness, hp a_vec/b_vec, Some/None hp rand,
        // Some/None accumulator randomness (σ).
        let wit = common::from_hex(seed["acc_witness_hex"].as_str().unwrap());
        let mut o = 0usize;
        let blinded_witness = take_fr_vec::<C>(&wit, &mut o);
        let hp_a_vec = take_fr_vec::<C>(&wit, &mut o);
        let hp_b_vec = take_fr_vec::<C>(&wit, &mut o);
        let hp_rand = take_opt_triple::<C>(&wit, &mut o);
        let sigmas = take_opt_triple::<C>(&wit, &mut o);
        assert_eq!(hp_rand.is_some(), zk, "hp randomness Some-ness must match zk mode");
        assert_eq!(sigmas.is_some(), zk, "accumulator randomness Some-ness must match zk mode");

        let mut z = r1cs_input;
        z.extend(blinded_witness);
        // rand6 = (σ_a,σ_b,σ_c, ρ₁,ρ₂,ρ₃); all 0 on the no-zk path.
        let rand6: Vec<Fr<C>> = match (sigmas, hp_rand) {
            (Some(s), Some(r)) => vec![s[0], s[1], s[2], r[0], r[1], r[2]],
            _ => vec![Fr::<C>::zero(); 6],
        };

        let got = fused::run_decide_r1cs::<C>(exe, &bases_h, &z, &hp_a_vec, &hp_b_vec, &rand6);
        for (i, label) in labels.iter().enumerate() {
            let mut b = Vec::new();
            got[i].serialize(&mut b).unwrap();
            assert_eq!(to_hex(&b), want[i], "[{}] seed {} {label}", fixture, seed["seed"]);
        }
        println!(
            "    [{}] seed {}: 6 decider commitments byte-match arkworks ({} vars)",
            fixture, seed["seed"], z.len()
        );
    }
}

#[test]
#[ignore = "needs XLA_PJRT_PLUGIN + artifacts/as_decider_{pallas,vesta}.mlirbc + a GPU"]
fn gpu_fused_r1cs_decide_byte_match() {
    println!("fused jax-exported R1CS-NARK accumulation DECIDER GPU byte-match (no-zk + zk, Pallas + Vesta):");
    println!("  [pallas, no-zk]");
    check_curve::<Pallas>("as_fixtures.json", "as_decider_pallas.mlirbc", false);
    println!("  [vesta, no-zk]");
    check_curve::<Vesta>("as_vesta_fixtures.json", "as_decider_vesta.mlirbc", false);
    println!("  [pallas, zk]");
    check_curve::<Pallas>("as_zk_fixtures.json", "as_decider_pallas.mlirbc", true);
    println!("  [vesta, zk]");
    check_curve::<Vesta>("as_zk_vesta_fixtures.json", "as_decider_vesta.mlirbc", true);
    println!("ALL FUSED R1CS-NARK-AS DECIDER GPU BYTE-MATCH CHECKS PASSED");
}
