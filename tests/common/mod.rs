//! Shared fixture parsing for the fused-prove GPU tests (the byte-match gate and
//! the scale bench). Builds the general zk core's runtime inputs from a fixture
//! seed entry — the same per-seed witness/randomness `export/export_prove.py
//! build_core` reads — so one lowered `prove_zk_general.mlirbc` is exercised per
//! seed.
#![allow(dead_code)] // each `mod common;` test crate uses a subset of these.

use accumulation_zorch::fused::ZkProveInputs;
use accumulation_zorch::gpu::PastaCurve;
use ark_ec::models::ModelParameters;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ff::PrimeField;
use serde_json::Value;

type Affine<C> = GroupAffine<<C as PastaCurve>::Params>;
type Fr<C> = <<C as PastaCurve>::Params as ModelParameters>::ScalarField;
type Fq<C> = <<C as PastaCurve>::Params as ModelParameters>::BaseField;

/// Decode an even-length lowercase hex string to bytes.
pub fn from_hex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd-length hex");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

/// Encode bytes to a lowercase hex string (matches the fixture's `*_hex`).
pub fn to_hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{:02x}", x));
    }
    s
}

/// A canonical-LE scalar field element from its fixture hex (the `fe_hex` form).
pub fn fr_from_hex<C: PastaCurve>(s: &str) -> Fr<C> {
    Fr::<C>::from_le_bytes_mod_order(&from_hex(s))
}

/// An affine point from a fixture `{x_le_hex, y_le_hex}` object (canonical-LE
/// coordinates; finite — the committer-key points are never the identity).
pub fn point_from_json<C: PastaCurve>(v: &Value) -> Affine<C>
where
    Fq<C>: PrimeField,
{
    let x = Fq::<C>::from_le_bytes_mod_order(&from_hex(v["x_le_hex"].as_str().unwrap()));
    let y = Fq::<C>::from_le_bytes_mod_order(&from_hex(v["y_le_hex"].as_str().unwrap()));
    GroupAffine::<C::Params>::new(x, y, false)
}

fn fr_vec<C: PastaCurve>(arr: &Value) -> Vec<Fr<C>> {
    arr.as_array().unwrap().iter().map(|h| fr_from_hex::<C>(h.as_str().unwrap())).collect()
}

/// Build the general zk core's runtime inputs from a fixture seed entry `s`,
/// mirroring `r1cs_nark_as._build_zk_core`'s host-arg order: the assignment, the
/// NARK `r` + 8 blinders, the AS `r1cs_r_*` (= `vec![rand; len]`) + 3 commitment
/// blinders, and the 5 HP randomness values `[hiding_a, hiding_b, hr1, hr2, hr3]`.
pub fn zk_inputs_from_seed<C: PastaCurve>(s: &Value) -> ZkProveInputs<C> {
    let g = |k: &str| fr_from_hex::<C>(s[k].as_str().unwrap());
    let r1cs_input = fr_vec::<C>(&s["r1cs_input"]);
    let witness = fr_vec::<C>(&s["witness"]);
    let input_len = r1cs_input.len();
    let witness_len = witness.len();
    ZkProveInputs {
        r1cs_input,
        witness,
        nark_r: fr_vec::<C>(&s["r"]),
        nark_blinders: ["a_blinder", "b_blinder", "c_blinder", "r_a_blinder", "r_b_blinder",
                        "r_c_blinder", "blinder_1", "blinder_2"]
            .iter()
            .map(|k| g(k))
            .collect(),
        r1cs_r_input: vec![g("as_r1cs_r_input"); input_len],
        r1cs_r_witness: vec![g("as_r1cs_r_witness"); witness_len],
        as_rand: ["as_rand_1", "as_rand_2", "as_rand_3"].iter().map(|k| g(k)).collect(),
        hp_rand: ["hp_hiding_a", "hp_hiding_b", "hp_rand_1", "hp_rand_2", "hp_rand_3"]
            .iter()
            .map(|k| g(k))
            .collect(),
    }
}
