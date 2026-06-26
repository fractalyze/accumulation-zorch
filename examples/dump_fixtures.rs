//! Slice-1 substrate fixtures for the jax prover-port byte-match (zorch#303).
//!
//! Emits, as JSON, the arkworks `CanonicalSerialize` bytes of a handful of
//! `ark_pallas::Fq` / `Fr` field elements and `ark_pallas::Affine` points (the
//! same `serialize` call `oracle.rs` drives), so the Python side in
//! `python/` can reproduce them byte-for-byte. This pins, empirically:
//!   * the Fq/Fr ↔ zk_dtypes (`vesta_sf`/`pallas_sf`) mapping,
//!   * the point ↔ curve dtype mapping (`pallas_g1` vs `vesta_g1`), and
//!   * arkworks' compressed SW-affine layout (x LE + flag bits).
//!
//! Run: `cargo run --example dump_fixtures > python/testdata/substrate_fixtures.json`

use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{Field, One, PrimeField, Zero};
use ark_pallas::{Affine, Fq, Fr};
use ark_poly_commit::trivial_pc::PedersenCommitment;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn ser_compressed<T: CanonicalSerialize>(v: &T) -> String {
    let mut b = Vec::new();
    v.serialize(&mut b).unwrap();
    hex(&b)
}

fn ser_uncompressed<T: CanonicalSerialize>(v: &T) -> String {
    let mut b = Vec::new();
    v.serialize_uncompressed(&mut b).unwrap();
    hex(&b)
}

/// Decimal string of a prime-field element (via its big-integer repr).
fn dec<F: PrimeField>(v: &F) -> String {
    v.into_repr().to_string()
}

fn field_entry<F: PrimeField>(label: &str, v: &F) -> String {
    format!(
        "{{\"label\":\"{}\",\"value\":\"{}\",\"canonical_hex\":\"{}\"}}",
        label,
        dec(v),
        ser_compressed(v),
    )
}

fn point_entry(label: &str, p: &Affine) -> String {
    // x/y as standalone Fq canonical bytes (32B LE each) — the raw coords the
    // zk_dtypes 64B `x‖y` affine encoding is built from on the Python side.
    let (x_hex, y_hex) = if p.is_zero() {
        (hex(&[0u8; 32]), hex(&[0u8; 32]))
    } else {
        (ser_compressed(&p.x), ser_compressed(&p.y))
    };
    format!(
        "{{\"label\":\"{}\",\"infinity\":{},\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\",\
         \"canonical_hex\":\"{}\",\"uncompressed_hex\":\"{}\"}}",
        label,
        p.is_zero(),
        x_hex,
        y_hex,
        ser_compressed(p),
        ser_uncompressed(p),
    )
}

fn main() {
    let g = Affine::prime_subgroup_generator();
    let mul = |k: u64| g.mul(Fr::from(k).into_repr()).into_affine();

    // Field fixtures: 0, 1, 2, 12345, p-1 over both Fq (base) and Fr (scalar).
    let fq_vals: Vec<(String, Fq)> = vec![
        ("zero".into(), Fq::zero()),
        ("one".into(), Fq::one()),
        ("two".into(), Fq::from(2u64)),
        ("k12345".into(), Fq::from(12345u64)),
        ("modulus_minus_one".into(), -Fq::one()),
    ];
    let fr_vals: Vec<(String, Fr)> = vec![
        ("zero".into(), Fr::zero()),
        ("one".into(), Fr::one()),
        ("two".into(), Fr::from(2u64)),
        ("k12345".into(), Fr::from(12345u64)),
        ("modulus_minus_one".into(), -Fr::one()),
    ];

    let fq_json: Vec<String> = fq_vals.iter().map(|(l, v)| field_entry(l, v)).collect();
    let fr_json: Vec<String> = fr_vals.iter().map(|(l, v)| field_entry(l, v)).collect();

    // Point fixtures: generator, identity, 2G, 12345G (+ a sum to exercise add).
    let points = vec![
        point_entry("generator", &g),
        point_entry("identity", &Affine::zero()),
        point_entry("two_g", &mul(2)),
        point_entry("k12345_g", &mul(12345)),
        point_entry(
            "g_plus_2g",
            &(g.into_projective() + mul(2).into_projective()).into_affine(),
        ),
    ];

    // Pedersen commit fixture: the real `PedersenCommitment::{setup,trim,commit}`
    // path the prover drives, dumped so Python can replay the same generators
    // and byte-match the result of a CPU MSM reduction (no-hiding + hiding).
    let coords_json = |p: &Affine| -> String {
        let (x, y) = if p.is_zero() {
            (hex(&[0u8; 32]), hex(&[0u8; 32]))
        } else {
            (ser_compressed(&p.x), ser_compressed(&p.y))
        };
        format!("{{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}}", x, y)
    };

    let n = 8usize;
    let pp = PedersenCommitment::<Affine>::setup(n);
    let ck = PedersenCommitment::<Affine>::trim(&pp, n);
    // Recover generators + hiding generator from the key's uncompressed canonical
    // form (fields are pub(crate)); field order: `generators: Vec<G>`, `hiding`.
    let (generators, hiding) = {
        let mut b = Vec::new();
        ck.serialize_uncompressed(&mut b).unwrap();
        let mut r = &b[..];
        let g = Vec::<Affine>::deserialize_uncompressed(&mut r).unwrap();
        let h = Affine::deserialize_uncompressed(&mut r).unwrap();
        (g, h)
    };
    let elem_vals: Vec<u64> = vec![3, 5, 7];
    let elems: Vec<Fr> = elem_vals.iter().map(|&v| Fr::from(v)).collect();
    let randomizer: u64 = 99;
    let result_plain = PedersenCommitment::<Affine>::commit(&ck, &elems, None);
    let result_hiding =
        PedersenCommitment::<Affine>::commit(&ck, &elems, Some(Fr::from(randomizer)));

    let gens_json: Vec<String> = generators[..elems.len()]
        .iter()
        .map(coords_json)
        .collect();
    let elems_json: Vec<String> = elem_vals.iter().map(|v| format!("\"{}\"", v)).collect();
    let pedersen_json = format!(
        "{{\"n\":{},\"generators\":[{}],\"hiding\":{},\"elems\":[{}],\
         \"cases\":[{{\"randomizer\":null,\"result_canonical_hex\":\"{}\"}},\
         {{\"randomizer\":\"{}\",\"result_canonical_hex\":\"{}\"}}]}}",
        n,
        gens_json.join(","),
        coords_json(&hiding),
        elems_json.join(","),
        ser_compressed(&result_plain),
        randomizer,
        ser_compressed(&result_hiding),
    );

    let fq_modulus = dec(&(-Fq::one())); // p-1; p = that + 1
    let fr_modulus = dec(&(-Fr::one()));

    println!("{{");
    println!("  \"note\": \"arkworks ark_pallas CanonicalSerialize fixtures for zorch#303 slice 1\",");
    println!("  \"fq_modulus_minus_one\": \"{}\",", fq_modulus);
    println!("  \"fr_modulus_minus_one\": \"{}\",", fr_modulus);
    println!("  \"fields\": {{");
    println!("    \"fq\": [{}],", fq_json.join(","));
    println!("    \"fr\": [{}]", fr_json.join(","));
    println!("  }},");
    println!("  \"points\": [{}],", points.join(","));
    println!("  \"pedersen\": {}", pedersen_json);
    println!("}}");
}
