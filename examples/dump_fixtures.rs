//! Substrate fixtures for the frx prover-port byte-match.
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
use ark_ff::{One, PrimeField, Zero};
use ark_pallas::{Affine, Fq, Fr};
use ark_poly_commit::trivial_pc::PedersenCommitment;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::Serialize;

use fixture_json::{hex, ser_hex, PointJson};

/// Hex of a value's arkworks *uncompressed* canonical serialization. The
/// compressed form is `fixture_json::ser_hex`; only this dumper pins both.
fn ser_uncompressed<T: CanonicalSerialize>(v: &T) -> String {
    let mut b = Vec::new();
    v.serialize_uncompressed(&mut b).unwrap();
    hex(&b)
}

/// Decimal string of a prime-field element (via its big-integer repr).
fn dec<F: PrimeField>(v: &F) -> String {
    v.into_repr().to_string()
}

/// One field-element fixture: its value and its canonical (compressed) bytes.
#[derive(Serialize)]
struct FieldEntry {
    label: String,
    value: String,
    canonical_hex: String,
}

fn field_entry<F: PrimeField>(label: &str, v: &F) -> FieldEntry {
    FieldEntry {
        label: label.to_string(),
        value: dec(v),
        canonical_hex: ser_hex(v),
    }
}

#[derive(Serialize)]
struct Fields {
    fq: Vec<FieldEntry>,
    fr: Vec<FieldEntry>,
}

/// One point fixture. `coords` is flattened so `x_le_hex`/`y_le_hex` land
/// between `infinity` and `canonical_hex`, matching the fixture's key order.
#[derive(Serialize)]
struct PointEntry {
    label: String,
    infinity: bool,
    #[serde(flatten)]
    coords: PointJson,
    canonical_hex: String,
    uncompressed_hex: String,
}

fn point_entry(label: &str, p: &Affine) -> PointEntry {
    PointEntry {
        label: label.to_string(),
        infinity: p.is_zero(),
        // x/y as standalone Fq canonical bytes (32B LE each) — the raw coords the
        // zk_dtypes 64B `x‖y` affine encoding is built from on the Python side.
        coords: PointJson::from_affine(p),
        canonical_hex: ser_hex(p),
        uncompressed_hex: ser_uncompressed(p),
    }
}

/// One Pedersen commit case: the (optional) randomizer and the resulting commitment.
#[derive(Serialize)]
struct PedersenCase {
    randomizer: Option<String>,
    result_canonical_hex: String,
}

#[derive(Serialize)]
struct Pedersen {
    n: usize,
    generators: Vec<PointJson>,
    hiding: PointJson,
    elems: Vec<String>,
    cases: Vec<PedersenCase>,
}

#[derive(Serialize)]
struct SubstrateFixture {
    note: String,
    fq_modulus_minus_one: String,
    fr_modulus_minus_one: String,
    fields: Fields,
    points: Vec<PointEntry>,
    pedersen: Pedersen,
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

    let fq_json: Vec<FieldEntry> = fq_vals.iter().map(|(l, v)| field_entry(l, v)).collect();
    let fr_json: Vec<FieldEntry> = fr_vals.iter().map(|(l, v)| field_entry(l, v)).collect();

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

    let pedersen = Pedersen {
        n,
        generators: fixture_json::point_list(&generators[..elems.len()]),
        hiding: PointJson::from_affine(&hiding),
        elems: elem_vals.iter().map(|v| v.to_string()).collect(),
        cases: vec![
            PedersenCase {
                randomizer: None,
                result_canonical_hex: ser_hex(&result_plain),
            },
            PedersenCase {
                randomizer: Some(randomizer.to_string()),
                result_canonical_hex: ser_hex(&result_hiding),
            },
        ],
    };

    let fixture = SubstrateFixture {
        note: "arkworks ark_pallas CanonicalSerialize fixtures".to_string(),
        fq_modulus_minus_one: dec(&(-Fq::one())), // p-1; p = that + 1
        fr_modulus_minus_one: dec(&(-Fr::one())),
        fields: Fields {
            fq: fq_json,
            fr: fr_json,
        },
        points,
        pedersen,
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}
