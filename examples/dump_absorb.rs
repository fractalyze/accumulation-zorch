//! Absorbable + fork + NARK gamma-challenge fixtures for the frx port.
//!
//! Reconstructs, with the *public* ark-sponge / ark_pallas API (the same way
//! `dump_sponge.rs` reconstructs the reference sponge), the field-element
//! packing that `Absorbable` and `CryptographicSponge::fork` feed into the
//! Poseidon sponge, plus the real NARK `compute_challenge` (gamma) absorb
//! ordering. Each fixture is a fresh sponge driven through one primitive so the
//! Python port can byte-match it in isolation:
//!
//!   * `identity_to_field_elements_dec` — arkworks `Affine::zero()` packs as
//!     `[x=0, y=1, infinity=1]` (NOT the all-zero zk_dtypes encoding) — the one
//!     point→field trap. Dumped as decimals so the Python side pins it directly.
//!   * `fork`         — `Sponge::new().fork(domain)` then squeeze, per protocol.
//!   * `bytes_absorb` — absorb a raw `&[u8]` (the `u8` batch length-prefix +
//!     31-byte = CAPACITY/8 chunking) at chunk-boundary lengths, then squeeze.
//!   * `point_absorb` — absorb SW-affine points (incl. the identity), squeeze.
//!   * `gamma`        — the full `R1CSNark::compute_challenge`: fork the base
//!     sponge with `R1CS-NARK-2020`, absorb the 32-byte matrices hash, the
//!     scalar inputs (each 32B LE), and the no-zk `FirstRoundMessage`
//!     (`comm_a,comm_b,comm_c` + `None` randomness), then truncated-128 squeeze.
//!
//! Run: `cargo run --example dump_absorb > python/testdata/absorb_fixtures.json`

use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{BigInteger, PrimeField, ToConstraintField, Zero};
use ark_pallas::{Affine, Fq, Fr};
use ark_sponge::poseidon::PoseidonSponge;
use ark_sponge::{CryptographicSponge, FieldElementSize};
use serde::Serialize;

use fixture_json::{fe_hex, fe_list, hex, PointJson};

const PROTOCOL_NAMES: [&str; 3] = ["R1CS-NARK-2020", "AS-FOR-R1CS-NARK-2020", "AS-FOR-HP-2020"];

/// Squeeze `n` Fq elements from `sponge` and render them as canonical-LE hex.
fn squeeze_hex(sponge: &mut PoseidonSponge<Fq>, n: usize) -> Vec<String> {
    fe_list(&sponge.squeeze_field_elements(n))
}

/// `{infinity, x_le_hex, y_le_hex}` of a point — the coords the Python side
/// rebuilds the point from (and the input the point→field packing consumes).
#[derive(Serialize)]
struct AbsorbPoint {
    infinity: bool,
    #[serde(flatten)]
    coords: PointJson,
}

fn point_json(p: &Affine) -> AbsorbPoint {
    AbsorbPoint {
        infinity: p.is_zero(),
        coords: PointJson::from_affine(p),
    }
}

#[derive(Serialize)]
struct ForkEntry {
    domain_utf8: String,
    domain_hex: String,
    squeeze: Vec<String>,
}

#[derive(Serialize)]
struct BytesEntry {
    len: usize,
    data_hex: String,
    squeeze: Vec<String>,
}

#[derive(Serialize)]
struct PointAbsorbEntry {
    label: String,
    points: Vec<AbsorbPoint>,
    squeeze: Vec<String>,
}

#[derive(Serialize)]
struct Gamma {
    matrices_hash_hex: String,
    inputs_le_hex: Vec<String>,
    comms: Vec<AbsorbPoint>,
    randomness: Option<String>,
    gamma_hex: String,
}

#[derive(Serialize)]
struct AbsorbFixture {
    note: String,
    identity_to_field_elements_le_hex: Vec<String>,
    fork: Vec<ForkEntry>,
    bytes_absorb: Vec<BytesEntry>,
    point_absorb: Vec<PointAbsorbEntry>,
    gamma: Gamma,
}

fn main() {
    let g = Affine::prime_subgroup_generator();
    let mul = |k: u64| g.mul(Fr::from(k).into_repr()).into_affine();

    // --- anchor: arkworks `Affine::zero().to_field_elements()` = [0, 1, 1],
    //     as canonical-LE Fq hex (each 32B). ---
    let id_fes: Vec<String> = fe_list::<Fq>(&Affine::zero().to_field_elements().unwrap());

    // --- fork: Sponge::new().fork(domain) then squeeze 2. ---
    let fork_json: Vec<ForkEntry> = PROTOCOL_NAMES
        .iter()
        .map(|name| {
            let mut sp = PoseidonSponge::<Fq>::new().fork(name.as_bytes());
            ForkEntry {
                domain_utf8: name.to_string(),
                domain_hex: hex(name.as_bytes()),
                squeeze: squeeze_hex(&mut sp, 2),
            }
        })
        .collect();

    // --- bytes_absorb: absorb a raw &[u8] (u8 batch) then squeeze 2. Lengths
    //     straddle the 31-byte (CAPACITY/8) chunk boundary. ---
    let byte_lens: [usize; 6] = [0, 5, 31, 32, 40, 63];
    let bytes_json: Vec<BytesEntry> = byte_lens
        .iter()
        .map(|&len| {
            let data: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(7).wrapping_add(1)).collect();
            let mut sp = PoseidonSponge::<Fq>::new();
            sp.absorb(&data.as_slice());
            BytesEntry {
                len,
                data_hex: hex(&data),
                squeeze: squeeze_hex(&mut sp, 2),
            }
        })
        .collect();

    // --- point_absorb: absorb a list of points then squeeze 2. ---
    let point_cases: Vec<(&str, Vec<Affine>)> = vec![
        ("generator", vec![g]),
        ("identity", vec![Affine::zero()]),
        ("two_then_id", vec![mul(2), Affine::zero(), mul(12345)]),
    ];
    let point_json_entries: Vec<PointAbsorbEntry> = point_cases
        .iter()
        .map(|(label, pts)| {
            let mut sp = PoseidonSponge::<Fq>::new();
            for p in pts {
                sp.absorb(p);
            }
            PointAbsorbEntry {
                label: label.to_string(),
                points: pts.iter().map(point_json).collect(),
                squeeze: squeeze_hex(&mut sp, 2),
            }
        })
        .collect();

    // --- gamma: the real R1CSNark::compute_challenge absorb ordering. ---
    // Arbitrary-but-fixed inputs replayed by the Python port.
    let matrices_hash: [u8; 32] = {
        let mut h = [0u8; 32];
        for (i, b) in h.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(11).wrapping_add(3);
        }
        h
    };
    let input_scalars: Vec<Fr> = vec![1u64, 2, 3, 1000, 1u64 << 40]
        .into_iter()
        .map(Fr::from)
        .collect();
    let (comm_a, comm_b, comm_c) = (mul(3), mul(5), mul(7));

    let mut sponge = PoseidonSponge::<Fq>::new().fork(b"R1CS-NARK-2020");
    sponge.absorb(&matrices_hash.as_ref());
    let input_bytes: Vec<u8> = input_scalars
        .iter()
        .flat_map(|s| s.into_repr().to_bytes_le())
        .collect();
    sponge.absorb(&input_bytes);
    // FirstRoundMessage (no-zk) to_sponge_field_elements: comm_a ++ comm_b ++
    // comm_c ++ Option::None (=> [Fq::from(false)]), absorbed as one Vec<Fq>.
    let mut frm_fes: Vec<Fq> = Vec::new();
    frm_fes.extend(comm_a.to_field_elements().unwrap());
    frm_fes.extend(comm_b.to_field_elements().unwrap());
    frm_fes.extend(comm_c.to_field_elements().unwrap());
    frm_fes.push(Fq::from(false));
    sponge.absorb(&frm_fes);
    let gamma: Fr = sponge
        .squeeze_nonnative_field_elements_with_sizes::<Fr>(&[FieldElementSize::Truncated(128)])
        .pop()
        .unwrap();

    let gamma_json = Gamma {
        matrices_hash_hex: hex(&matrices_hash),
        // Each scalar as canonical-LE Fq^Fr 32B hex — exactly the bytes the prover
        // concatenates via `into_repr().to_bytes_le()` into `input_bytes`.
        inputs_le_hex: fe_list(&input_scalars),
        comms: [comm_a, comm_b, comm_c].iter().map(point_json).collect(),
        randomness: None,
        gamma_hex: fe_hex(&gamma),
    };

    let fixture = AbsorbFixture {
        note: "ark-sponge Absorbable + fork + NARK gamma fixtures".to_string(),
        identity_to_field_elements_le_hex: id_fes,
        fork: fork_json,
        bytes_absorb: bytes_json,
        point_absorb: point_json_entries,
        gamma: gamma_json,
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}
