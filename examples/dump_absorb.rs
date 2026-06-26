//! Absorbable + fork + NARK gamma-challenge fixtures for the jax port.
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
use ark_serialize::CanonicalSerialize;
use ark_sponge::poseidon::PoseidonSponge;
use ark_sponge::{CryptographicSponge, FieldElementSize};

const PROTOCOL_NAMES: [&str; 3] = ["R1CS-NARK-2020", "AS-FOR-R1CS-NARK-2020", "AS-FOR-HP-2020"];

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Canonical 32-byte LE serialization of a field element.
fn fe_hex<F: CanonicalSerialize>(f: &F) -> String {
    let mut b = Vec::new();
    f.serialize(&mut b).unwrap();
    hex(&b)
}

/// Squeeze `n` Fq elements from `sponge` and render them as canonical-LE hex.
fn squeeze_hex(sponge: &mut PoseidonSponge<Fq>, n: usize) -> Vec<String> {
    sponge
        .squeeze_field_elements(n)
        .iter()
        .map(|f| format!("\"{}\"", fe_hex(f)))
        .collect()
}

/// `{infinity, x_le_hex, y_le_hex}` of a point — the coords the Python side
/// rebuilds the point from (and the input the point→field packing consumes).
fn point_json(p: &Affine) -> String {
    let (x, y) = if p.is_zero() {
        (hex(&[0u8; 32]), hex(&[0u8; 32]))
    } else {
        (fe_hex(&p.x), fe_hex(&p.y))
    };
    format!(
        "{{\"infinity\":{},\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}}",
        p.is_zero(),
        x,
        y
    )
}

fn main() {
    let g = Affine::prime_subgroup_generator();
    let mul = |k: u64| g.mul(Fr::from(k).into_repr()).into_affine();

    // --- anchor: arkworks `Affine::zero().to_field_elements()` = [0, 1, 1],
    //     as canonical-LE Fq hex (each 32B). ---
    let id_fes: Vec<String> = Affine::zero()
        .to_field_elements()
        .unwrap()
        .iter()
        .map(|f: &Fq| format!("\"{}\"", fe_hex(f)))
        .collect();

    // --- fork: Sponge::new().fork(domain) then squeeze 2. ---
    let fork_json: Vec<String> = PROTOCOL_NAMES
        .iter()
        .map(|name| {
            let mut sp = PoseidonSponge::<Fq>::new().fork(name.as_bytes());
            format!(
                "{{\"domain_utf8\":\"{}\",\"domain_hex\":\"{}\",\"squeeze\":[{}]}}",
                name,
                hex(name.as_bytes()),
                squeeze_hex(&mut sp, 2).join(",")
            )
        })
        .collect();

    // --- bytes_absorb: absorb a raw &[u8] (u8 batch) then squeeze 2. Lengths
    //     straddle the 31-byte (CAPACITY/8) chunk boundary. ---
    let byte_lens: [usize; 6] = [0, 5, 31, 32, 40, 63];
    let bytes_json: Vec<String> = byte_lens
        .iter()
        .map(|&len| {
            let data: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(7).wrapping_add(1)).collect();
            let mut sp = PoseidonSponge::<Fq>::new();
            sp.absorb(&data.as_slice());
            format!(
                "{{\"len\":{},\"data_hex\":\"{}\",\"squeeze\":[{}]}}",
                len,
                hex(&data),
                squeeze_hex(&mut sp, 2).join(",")
            )
        })
        .collect();

    // --- point_absorb: absorb a list of points then squeeze 2. ---
    let point_cases: Vec<(&str, Vec<Affine>)> = vec![
        ("generator", vec![g]),
        ("identity", vec![Affine::zero()]),
        ("two_then_id", vec![mul(2), Affine::zero(), mul(12345)]),
    ];
    let point_json_entries: Vec<String> = point_cases
        .iter()
        .map(|(label, pts)| {
            let mut sp = PoseidonSponge::<Fq>::new();
            for p in pts {
                sp.absorb(p);
            }
            let pts_json: Vec<String> = pts.iter().map(point_json).collect();
            format!(
                "{{\"label\":\"{}\",\"points\":[{}],\"squeeze\":[{}]}}",
                label,
                pts_json.join(","),
                squeeze_hex(&mut sp, 2).join(",")
            )
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

    // Each scalar as canonical-LE Fq^Fr 32B hex — exactly the bytes the prover
    // concatenates via `into_repr().to_bytes_le()` into `input_bytes`.
    let inputs_le_hex: Vec<String> = input_scalars
        .iter()
        .map(|s| format!("\"{}\"", fe_hex(s)))
        .collect();
    let comms_json = [comm_a, comm_b, comm_c]
        .iter()
        .map(point_json)
        .collect::<Vec<_>>()
        .join(",");
    let gamma_json = format!(
        "{{\"matrices_hash_hex\":\"{}\",\"inputs_le_hex\":[{}],\"comms\":[{}],\
         \"randomness\":null,\"gamma_hex\":\"{}\"}}",
        hex(&matrices_hash),
        inputs_le_hex.join(","),
        comms_json,
        fe_hex(&gamma),
    );

    println!("{{");
    println!(
        "  \"note\": \"ark-sponge Absorbable + fork + NARK gamma fixtures\","
    );
    println!("  \"identity_to_field_elements_le_hex\": [{}],", id_fes.join(","));
    println!("  \"fork\": [{}],", fork_json.join(","));
    println!("  \"bytes_absorb\": [{}],", bytes_json.join(","));
    println!("  \"point_absorb\": [{}],", point_json_entries.join(","));
    println!("  \"gamma\": {}", gamma_json);
    println!("}}");
}
