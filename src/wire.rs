//! The arkworks ↔ zk_dtypes Pasta on-the-wire byte bridge, generic over the
//! Pasta curve (Pallas or Vesta).
//!
//! The xla GPU `lax.msm` boundary is the zk_dtypes one: a field element is its
//! **standard-form** (non-Montgomery) little-endian 32-byte representation, a G1
//! affine point is `x ‖ y` (64 bytes), and the identity is all-zero. The GPU MSM
//! kernel dispatches on the bases' element type (`PALLAS_G1_AFFINE` /
//! `VESTA_G1_AFFINE`), so scalars and points must be serialized in exactly this
//! layout for the per-curve MSM runner to consume them.
//!
//! Pallas and Vesta are structurally identical here (32-byte standard-form LE
//! scalars, 64-byte `x ‖ y` G1 affine, identity all-zero), so this bridge is
//! generic over the short-Weierstrass parameters `P` and instantiated per curve.
//! Pasta is pairing-free, so G1 only (no G2).
//!
//! This conversion is the one correctness risk in the GPU backend: a wrong limb
//! order or a missed identity mapping silently corrupts the MSM. It is pure CPU
//! with no GPU dependency, so the layout is unit-tested here and re-validated
//! end-to-end by the GPU == CPU == arkworks byte-match.

use ark_ec::models::SWModelParameters;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ff::{BigInteger, PrimeField, Zero};

/// Bytes per field element on the wire (standard-form 32-byte LE).
pub const SF_BYTES: usize = 32;
/// Bytes per G1 affine point on the wire (`x ‖ y`).
pub const G1_BYTES: usize = 2 * SF_BYTES;
/// ark-ff `[u8]::to_field_elements` chunk size = base-field `CAPACITY / 8` (both
/// Pasta `fq` are CAPACITY 254 → 31). Curve-invariant.
const BYTES_PER_FE: usize = 31;

/// Standard-form little-endian 32-byte encoding of a field element.
///
/// `into_repr` leaves arkworks' internal Montgomery form, so the bytes are the
/// canonical value — matching the zk_dtypes boundary and the non-`_MONT`
/// `*_SF` / `*_G1_AFFINE` buffer types.
fn field_to_le<F: PrimeField>(f: &F) -> [u8; SF_BYTES] {
    let mut out = [0u8; SF_BYTES];
    let le = f.into_repr().to_bytes_le();
    debug_assert!(le.len() <= SF_BYTES, "Pasta field element exceeds 32 bytes");
    out[..le.len()].copy_from_slice(&le);
    out
}

/// Inverse of [`field_to_le`]. On-the-wire coordinates are always canonical
/// (`< modulus`), so the mod-order reduction is a no-op that returns the exact
/// value.
#[inline]
fn field_from_le<F: PrimeField>(bytes: &[u8]) -> F {
    F::from_le_bytes_mod_order(bytes)
}

/// Serializes a Pasta scalar to 32 standard-form little-endian bytes.
pub fn scalar_to_bytes<P: SWModelParameters>(s: &P::ScalarField) -> [u8; SF_BYTES] {
    field_to_le(s)
}

/// Concatenates `scalars` into the MSM scalar input (`32 · len` bytes).
pub fn scalars_to_bytes<P: SWModelParameters>(scalars: &[P::ScalarField]) -> Vec<u8> {
    let mut b = Vec::with_capacity(scalars.len() * SF_BYTES);
    for s in scalars {
        b.extend_from_slice(&scalar_to_bytes::<P>(s));
    }
    b
}

/// ark-ff `u8::batch_to_sponge_field_elements` of a `Vec<Fr>`: serialize the
/// scalars to their 32-byte-LE concatenation, prepend `(byte_len as u64).LE`, then
/// re-chunk into 31-byte little-endian groups each zero-padded to a 32-byte field
/// repr — the **base-field** sponge elements the classic-accumulation Fiat-Shamir
/// absorbs, on the wire (standard-form 32-byte LE each).
///
/// The general zk core feeds this consumer-side pre-encoding for the
/// `r1cs_input` / `r1cs_r_input` sponge absorbs: the in-trace `fr→u8` rechunk the
/// xla GPU plugin mis-lowers is done here instead. Each 31-byte chunk is
/// `< 2^248 < the Pasta base-field modulus`, so the zero-padded bytes are already
/// the canonical LE base-field encoding (no reduction). Mirrors
/// `python/accumulation_zorch/absorbable.py:u8_batch_field_array`.
pub fn u8_batch_field_array<P: SWModelParameters>(scalars: &[P::ScalarField]) -> Vec<u8> {
    let data = scalars_to_bytes::<P>(scalars);
    let mut buf = (data.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(&data);
    let mut out = Vec::with_capacity(buf.len().div_ceil(BYTES_PER_FE) * SF_BYTES);
    for chunk in buf.chunks(BYTES_PER_FE) {
        let mut fe = [0u8; SF_BYTES];
        fe[..chunk.len()].copy_from_slice(chunk);
        out.extend_from_slice(&fe);
    }
    out
}

/// Serializes a Pasta G1 affine point to 64 bytes (`x ‖ y`); the identity maps
/// to all-zero.
///
/// arkworks stores the identity as `(0, 1, ∞)`, so the explicit infinity guard
/// is required — a naive `x ‖ y` would emit `y = 1` instead of all-zero.
pub fn g1_to_bytes<P: SWModelParameters>(p: &GroupAffine<P>) -> [u8; G1_BYTES]
where
    P::BaseField: PrimeField,
{
    let mut out = [0u8; G1_BYTES];
    if p.is_zero() {
        return out;
    }
    out[..SF_BYTES].copy_from_slice(&field_to_le(&p.x));
    out[SF_BYTES..].copy_from_slice(&field_to_le(&p.y));
    out
}

/// Concatenates `points` into the MSM bases input (`64 · len` bytes).
pub fn g1_array_to_bytes<P: SWModelParameters>(points: &[GroupAffine<P>]) -> Vec<u8>
where
    P::BaseField: PrimeField,
{
    let mut b = Vec::with_capacity(points.len() * G1_BYTES);
    for p in points {
        b.extend_from_slice(&g1_to_bytes(p));
    }
    b
}

/// Reads a Pasta G1 affine point from 64 wire bytes (`x ‖ y`); all-zero is the
/// identity.
pub fn g1_from_bytes<P: SWModelParameters>(bytes: &[u8]) -> GroupAffine<P>
where
    P::BaseField: PrimeField,
{
    assert_eq!(bytes.len(), G1_BYTES, "G1 wire point must be 64 bytes");
    if bytes.iter().all(|&b| b == 0) {
        return GroupAffine::zero();
    }
    let x = field_from_le::<P::BaseField>(&bytes[..SF_BYTES]);
    let y = field_from_le::<P::BaseField>(&bytes[SF_BYTES..]);
    GroupAffine::new(x, y, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::{AffineCurve, ProjectiveCurve};
    use ark_ff::{One, UniformRand};
    use ark_pallas::{Affine, Fr, PallasParameters as P};

    /// `k · G` as an affine point, the canonical way to obtain non-trivial,
    /// on-curve test points.
    fn point(k: &Fr) -> Affine {
        Affine::prime_subgroup_generator()
            .mul(k.into_repr())
            .into_affine()
    }

    #[test]
    fn scalar_roundtrips() {
        let mut rng = ark_std::test_rng();
        for s in [Fr::zero(), Fr::one(), Fr::rand(&mut rng), Fr::rand(&mut rng)] {
            let back = field_from_le::<Fr>(&scalar_to_bytes::<P>(&s));
            assert_eq!(back, s);
        }
    }

    /// The decisive standard-form (non-Montgomery) check: the wire encoding of
    /// `1` is literally `01 00 .. 00`. In Montgomery form it would be `R mod p`,
    /// which is not `1`.
    #[test]
    fn scalar_one_is_standard_form() {
        let mut expected = [0u8; SF_BYTES];
        expected[0] = 1;
        assert_eq!(scalar_to_bytes::<P>(&Fr::one()), expected);
        assert_eq!(scalar_to_bytes::<P>(&Fr::zero()), [0u8; SF_BYTES]);
    }

    #[test]
    fn point_roundtrips() {
        let mut rng = ark_std::test_rng();
        let mut pts = vec![Affine::zero(), Affine::prime_subgroup_generator()];
        for _ in 0..4 {
            pts.push(point(&Fr::rand(&mut rng)));
        }
        for p in pts {
            let back = g1_from_bytes::<P>(&g1_to_bytes(&p));
            assert_eq!(back, p, "point did not survive the byte round-trip");
        }
    }

    #[test]
    fn identity_is_all_zero() {
        assert_eq!(g1_to_bytes(&Affine::zero()), [0u8; G1_BYTES]);
        assert!(g1_from_bytes::<P>(&[0u8; G1_BYTES]).is_zero());
        // A finite point must never serialize to all-zero (which means identity).
        assert_ne!(
            g1_to_bytes(&Affine::prime_subgroup_generator()),
            [0u8; G1_BYTES]
        );
    }

    /// The 64-byte layout is exactly `x_le ‖ y_le` for a finite point.
    #[test]
    fn point_layout_is_x_then_y() {
        let g = Affine::prime_subgroup_generator();
        let bytes = g1_to_bytes(&g);
        assert_eq!(&bytes[..SF_BYTES], &field_to_le(&g.x));
        assert_eq!(&bytes[SF_BYTES..], &field_to_le(&g.y));
    }

    /// `u8_batch_field_array([1])`: the byte stream is `(32 as u64).LE ‖ 1.LE32`
    /// (8 + 32 = 40 bytes), re-chunked into two 31-byte groups → two base-field
    /// elements (64 wire bytes). The length byte `0x20` lands first; the scalar
    /// value `0x01` lands at offset 8 (after the u64 prefix); everything else is
    /// zero. This pins the prefix + 31-byte-chunk layout the sponge absorbs.
    #[test]
    fn u8_batch_one_scalar_layout() {
        let out = u8_batch_field_array::<P>(&[Fr::one()]);
        assert_eq!(out.len(), 2 * SF_BYTES, "40 bytes → two 31-byte chunks");
        let mut expected = vec![0u8; 2 * SF_BYTES];
        expected[0] = 0x20; // (byte_len = 32) as u64 LE, low byte
        expected[8] = 0x01; // the scalar `1` begins after the 8-byte u64 prefix
        assert_eq!(out, expected);
    }

    /// The element count is `ceil((8 + 32·n) / 31)` and each element is a canonical
    /// 32-byte LE base-field value (top byte zero — a 31-byte chunk is `< 2^248`).
    #[test]
    fn u8_batch_element_count_and_canonical() {
        let mut rng = ark_std::test_rng();
        for n in [1usize, 5, 10] {
            let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let out = u8_batch_field_array::<P>(&scalars);
            let n_fe = (8 + 32 * n).div_ceil(31);
            assert_eq!(out.len(), n_fe * SF_BYTES, "fe count for n={n}");
            for fe in out.chunks(SF_BYTES) {
                assert_eq!(fe[31], 0, "31-byte chunk leaves the top wire byte zero");
            }
        }
    }

    #[test]
    fn array_byte_lengths() {
        let mut rng = ark_std::test_rng();
        let scalars: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();
        let points: Vec<Affine> = (0..5).map(|_| point(&Fr::rand(&mut rng))).collect();
        assert_eq!(scalars_to_bytes::<P>(&scalars).len(), 5 * SF_BYTES);
        assert_eq!(g1_array_to_bytes(&points).len(), 5 * G1_BYTES);
    }
}
