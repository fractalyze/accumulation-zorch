//! The golden-fixture JSON codec, shared by both sides of the byte-match gate.
//!
//! The fixture dumpers (`examples/dump_*.rs`) write these types with
//! `serde_json`; the GPU byte-match tests (`tests/gpu_fused_*.rs`) read them
//! back. Both directions live here so that pair shares one definition —
//! previously the writers hand-rolled it 13 times and the readers another 5, and
//! the copies had begun to drift.
//!
//! **Not yet the only definition in the repo.** `tests/recursion_step.rs`, which
//! writes the off-tree recursion fixtures, still carries its own copies (two, one
//! per cycle direction) and does not use this crate — so the recursion gate's
//! writer and reader are on separate encodings, including separate identity
//! guards. Folding it in is tracked separately; until then, a change to the
//! shapes here must be mirrored there by hand.
//!
//! The point encoding is **not re-derived**: [`PointJson`] delegates to
//! [`accumulation_zorch::wire`], which is also what the GPU MSM boundary uses.
//! That module's doc names the trap this crate therefore cannot fall into —
//! arkworks stores the identity as `(0, 1, ∞)`, so an infinity guard is required
//! or the identity serializes with `y = 1`.
//!
//! Everything is generic over `SWModelParameters` rather than the `gpu`-gated
//! `PastaCurve`, so `examples/` (built without the `gpu` feature) and `tests/`
//! (built with it) can share one crate.

use ark_ec::models::{ModelParameters, SWModelParameters};
use ark_ec::short_weierstrass_jacobian::GroupAffine;
use ark_ff::{BigInteger, PrimeField};
use ark_relations::r1cs::Matrix;
use ark_serialize::CanonicalSerialize;
use serde::{Deserialize, Serialize};

use accumulation_zorch::wire::{self, G1_BYTES, SF_BYTES};

// Re-exported so `curve_main!` expands to paths that resolve without the caller
// importing the curve crates.
pub use ark_pallas;
pub use ark_vesta;

/// The base field of a short-Weierstrass curve (`Fq`).
pub type Fq<P> = <P as ModelParameters>::BaseField;

/// Encode bytes as a lowercase hex string — the fixtures' `*_hex` form.
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        // Infallible: the only error a String sink can report is allocation failure,
        // which aborts. `write!` formats straight into `s`; `push_str(&format!(..))`
        // would allocate and drop a String per byte.
        write!(&mut s, "{:02x}", b).unwrap();
    }
    s
}

/// Decode an even-length lowercase hex string.
///
/// The length check is what makes a malformed fixture say so, instead of
/// panicking somewhere inside `str` slicing.
pub fn from_hex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd-length hex");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

/// Canonical-LE hex of a field element (leaves arkworks' Montgomery form, so the
/// bytes are the value, matching the wire encoding).
pub fn fe_hex<F: PrimeField>(f: &F) -> String {
    hex(&f.into_repr().to_bytes_le())
}

/// Inverse of [`fe_hex`]. Fixture coordinates are canonical (`< modulus`), so the
/// reduction is a no-op returning the exact value.
pub fn fe_from_hex<F: PrimeField>(s: &str) -> F {
    F::from_le_bytes_mod_order(&from_hex(s))
}

/// Hex of a value's arkworks canonical serialization — the golden bytes the
/// byte-match compares.
pub fn ser_hex<T: CanonicalSerialize>(v: &T) -> String {
    let mut b = Vec::new();
    v.serialize(&mut b).expect("canonical serialization");
    hex(&b)
}

/// Field elements as their fixture hex list.
pub fn fe_list<F: PrimeField>(xs: &[F]) -> Vec<String> {
    xs.iter().map(fe_hex).collect()
}

/// An affine point as the fixture's `{x_le_hex, y_le_hex}` object.
///
/// The identity is all-zero in **both** coordinates, per the wire encoding — not
/// arkworks' in-memory `(0, 1, ∞)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointJson {
    pub x_le_hex: String,
    pub y_le_hex: String,
}

impl PointJson {
    /// Encode an affine point, delegating to [`wire::g1_to_bytes`] so the
    /// identity guard has one implementation.
    pub fn from_affine<P: SWModelParameters>(p: &GroupAffine<P>) -> Self
    where
        Fq<P>: PrimeField,
    {
        let b = wire::g1_to_bytes(p);
        PointJson {
            x_le_hex: hex(&b[..SF_BYTES]),
            y_le_hex: hex(&b[SF_BYTES..]),
        }
    }

    /// Decode back to an affine point, via [`wire::g1_from_bytes`] (all-zero maps
    /// to the identity).
    pub fn to_affine<P: SWModelParameters>(&self) -> GroupAffine<P>
    where
        Fq<P>: PrimeField,
    {
        let mut b = [0u8; G1_BYTES];
        b[..SF_BYTES].copy_from_slice(&from_hex(&self.x_le_hex));
        b[SF_BYTES..].copy_from_slice(&from_hex(&self.y_le_hex));
        wire::g1_from_bytes(&b)
    }
}

/// Affine points as fixture point objects.
pub fn point_list<P: SWModelParameters>(ps: &[GroupAffine<P>]) -> Vec<PointJson>
where
    Fq<P>: PrimeField,
{
    ps.iter().map(PointJson::from_affine).collect()
}

/// One sparse matrix row entry: `[coeff_le_hex, var_index]`.
///
/// A tuple so serde emits the fixtures' 2-element array rather than an object.
pub type MatrixEntry = (String, usize);
/// The sparse `Matrix<F>` layout: `[[[coeff_le_hex, var_index], ...], ...]`.
pub type MatrixJson = Vec<Vec<MatrixEntry>>;

/// Encode an R1CS matrix in the fixtures' sparse row layout.
pub fn matrix_json<F: PrimeField>(m: &Matrix<F>) -> MatrixJson {
    m.iter()
        .map(|row| row.iter().map(|(coeff, idx)| (fe_hex(coeff), *idx)).collect())
        .collect()
}

/// Blake2b-256 over `domain ‖ a ‖ b ‖ c`, matching the upstream index hash the
/// NARK and AS bind their matrices with.
pub fn hash_matrices<F: PrimeField>(
    domain: &[u8],
    a: &Matrix<F>,
    b: &Matrix<F>,
    c: &Matrix<F>,
) -> [u8; 32] {
    use blake2::VarBlake2b;
    use digest::{Update, VariableOutput};

    let mut serialized = domain.to_vec();
    a.serialize(&mut serialized).unwrap();
    b.serialize(&mut serialized).unwrap();
    c.serialize(&mut serialized).unwrap();
    let mut hasher = VarBlake2b::new(32).unwrap();
    hasher.update(&serialized);
    let mut out = [0u8; 32];
    hasher.finalize_variable(|res| out.copy_from_slice(res));
    out
}

/// The off-tree artifacts directory (`$ACCUMULATION_ZORCH_ARTIFACTS`, default
/// `<manifest>/artifacts`) holding the large recursion fixtures and the exported
/// `.mlirbc` cores — neither is committed.
///
/// `manifest_dir` is the caller's `env!("CARGO_MANIFEST_DIR")`: this crate's own
/// manifest dir is `crates/fixture-json`, not the repo root.
pub fn artifacts_dir(manifest_dir: &str) -> std::path::PathBuf {
    std::env::var("ACCUMULATION_ZORCH_ARTIFACTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::Path::new(manifest_dir).join("artifacts"))
}

/// The `main` every curve-generic dumper needs: dispatch `pallas` (default) or
/// `vesta` from `argv[1]` into a `fn dump<P: SWModelParameters>(name: &str)`.
///
/// A macro rather than a function because the two arms instantiate the caller's
/// generic at two different types.
#[macro_export]
macro_rules! curve_main {
    ($dump:ident) => {
        fn main() {
            match ::std::env::args().nth(1).as_deref().unwrap_or("pallas") {
                "pallas" => $dump::<$crate::ark_pallas::PallasParameters>("pallas"),
                "vesta" => $dump::<$crate::ark_vesta::VestaParameters>("vesta"),
                other => panic!("unknown curve {} (expected pallas|vesta)", other),
            }
        }
    };
}

/// The `a · b = c` circuit the AS/NARK fixtures are built over, padded out to
/// `num_inputs` public inputs and `num_constraints` constraints.
#[derive(Clone)]
pub struct DummyCircuit<F: ark_ff::Field> {
    pub a: Option<F>,
    pub b: Option<F>,
    pub num_inputs: usize,
    pub num_constraints: usize,
}

impl<F: ark_ff::Field> ark_relations::r1cs::ConstraintSynthesizer<F> for DummyCircuit<F> {
    fn generate_constraints(
        self,
        cs: ark_relations::r1cs::ConstraintSystemRef<F>,
    ) -> Result<(), ark_relations::r1cs::SynthesisError> {
        use ark_relations::lc;
        use ark_relations::r1cs::SynthesisError;

        let a = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        let b = cs.new_witness_variable(|| self.b.ok_or(SynthesisError::AssignmentMissing))?;
        let c = cs.new_input_variable(|| {
            let a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
            let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;
            Ok(a * b)
        })?;
        for _ in 0..(self.num_inputs - 1) {
            cs.new_input_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        }
        for _ in 0..(self.num_constraints - 1) {
            cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        }
        cs.enforce_constraint(lc!(), lc!(), lc!())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::{AffineCurve, ProjectiveCurve};
    use ark_ff::{One, UniformRand, Zero};
    use ark_pallas::{Affine, Fq as PFq, Fr as PFr, PallasParameters as P};

    fn point(k: &PFr) -> Affine {
        Affine::prime_subgroup_generator().mul(k.into_repr()).into_affine()
    }

    #[test]
    fn hex_roundtrips() {
        for b in [vec![], vec![0u8], vec![0xde, 0xad, 0xbe, 0xef], vec![0xff; 32]] {
            assert_eq!(from_hex(&hex(&b)), b);
        }
    }

    #[test]
    #[should_panic(expected = "odd-length hex")]
    fn from_hex_rejects_odd_length() {
        // The guard one hand-rolled copy had dropped; without it this panics
        // deep inside str slicing instead.
        from_hex("abc");
    }

    #[test]
    fn fe_hex_roundtrips() {
        let mut rng = ark_std::test_rng();
        for f in [PFr::zero(), PFr::one(), PFr::rand(&mut rng)] {
            assert_eq!(fe_from_hex::<PFr>(&fe_hex(&f)), f);
        }
    }

    #[test]
    fn fe_hex_is_canonical_le_32_bytes() {
        // One, so the LE encoding pins the byte order unambiguously.
        assert_eq!(
            fe_hex(&PFr::one()),
            "0100000000000000000000000000000000000000000000000000000000000000"
        );
    }

    #[test]
    fn point_json_roundtrips() {
        let mut rng = ark_std::test_rng();
        for k in [PFr::one(), PFr::from(12345u64), PFr::rand(&mut rng)] {
            let p = point(&k);
            assert_eq!(PointJson::from_affine(&p).to_affine::<P>(), p);
        }
    }

    #[test]
    fn point_json_maps_identity_to_all_zero() {
        // arkworks holds the identity as (0, 1, ∞); the fixture form is all-zero
        // in both coordinates. Getting this wrong emits y = 1.
        let id = Affine::zero();
        let j = PointJson::from_affine(&id);
        assert_eq!(j.x_le_hex, "00".repeat(32));
        assert_eq!(j.y_le_hex, "00".repeat(32));
        assert!(j.to_affine::<P>().is_zero());
    }

    #[test]
    fn point_json_serializes_as_the_fixture_object() {
        let j = PointJson {
            x_le_hex: "0a".into(),
            y_le_hex: "0b".into(),
        };
        assert_eq!(
            serde_json::to_string(&j).unwrap(),
            r#"{"x_le_hex":"0a","y_le_hex":"0b"}"#
        );
    }

    #[test]
    fn matrix_entry_serializes_as_a_two_element_array() {
        // The fixtures' sparse layout is [coeff_le_hex, var_index], not an
        // object — a struct here would silently change every matrix fixture.
        let m: MatrixJson = vec![vec![("0a".to_string(), 3)]];
        assert_eq!(serde_json::to_string(&m).unwrap(), r#"[[["0a",3]]]"#);
    }

    #[test]
    fn fe_list_roundtrips_elementwise() {
        let xs = vec![PFr::one(), PFr::from(7u64)];
        let back: Vec<PFr> = fe_list(&xs).iter().map(|h| fe_from_hex(h)).collect();
        assert_eq!(back, xs);
    }

    #[test]
    fn base_field_points_use_the_base_field_codec() {
        // Fq and Fr are distinct types on Pasta; a point's coordinates are Fq.
        let p = point(&PFr::from(3u64));
        let j = PointJson::from_affine(&p);
        assert_eq!(fe_from_hex::<PFq>(&j.x_le_hex), p.x);
        assert_eq!(fe_from_hex::<PFq>(&j.y_le_hex), p.y);
    }

    #[test]
    fn artifacts_dir_honors_the_env_override() {
        std::env::set_var("ACCUMULATION_ZORCH_ARTIFACTS", "/tmp/some-artifacts");
        assert_eq!(artifacts_dir("/repo"), std::path::PathBuf::from("/tmp/some-artifacts"));
        std::env::remove_var("ACCUMULATION_ZORCH_ARTIFACTS");
        assert_eq!(artifacts_dir("/repo"), std::path::PathBuf::from("/repo/artifacts"));
    }
}
