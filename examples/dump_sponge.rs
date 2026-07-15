//! NARK/AS Poseidon Fiat-Shamir sponge fixtures for the frx port,
//! over either Pasta cycle curve (Pallas or Vesta).
//!
//! Emits the default ark-sponge `PoseidonSponge<CF>` round constants (the 117
//! ARK, regenerated exactly as `new()` does — from
//! `ChaChaRng::seed_from_u64(123456789)` over the curve's constraint field
//! `CF = P::BaseField`), plus the squeezed field elements of several
//! absorb/squeeze schedules driven through the *real* sponge. The Python side
//! builds `PoseidonParams`+`Poseidon`+`DuplexSponge` over the same ARK and must
//! reproduce every squeezed element byte-for-byte — which also validates the
//! regenerated ARK end-to-end (a wrong constant diverges).
//!
//! The sponge field is the curve's base field (`ark_pallas::Fq` for Pallas,
//! `ark_vesta::Fq = ark_pallas::Fr` for Vesta — the Pasta cycle swaps base and
//! scalar); the nonnative truncated squeeze targets the curve's scalar field,
//! where the NARK gamma challenge lives. The constants differ per curve (the
//! same RNG bytes reduce mod a different prime), so each curve needs its own
//! fixture.
//!
//! Config (ark-sponge `PoseidonSponge::new`): full=8, partial=31, alpha=17,
//! mds = [[1,0,1],[1,1,0],[0,1,1]], rate=2, capacity=1, width=3.
//!
//! Generic over the curve (no per-curve copy) — the curve is a CLI arg,
//! defaulting to Pallas:
//!
//!   cargo run --example dump_sponge -- pallas > python/testdata/sponge_fixtures.json
//!   cargo run --example dump_sponge -- vesta  > python/testdata/sponge_vesta_fixtures.json

use ark_ec::models::ModelParameters;
use ark_ec::SWModelParameters;
use ark_ff::{Field, PrimeField};
use ark_serialize::CanonicalSerialize;
use ark_sponge::poseidon::PoseidonSponge;
use ark_sponge::{Absorbable, CryptographicSponge, FieldElementSize};
use ark_std::UniformRand;
use rand_chacha::ChaChaRng;
use rand_core::SeedableRng;

const FULL_ROUNDS: usize = 8;
const PARTIAL_ROUNDS: usize = 31;
const WIDTH: usize = 3;

/// `ConstraintF<G>` (the sponge / constraint field) — `pub(crate)` upstream, so
/// re-derived. For the Pasta curves the base field is already prime, so this is
/// just the base field.
type CF<P> = <<P as ModelParameters>::BaseField as Field>::BasePrimeField;

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

enum Op {
    Absorb(Vec<u64>),
    Squeeze(usize),
}

/// Runs a schedule on a fresh real sponge over the sponge field `F` and renders
/// it (with squeezed outputs) as a JSON object.
fn run_schedule<F>(name: &str, ops: &[Op]) -> String
where
    F: PrimeField + Absorbable<F>,
{
    let mut sponge = PoseidonSponge::<F>::new();
    let mut op_jsons = Vec::new();
    for op in ops {
        match op {
            Op::Absorb(vals) => {
                for &v in vals {
                    sponge.absorb(&F::from(v));
                }
                let vs: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
                op_jsons.push(format!("{{\"op\":\"absorb\",\"vals\":[{}]}}", vs.join(",")));
            }
            Op::Squeeze(n) => {
                let out = sponge.squeeze_field_elements(*n);
                let outs: Vec<String> = out.iter().map(|f| format!("\"{}\"", fe_hex(f))).collect();
                op_jsons.push(format!(
                    "{{\"op\":\"squeeze\",\"n\":{},\"out\":[{}]}}",
                    n,
                    outs.join(",")
                ));
            }
        }
    }
    format!("{{\"name\":\"{}\",\"ops\":[{}]}}", name, op_jsons.join(","))
}

fn dump<P>(curve: &str)
where
    P: SWModelParameters,
    CF<P>: PrimeField + Absorbable<CF<P>>,
{
    // Regenerate the 117 ARK (row-major over 39 rounds × width 3), identical to
    // `PoseidonSponge::<CF<P>>::new()`'s `ark` field.
    let mut ark_rng = ChaChaRng::seed_from_u64(123456789u64);
    let mut ark_hex = Vec::new();
    for _ in 0..(FULL_ROUNDS + PARTIAL_ROUNDS) {
        for _ in 0..WIDTH {
            ark_hex.push(format!("\"{}\"", fe_hex(&<CF<P>>::rand(&mut ark_rng))));
        }
    }

    use Op::{Absorb as A, Squeeze as S};
    let schedules = vec![
        run_schedule::<CF<P>>("absorb1_squeeze1", &[A(vec![1]), S(1)]),
        run_schedule::<CF<P>>("absorb2_squeeze1", &[A(vec![1, 2]), S(1)]),
        run_schedule::<CF<P>>("absorb3_squeeze1", &[A(vec![1, 2, 3]), S(1)]),
        run_schedule::<CF<P>>("absorb5_squeeze3", &[A(vec![1, 2, 3, 4, 5]), S(3)]),
        run_schedule::<CF<P>>("absorb1_squeeze2_squeeze2", &[A(vec![7]), S(2), S(2)]),
        run_schedule::<CF<P>>(
            "interleaved",
            &[A(vec![10]), S(1), A(vec![20, 30]), S(2), A(vec![40, 50, 60]), S(4)],
        ),
    ];

    // Nonnative truncated-128 squeeze → ScalarField (the gamma-challenge
    // primitive). Absorb a few CF, then
    // `squeeze_nonnative_field_elements_with_sizes::<ScalarField>([Trunc(128);k])`.
    let nonnative = |name: &str, absorb_vals: &[u64], k: usize| -> String {
        let mut sp = PoseidonSponge::<CF<P>>::new();
        for &v in absorb_vals {
            sp.absorb(&<CF<P>>::from(v));
        }
        let sizes = vec![FieldElementSize::Truncated(128); k];
        let out: Vec<P::ScalarField> =
            sp.squeeze_nonnative_field_elements_with_sizes::<P::ScalarField>(&sizes);
        let absorbs: Vec<String> = absorb_vals.iter().map(|v| v.to_string()).collect();
        let outs: Vec<String> = out.iter().map(|f| format!("\"{}\"", fe_hex(f))).collect();
        format!(
            "{{\"name\":\"{}\",\"absorb\":[{}],\"k\":{},\"challenges\":[{}]}}",
            name,
            absorbs.join(","),
            k,
            outs.join(",")
        )
    };
    let nonnative_fixtures = vec![
        nonnative("one_challenge", &[1, 2, 3], 1),
        nonnative("two_challenges", &[7], 2),
        // k=4 -> 512 bits -> 3 squeezed CF, bits concatenated across elements.
        nonnative("four_challenges_cross_element", &[10, 20, 30], 4),
    ];

    println!("{{");
    println!(
        "  \"note\": \"ark-sponge PoseidonSponge<{} base field> default-config fixtures\",",
        curve
    );
    println!("  \"curve\": \"{}\",", curve);
    println!("  \"config\": {{\"full_rounds\":{},\"partial_rounds\":{},\"alpha\":17,\"width\":{},\"rate\":2,\"capacity\":1,\"mds\":[[1,0,1],[1,1,0],[0,1,1]]}},", FULL_ROUNDS, PARTIAL_ROUNDS, WIDTH);
    println!("  \"ark_le_hex\": [{}],", ark_hex.join(","));
    println!("  \"schedules\": [{}],", schedules.join(","));
    println!("  \"nonnative_squeeze\": [{}]", nonnative_fixtures.join(","));
    println!("}}");
}

fn main() {
    match std::env::args().nth(1).as_deref().unwrap_or("pallas") {
        "pallas" => dump::<ark_pallas::PallasParameters>("pallas"),
        "vesta" => dump::<ark_vesta::VestaParameters>("vesta"),
        other => panic!("unknown curve {} (expected pallas|vesta)", other),
    }
}
