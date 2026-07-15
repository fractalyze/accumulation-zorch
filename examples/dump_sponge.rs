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
use ark_sponge::poseidon::PoseidonSponge;
use ark_sponge::{Absorbable, CryptographicSponge, FieldElementSize};
use ark_std::UniformRand;
use rand_chacha::ChaChaRng;
use rand_core::SeedableRng;
use serde::Serialize;

use fixture_json::{curve_main, fe_hex, fe_list};

const FULL_ROUNDS: usize = 8;
const PARTIAL_ROUNDS: usize = 31;
const WIDTH: usize = 3;
const ALPHA: u32 = 17;
const RATE: usize = 2;
const CAPACITY: usize = 1;
const MDS: [[u8; 3]; 3] = [[1, 0, 1], [1, 1, 0], [0, 1, 1]];

/// `ConstraintF<G>` (the sponge / constraint field) — `pub(crate)` upstream, so
/// re-derived. For the Pasta curves the base field is already prime, so this is
/// just the base field.
type CF<P> = <<P as ModelParameters>::BaseField as Field>::BasePrimeField;

/// The `PoseidonSponge::new` config the Python side must mirror.
#[derive(Serialize)]
struct Config {
    full_rounds: usize,
    partial_rounds: usize,
    alpha: u32,
    width: usize,
    rate: usize,
    capacity: usize,
    mds: [[u8; 3]; 3],
}

/// One step of a schedule. Internally tagged, so an absorb renders as
/// `{"op":"absorb","vals":[..]}` and a squeeze as `{"op":"squeeze","n":..,"out":[..]}`.
#[derive(Serialize)]
#[serde(tag = "op", rename_all = "lowercase")]
enum OpJson {
    Absorb { vals: Vec<u64> },
    Squeeze { n: usize, out: Vec<String> },
}

#[derive(Serialize)]
struct Schedule {
    name: String,
    ops: Vec<OpJson>,
}

#[derive(Serialize)]
struct Nonnative {
    name: String,
    absorb: Vec<u64>,
    k: usize,
    challenges: Vec<String>,
}

#[derive(Serialize)]
struct SpongeFixture {
    note: String,
    curve: String,
    config: Config,
    ark_le_hex: Vec<String>,
    schedules: Vec<Schedule>,
    nonnative_squeeze: Vec<Nonnative>,
}

enum Op {
    Absorb(Vec<u64>),
    Squeeze(usize),
}

/// Runs a schedule on a fresh real sponge over the sponge field `F`, recording
/// the squeezed outputs.
fn run_schedule<F>(name: &str, ops: &[Op]) -> Schedule
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
                op_jsons.push(OpJson::Absorb { vals: vals.clone() });
            }
            Op::Squeeze(n) => {
                let out = sponge.squeeze_field_elements(*n);
                op_jsons.push(OpJson::Squeeze {
                    n: *n,
                    out: fe_list(&out),
                });
            }
        }
    }
    Schedule {
        name: name.to_string(),
        ops: op_jsons,
    }
}

fn dump<P>(curve: &str)
where
    P: SWModelParameters,
    CF<P>: PrimeField + Absorbable<CF<P>>,
{
    // Regenerate the 117 ARK (row-major over 39 rounds × width 3), identical to
    // `PoseidonSponge::<CF<P>>::new()`'s `ark` field.
    let mut ark_rng = ChaChaRng::seed_from_u64(123456789u64);
    let mut ark_le_hex = Vec::new();
    for _ in 0..(FULL_ROUNDS + PARTIAL_ROUNDS) {
        for _ in 0..WIDTH {
            ark_le_hex.push(fe_hex(&<CF<P>>::rand(&mut ark_rng)));
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
    let nonnative = |name: &str, absorb_vals: &[u64], k: usize| -> Nonnative {
        let mut sp = PoseidonSponge::<CF<P>>::new();
        for &v in absorb_vals {
            sp.absorb(&<CF<P>>::from(v));
        }
        let sizes = vec![FieldElementSize::Truncated(128); k];
        let out: Vec<P::ScalarField> =
            sp.squeeze_nonnative_field_elements_with_sizes::<P::ScalarField>(&sizes);
        Nonnative {
            name: name.to_string(),
            absorb: absorb_vals.to_vec(),
            k,
            challenges: fe_list(&out),
        }
    };
    let nonnative_squeeze = vec![
        nonnative("one_challenge", &[1, 2, 3], 1),
        nonnative("two_challenges", &[7], 2),
        // k=4 -> 512 bits -> 3 squeezed CF, bits concatenated across elements.
        nonnative("four_challenges_cross_element", &[10, 20, 30], 4),
    ];

    let fixture = SpongeFixture {
        note: format!("ark-sponge PoseidonSponge<{} base field> default-config fixtures", curve),
        curve: curve.to_string(),
        config: Config {
            full_rounds: FULL_ROUNDS,
            partial_rounds: PARTIAL_ROUNDS,
            alpha: ALPHA,
            width: WIDTH,
            rate: RATE,
            capacity: CAPACITY,
            mds: MDS,
        },
        ark_le_hex,
        schedules,
        nonnative_squeeze,
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}

curve_main!(dump);
