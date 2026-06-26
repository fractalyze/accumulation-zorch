//! Slice-6b HP-AS prove (zk) fixtures for the jax port (zorch#303).
//!
//! Drives the crate's real `ASForHadamardProducts::prove` with `make_zk = true`
//! over a single hiding HP input (so the prover adds the zero placeholder input,
//! exactly as the AS path's single HP input does) and dumps the golden zk
//! accumulator instance + witness + `Proof` (now with hiding commitments), the
//! replay inputs, and the prover's sampled hiding randomness (recovered by
//! replaying the `generate_prover_randomness` draw schedule on a fresh same-seed
//! `StdRng`: hiding `a`,`b` are `vec![rand; hp_vec_len]` — a single draw each —
//! then `rand_1`, `rand_2`, `rand_3`).
//!
//! Run: `cargo run --example dump_hp_zk > python/testdata/hp_zk_fixtures.json`

use ark_ff::{BigInteger, PrimeField, UniformRand};
use ark_pallas::{Affine, Fr};
use ark_poly_commit::trivial_pc::PedersenCommitment;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::{rngs::StdRng, SeedableRng};

use accumulation_zorch::hp_as::{ASForHadamardProducts, InputInstance, InputWitness, InputWitnessRandomness};
use accumulation_zorch::{AccumulationScheme, Accumulator, Input, MakeZK};

type CF = ark_pallas::Fq;
type Sponge = ark_sponge::poseidon::PoseidonSponge<CF>;
type AS = ASForHadamardProducts<Affine, Sponge>;

const HP_VEC_LEN: usize = 4;
const SEED: u64 = 11;

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn fr_hex(f: &Fr) -> String {
    hex(&f.into_repr().to_bytes_le())
}

fn ser_hex<T: CanonicalSerialize>(v: &T) -> String {
    let mut b = Vec::new();
    v.serialize(&mut b).unwrap();
    hex(&b)
}

fn fr_list_json(xs: &[Fr]) -> String {
    let v: Vec<String> = xs.iter().map(|f| format!("\"{}\"", fr_hex(f))).collect();
    format!("[{}]", v.join(","))
}

fn point_json(p: &Affine) -> String {
    use ark_ff::Zero;
    let (x, y) = if p.is_zero() {
        (hex(&[0u8; 32]), hex(&[0u8; 32]))
    } else {
        (hex(&p.x.into_repr().to_bytes_le()), hex(&p.y.into_repr().to_bytes_le()))
    };
    format!("{{\"x_le_hex\":\"{}\",\"y_le_hex\":\"{}\"}}", x, y)
}

fn hadamard(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
    a.iter().zip(b).map(|(x, y)| *x * y).collect()
}

fn main() {
    let (ck, _vk, _dk) = AS::index(&(), &(), &HP_VEC_LEN).unwrap();

    // One hiding HP input: instance commits a/b/(a∘b) with the input randomness.
    let a_vec: Vec<Fr> = (0..HP_VEC_LEN).map(|i| Fr::from(1 + i as u64)).collect();
    let b_vec: Vec<Fr> = (0..HP_VEC_LEN).map(|i| Fr::from(11 + i as u64)).collect();
    let (in_rand_1, in_rand_2, in_rand_3) = (Fr::from(100u64), Fr::from(200u64), Fr::from(300u64));
    let comm_1 = PedersenCommitment::<Affine>::commit(&ck, &a_vec, Some(in_rand_1));
    let comm_2 = PedersenCommitment::<Affine>::commit(&ck, &b_vec, Some(in_rand_2));
    let comm_3 = PedersenCommitment::<Affine>::commit(&ck, &hadamard(&a_vec, &b_vec), Some(in_rand_3));
    let input = Input::<CF, Sponge, AS> {
        instance: InputInstance { comm_1, comm_2, comm_3 },
        witness: InputWitness {
            a_vec: a_vec.clone(),
            b_vec: b_vec.clone(),
            randomness: Some(InputWitnessRandomness { rand_1: in_rand_1, rand_2: in_rand_2, rand_3: in_rand_3 }),
        },
    };
    let inputs = vec![input];
    let no_accs: Vec<Accumulator<CF, Sponge, AS>> = Vec::new();

    let mut rng = StdRng::seed_from_u64(SEED);
    let (accumulator, proof) = AS::prove(
        &ck,
        Input::<CF, Sponge, AS>::map_to_refs(&inputs),
        Accumulator::<CF, Sponge, AS>::map_to_refs(&no_accs),
        MakeZK::Enabled(&mut rng),
        None,
    )
    .unwrap();

    // Committer-key generators + hiding generator.
    let (generators, hiding) = {
        let mut b = Vec::new();
        ck.serialize_uncompressed(&mut b).unwrap();
        let mut r = &b[..];
        let g = Vec::<Affine>::deserialize_uncompressed(&mut r).unwrap();
        let h = Affine::deserialize_uncompressed(&mut r).unwrap();
        (g, h)
    };
    let gens_json: Vec<String> = generators.iter().map(point_json).collect();

    // Replay the make_zk generate_prover_randomness draw schedule.
    let mut rep = StdRng::seed_from_u64(SEED);
    let hiding_a = Fr::rand(&mut rep); // vec![rand; hp_vec_len] — one draw, cloned
    let hiding_b = Fr::rand(&mut rep);
    let hiding_rand_1 = Fr::rand(&mut rep);
    let hiding_rand_2 = Fr::rand(&mut rep);
    let hiding_rand_3 = Fr::rand(&mut rep);

    println!("{{");
    println!("  \"note\": \"HP-AS zk prove fixtures (zorch#303 slice 6b)\",");
    println!("  \"hp_vec_len\": {},", HP_VEC_LEN);
    println!("  \"supported_num_elems\": {},", ck.supported_num_elems());
    println!("  \"generators\": [{}],", gens_json.join(","));
    println!("  \"hiding\": {},", point_json(&hiding));
    println!("  \"a_vec\": {},", fr_list_json(&a_vec));
    println!("  \"b_vec\": {},", fr_list_json(&b_vec));
    println!("  \"in_rand_1\": \"{}\",", fr_hex(&in_rand_1));
    println!("  \"in_rand_2\": \"{}\",", fr_hex(&in_rand_2));
    println!("  \"in_rand_3\": \"{}\",", fr_hex(&in_rand_3));
    println!("  \"comm_1\": {},", point_json(&comm_1));
    println!("  \"comm_2\": {},", point_json(&comm_2));
    println!("  \"comm_3\": {},", point_json(&comm_3));
    println!("  \"hiding_a\": \"{}\",", fr_hex(&hiding_a));
    println!("  \"hiding_b\": \"{}\",", fr_hex(&hiding_b));
    println!("  \"hiding_rand_1\": \"{}\",", fr_hex(&hiding_rand_1));
    println!("  \"hiding_rand_2\": \"{}\",", fr_hex(&hiding_rand_2));
    println!("  \"hiding_rand_3\": \"{}\",", fr_hex(&hiding_rand_3));
    println!("  \"acc_instance_hex\": \"{}\",", ser_hex(&accumulator.instance));
    println!("  \"acc_witness_hex\": \"{}\",", ser_hex(&accumulator.witness));
    println!("  \"proof_hex\": \"{}\"", ser_hex(&proof));
    println!("}}");
}
