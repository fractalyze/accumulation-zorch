//! HP-AS prove (zk) fixtures for the frx port.
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

use ark_ff::UniformRand;
use ark_pallas::{Affine, Fr};
use ark_poly_commit::trivial_pc::PedersenCommitment;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;

use ark_accumulation::hp_as::{ASForHadamardProducts, InputInstance, InputWitness, InputWitnessRandomness};
use ark_accumulation::{AccumulationScheme, Accumulator, Input, MakeZK};
use fixture_json::{fe_hex, fe_list, point_list, ser_hex, PointJson};

type CF = ark_pallas::Fq;
type Sponge = ark_sponge::poseidon::PoseidonSponge<CF>;
type AS = ASForHadamardProducts<Affine, Sponge>;

const HP_VEC_LEN: usize = 4;
const SEED: u64 = 11;

/// The fixture schema. Field order is the emitted key order.
#[derive(Serialize)]
struct HpZkFixture {
    note: String,
    hp_vec_len: usize,
    supported_num_elems: usize,
    generators: Vec<PointJson>,
    hiding: PointJson,
    a_vec: Vec<String>,
    b_vec: Vec<String>,
    in_rand_1: String,
    in_rand_2: String,
    in_rand_3: String,
    comm_1: PointJson,
    comm_2: PointJson,
    comm_3: PointJson,
    hiding_a: String,
    hiding_b: String,
    hiding_rand_1: String,
    hiding_rand_2: String,
    hiding_rand_3: String,
    acc_instance_hex: String,
    acc_witness_hex: String,
    proof_hex: String,
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
    // Replay the make_zk generate_prover_randomness draw schedule.
    let mut rep = StdRng::seed_from_u64(SEED);
    let hiding_a = Fr::rand(&mut rep); // vec![rand; hp_vec_len] — one draw, cloned
    let hiding_b = Fr::rand(&mut rep);
    let hiding_rand_1 = Fr::rand(&mut rep);
    let hiding_rand_2 = Fr::rand(&mut rep);
    let hiding_rand_3 = Fr::rand(&mut rep);

    let fixture = HpZkFixture {
        note: "HP-AS zk prove fixtures".to_string(),
        hp_vec_len: HP_VEC_LEN,
        supported_num_elems: ck.supported_num_elems(),
        generators: point_list(&generators),
        hiding: PointJson::from_affine(&hiding),
        a_vec: fe_list(&a_vec),
        b_vec: fe_list(&b_vec),
        in_rand_1: fe_hex(&in_rand_1),
        in_rand_2: fe_hex(&in_rand_2),
        in_rand_3: fe_hex(&in_rand_3),
        comm_1: PointJson::from_affine(&comm_1),
        comm_2: PointJson::from_affine(&comm_2),
        comm_3: PointJson::from_affine(&comm_3),
        hiding_a: fe_hex(&hiding_a),
        hiding_b: fe_hex(&hiding_b),
        hiding_rand_1: fe_hex(&hiding_rand_1),
        hiding_rand_2: fe_hex(&hiding_rand_2),
        hiding_rand_3: fe_hex(&hiding_rand_3),
        acc_instance_hex: ser_hex(&accumulator.instance),
        acc_witness_hex: ser_hex(&accumulator.witness),
        proof_hex: ser_hex(&proof),
    };
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}
