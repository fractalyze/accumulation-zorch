//! Slice-4 HP-AS prove (no-zk) fixtures for the jax port (zorch#303).
//!
//! Drives the crate's real `ASForHadamardProducts::prove` (no-zk) over two
//! Hadamard-product inputs and dumps the golden accumulator instance + `Proof`,
//! plus the replay inputs (the committer-key generators, the per-input `a`/`b`
//! vectors and their commitments). The no-zk HP prove draws no randomness; its
//! mu/nu challenges come from a fresh `S::new()` sponge (standalone — the AS
//! path forks it with `AS-FOR-HP-2020`, wired in slice 5).
//!
//! Run: `cargo run --example dump_hp > python/testdata/hp_fixtures.json`

use ark_ff::{BigInteger, PrimeField, Zero};
use ark_pallas::{Affine, Fr};
use ark_poly_commit::trivial_pc::PedersenCommitment;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use accumulation_zorch::hp_as::{ASForHadamardProducts, InputInstance, InputWitness};
use accumulation_zorch::{AccumulationScheme, Accumulator, Input, MakeZK};

type CF = ark_pallas::Fq;
type Sponge = ark_sponge::poseidon::PoseidonSponge<CF>;
type AS = ASForHadamardProducts<Affine, Sponge>;

const HP_VEC_LEN: usize = 4;

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
    // Committer key for length-HP_VEC_LEN vectors.
    let (ck, _vk, _dk) = AS::index(&(), &(), &HP_VEC_LEN).unwrap();

    // Two deterministic HP inputs; each instance commits its a/b/(a∘b) vectors.
    let make_input = |base: u64| -> Input<CF, Sponge, AS> {
        let a_vec: Vec<Fr> = (0..HP_VEC_LEN).map(|i| Fr::from(base + i as u64)).collect();
        let b_vec: Vec<Fr> = (0..HP_VEC_LEN).map(|i| Fr::from(base + 10 + i as u64)).collect();
        let ab = hadamard(&a_vec, &b_vec);
        let comm_1 = PedersenCommitment::<Affine>::commit(&ck, &a_vec, None);
        let comm_2 = PedersenCommitment::<Affine>::commit(&ck, &b_vec, None);
        let comm_3 = PedersenCommitment::<Affine>::commit(&ck, &ab, None);
        Input::<CF, Sponge, AS> {
            instance: InputInstance { comm_1, comm_2, comm_3 },
            witness: InputWitness { a_vec, b_vec, randomness: None },
        }
    };
    let inputs = vec![make_input(1), make_input(100)];
    let no_accs: Vec<Accumulator<CF, Sponge, AS>> = Vec::new();

    let (accumulator, proof) = AS::prove(
        &ck,
        Input::<CF, Sponge, AS>::map_to_refs(&inputs),
        Accumulator::<CF, Sponge, AS>::map_to_refs(&no_accs),
        MakeZK::Disabled,
        None,
    )
    .unwrap();

    // Committer-key generators (recovered from the key's uncompressed form).
    let generators = {
        let mut b = Vec::new();
        ck.serialize_uncompressed(&mut b).unwrap();
        let mut r = &b[..];
        let g = Vec::<Affine>::deserialize_uncompressed(&mut r).unwrap();
        g
    };
    let gens_json: Vec<String> = generators.iter().map(point_json).collect();

    // Replay inputs: a/b vectors + the three instance commitments per input.
    let inputs_json: Vec<String> = inputs
        .iter()
        .map(|inp| {
            format!(
                "{{\"a_vec\":{},\"b_vec\":{},\"comm_1\":{},\"comm_2\":{},\"comm_3\":{}}}",
                fr_list_json(&inp.witness.a_vec),
                fr_list_json(&inp.witness.b_vec),
                point_json(&inp.instance.comm_1),
                point_json(&inp.instance.comm_2),
                point_json(&inp.instance.comm_3),
            )
        })
        .collect();

    println!("{{");
    println!("  \"note\": \"HP-AS no-zk prove fixtures (zorch#303 slice 4)\",");
    println!("  \"hp_vec_len\": {},", HP_VEC_LEN);
    println!("  \"num_inputs\": {},", inputs.len());
    println!("  \"supported_num_elems\": {},", ck.supported_num_elems());
    println!("  \"generators\": [{}],", gens_json.join(","));
    println!("  \"inputs\": [{}],", inputs_json.join(","));
    println!("  \"acc_instance_hex\": \"{}\",", ser_hex(&accumulator.instance));
    println!("  \"proof_hex\": \"{}\"", ser_hex(&proof));
    println!("}}");
}
