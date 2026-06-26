"""Slice-6c byte-match: the zk R1CS-NARK-AS prove vs arkworks — end-to-end.

The zk acceptance criterion for the jax prover port. Replays the full cross-prover
randomness (NARK + AS + HP) dumped from the crate's real `ASForR1CSNark::prove`
(make_zk) — `examples/dump_as_zk.rs`, seeded identically to `src/oracle.rs`'s
`prove_byte_identical_to_arkworks_zk` target — and asserts the ported zk prover
reproduces the serialized `(acc.instance ‖ acc.witness ‖ proof)` byte-for-byte
for seeds {0, 42}, num_inputs=5, num_constraints=10.

Run (from the repo's `python/` dir, in the accumulation-zorch venv):

    JAX_PLATFORMS=cpu PYTHONPATH=.:<pasta-zorch>/zorch \
      python accumulation_zorch/testing/as_zk_test.py
"""

import json
from pathlib import Path
from typing import Any

from accumulation_zorch import curve, r1cs_nark_as, sponge

cv = curve.PALLAS

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
_AS_ZK = _TESTDATA / "as_zk_fixtures.json"
_SPONGE = _TESTDATA / "sponge_fixtures.json"


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _matrix(rows: Any) -> r1cs_nark_as.nark.Matrix:
    return [[(_fr(coeff), idx) for coeff, idx in row] for row in rows]


def _point(p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _load() -> Any:
    d = json.loads(_AS_ZK.read_text())
    return d, _matrix(d["a"]), _matrix(d["b"]), _matrix(d["c"]), [_point(g) for g in d["generators"]]


def _prove(d: Any, a: Any, b: Any, c: Any, generators: Any, s: Any) -> tuple[bytes, bytes, bytes]:
    return r1cs_nark_as.prove_zk(
        cv, a, b, c, [_fr(h) for h in s["r1cs_input"]], [_fr(h) for h in s["witness"]],
        generators, _point(d["hiding"]), _params(),
        bytes.fromhex(d["nark_matrices_hash_hex"]), bytes.fromhex(d["as_matrices_hash_hex"]),
        d["supported_num_elems"],
        [_fr(h) for h in s["r"]],
        (_fr(s["a_blinder"]), _fr(s["b_blinder"]), _fr(s["c_blinder"]),
         _fr(s["r_a_blinder"]), _fr(s["r_b_blinder"]), _fr(s["r_c_blinder"]),
         _fr(s["blinder_1"]), _fr(s["blinder_2"])),
        _fr(s["as_r1cs_r_input"]), _fr(s["as_r1cs_r_witness"]),
        (_fr(s["as_rand_1"]), _fr(s["as_rand_2"]), _fr(s["as_rand_3"])),
        _fr(s["hp_hiding_a"]), _fr(s["hp_hiding_b"]),
        (_fr(s["hp_rand_1"]), _fr(s["hp_rand_2"]), _fr(s["hp_rand_3"])),
    )


def test_as_zk_prove_matches_arkworks() -> None:
    d, a, b, c, generators = _load()
    for s in d["seeds"]:
        seed = s["seed"]
        acc_instance, acc_witness, proof = _prove(d, a, b, c, generators, s)
        assert acc_instance.hex() == s["acc_instance_hex"], (
            f"seed {seed} acc.instance:\n got  {acc_instance.hex()}\n want {s['acc_instance_hex']}"
        )
        assert acc_witness.hex() == s["acc_witness_hex"], (
            f"seed {seed} acc.witness:\n got  {acc_witness.hex()}\n want {s['acc_witness_hex']}"
        )
        assert proof.hex() == s["proof_hex"], (
            f"seed {seed} proof:\n got  {proof.hex()}\n want {s['proof_hex']}"
        )
        print(f"  seed {seed}: (acc.instance {len(acc_instance)}B ‖ acc.witness "
              f"{len(acc_witness)}B ‖ proof {len(proof)}B) byte-matches arkworks")


def test_mutation_breaks_match() -> None:
    """Perturbing the NARK witness-blinder r must change the output."""
    d, a, b, c, generators = _load()
    s = dict(d["seeds"][0])
    bad_r = list(s["r"])
    bad = bytearray(bytes.fromhex(bad_r[0]))
    bad[0] ^= 0x01
    bad_r[0] = bytes(bad).hex()
    s["r"] = bad_r
    acc_instance, acc_witness, proof = _prove(d, a, b, c, generators, s)
    full = (acc_instance + acc_witness + proof).hex()
    g = d["seeds"][0]
    assert full != g["acc_instance_hex"] + g["acc_witness_hex"] + g["proof_hex"], "mutation no-op"
    print("  mutation check: a perturbed NARK blinder diverges from the golden bytes")


def main() -> None:
    print("slice-6c zk R1CS-NARK-AS prove end-to-end byte-match:")
    test_as_zk_prove_matches_arkworks()
    test_mutation_breaks_match()
    print("ALL SLICE-6c AS-ZK CHECKS PASSED")


if __name__ == "__main__":
    main()
