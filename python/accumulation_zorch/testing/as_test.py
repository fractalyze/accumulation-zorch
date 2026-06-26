"""Slice-5 byte-match: the no-zk R1CS-NARK-AS prove vs arkworks — end-to-end.

The acceptance criterion for zorch#303. Replays the inputs dumped from the
crate's real `ASForR1CSNark::prove` (no-zk) — `examples/dump_as.rs`, itself
seeded identically to `src/oracle.rs`'s `prove_byte_identical_to_arkworks_no_zk`
target — and asserts the ported prover reproduces the serialized
`(acc.instance ‖ acc.witness ‖ proof)` byte-for-byte for seeds {0, 42},
num_inputs=5, num_constraints=10.

The instance / witness / proof are checked separately (localizes a divergence to
one component) and then concatenated for the full match. A mutation check
confirms the byte-match is sensitive (perturbing a witness element breaks it).

Run (from the repo's `python/` dir, in the accumulation-zorch venv):

    JAX_PLATFORMS=cpu PYTHONPATH=.:<pasta-zorch>/zorch \
      python accumulation_zorch/testing/as_test.py
"""

import json
from pathlib import Path
from typing import Any

from accumulation_zorch import curve, r1cs_nark_as, sponge

cv = curve.PALLAS

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
_AS = _TESTDATA / "as_fixtures.json"
_SPONGE = _TESTDATA / "sponge_fixtures.json"


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _matrix(rows: Any) -> Any:
    """Sparse `Matrix<Fr>`: each row a list of `(coeff_int, var_index)`."""
    return [[(_fr(coeff), idx) for coeff, idx in row] for row in rows]


def _point(p: Any) -> Any:
    return cv.g1((
        int.from_bytes(bytes.fromhex(p["x_le_hex"]), "little"),
        int.from_bytes(bytes.fromhex(p["y_le_hex"]), "little"),
    ))


def _load() -> Any:
    d = json.loads(_AS.read_text())
    a, b, c = _matrix(d["a"]), _matrix(d["b"]), _matrix(d["c"])
    generators = [_point(g) for g in d["generators"]]
    return d, a, b, c, generators


def _prove(d: Any, a: Any, b: Any, c: Any, generators: Any, seed_entry: Any) -> Any:
    r1cs_input = [_fr(h) for h in seed_entry["r1cs_input"]]
    blinded_witness = [_fr(h) for h in seed_entry["blinded_witness"]]
    return r1cs_nark_as.prove_no_zk(
        cv, a, b, c, r1cs_input, blinded_witness, generators, d["supported_num_elems"], _params()
    )


def test_as_no_zk_prove_matches_arkworks() -> None:
    d, a, b, c, generators = _load()
    for seed_entry in d["seeds"]:
        seed = seed_entry["seed"]
        acc_instance, acc_witness, proof = _prove(d, a, b, c, generators, seed_entry)

        assert acc_instance.hex() == seed_entry["acc_instance_hex"], (
            f"seed {seed} acc.instance:\n got  {acc_instance.hex()}\n want {seed_entry['acc_instance_hex']}"
        )
        assert acc_witness.hex() == seed_entry["acc_witness_hex"], (
            f"seed {seed} acc.witness:\n got  {acc_witness.hex()}\n want {seed_entry['acc_witness_hex']}"
        )
        assert proof.hex() == seed_entry["proof_hex"], (
            f"seed {seed} proof:\n got  {proof.hex()}\n want {seed_entry['proof_hex']}"
        )

        full = (acc_instance + acc_witness + proof).hex()
        want_full = seed_entry["acc_instance_hex"] + seed_entry["acc_witness_hex"] + seed_entry["proof_hex"]
        assert full == want_full
        print(f"  seed {seed}: (acc.instance {len(acc_instance)}B ‖ acc.witness "
              f"{len(acc_witness)}B ‖ proof {len(proof)}B) byte-matches arkworks")


def test_mutation_breaks_match() -> None:
    """Perturbing a blinded-witness element must break the byte-match."""
    d, a, b, c, generators = _load()
    seed_entry = dict(d["seeds"][0])
    bad_w = list(seed_entry["blinded_witness"])
    bad = bytearray(bytes.fromhex(bad_w[0]))
    bad[0] ^= 0x01
    bad_w[0] = bytes(bad).hex()
    seed_entry["blinded_witness"] = bad_w

    acc_instance, acc_witness, proof = _prove(d, a, b, c, generators, seed_entry)
    full = (acc_instance + acc_witness + proof).hex()
    golden = d["seeds"][0]
    want = golden["acc_instance_hex"] + golden["acc_witness_hex"] + golden["proof_hex"]
    assert full != want, "mutation did not change the output — byte-match is not sensitive"
    print("  mutation check: a perturbed witness diverges from the golden bytes")


def main() -> None:
    print("slice-5 R1CS-NARK-AS no-zk prove end-to-end byte-match:")
    test_as_no_zk_prove_matches_arkworks()
    test_mutation_breaks_match()
    print("ALL SLICE-5 AS CHECKS PASSED")


if __name__ == "__main__":
    main()
