"""Slice-6a byte-match: the zk NARK prove vs arkworks.

Replays the randomness the crate's real `R1CSNark::prove` (make_zk) sampled —
dumped by `examples/dump_nark_zk.rs` via an identically-seeded draw replay — and
asserts the ported zk prover reproduces the serialized `Proof` byte-for-byte.
Also anchors `hash_matrices` (the blake2b matrices hash that feeds gamma)
against the Rust-computed golden, and mutation-checks sensitivity.

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:nark_zk_test
"""

import json
from pathlib import Path
from typing import Any

from absl.testing import absltest

from accumulation_zorch import curve, nark, sponge

cv = curve.PALLAS

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
_NARK_ZK = _TESTDATA / "nark_zk_fixtures.json"
_SPONGE = _TESTDATA / "sponge_fixtures.json"


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _matrix(rows: Any) -> nark.Matrix:
    return [[(_fr(coeff), idx) for coeff, idx in row] for row in rows]


def _point(p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _load() -> Any:
    return json.loads(_NARK_ZK.read_text())


def _prove(d: Any) -> nark.NarkZkProof:
    a, b, c = _matrix(d["a"]), _matrix(d["b"]), _matrix(d["c"])
    return nark.prove_zk(
        cv, a, b, c,
        [_fr(h) for h in d["input"]], [_fr(h) for h in d["witness"]],
        [_point(g) for g in d["generators"]], _point(d["hiding"]), _params(),
        bytes.fromhex(d["nark_matrices_hash_hex"]),
        [_fr(h) for h in d["r"]],
        _fr(d["a_blinder"]), _fr(d["b_blinder"]), _fr(d["c_blinder"]),
        _fr(d["r_a_blinder"]), _fr(d["r_b_blinder"]), _fr(d["r_c_blinder"]),
        _fr(d["blinder_1"]), _fr(d["blinder_2"]),
    )


class NarkZkTest(absltest.TestCase):
    def test_hash_matrices_matches_arkworks(self) -> None:
        d = _load()
        got = nark.hash_matrices(cv, b"R1CS-NARK-2020", _matrix(d["a"]), _matrix(d["b"]), _matrix(d["c"]))
        self.assertEqual(got.hex(), d["nark_matrices_hash_hex"], (
            f"hash_matrices:\n got  {got.hex()}\n want {d['nark_matrices_hash_hex']}"
        ))
        print(f"  hash_matrices (blake2b) byte-matches arkworks ({got.hex()[:16]}…)")

    def test_nark_zk_proof_matches_arkworks(self) -> None:
        d = _load()
        proof = nark.serialize_zk_proof(cv, _prove(d))
        self.assertEqual(proof.hex(), d["proof_hex"], (
            f"zk NARK proof:\n got  {proof.hex()}\n want {d['proof_hex']}"
        ))
        print(f"  zk NARK proof byte-matches arkworks ({len(proof)} bytes)")

    def test_mutation_breaks_match(self) -> None:
        """Perturbing a witness-blinder (r) must change the proof bytes."""
        d = dict(_load())
        bad_r = list(d["r"])
        bad = bytearray(bytes.fromhex(bad_r[0]))
        bad[0] ^= 0x01
        bad_r[0] = bytes(bad).hex()
        d["r"] = bad_r
        proof = nark.serialize_zk_proof(cv, _prove(d))
        self.assertNotEqual(proof.hex(), _load()["proof_hex"], "mutation did not change the proof")
        print("  mutation check: a perturbed witness-blinder diverges from the golden")


if __name__ == "__main__":
    absltest.main()
