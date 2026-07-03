"""Slice-4 byte-match: the no-zk HP-AS prove vs the real arkworks prover.

Replays two Hadamard-product inputs dumped from the crate's real
`ASForHadamardProducts::prove` (no-zk) — committer-key generators, per-input
`a`/`b` vectors and instance commitments (`cargo run --example dump_hp`) — and
asserts the ported prove reproduces both the `Proof` (product-polynomial
commitments) and the combined accumulator instance byte-for-byte.

The 117 Poseidon ARK constants come from the slice-2 sponge fixtures.

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:hp_test
"""

import json
from pathlib import Path
from typing import Any

from absl.testing import absltest

from accumulation_zorch import curve, hp_as, sponge

cv = curve.PALLAS

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
_HP = _TESTDATA / "hp_fixtures.json"
_SPONGE = _TESTDATA / "sponge_fixtures.json"


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _point(p: Any) -> Any:
    return cv.g1((
        int.from_bytes(bytes.fromhex(p["x_le_hex"]), "little"),
        int.from_bytes(bytes.fromhex(p["y_le_hex"]), "little"),
    ))


def _load() -> Any:
    d = json.loads(_HP.read_text())
    generators = [_point(g) for g in d["generators"]]
    instances, a_vecs, b_vecs = [], [], []
    for inp in d["inputs"]:
        instances.append((_point(inp["comm_1"]), _point(inp["comm_2"]), _point(inp["comm_3"])))
        a_vecs.append([int.from_bytes(bytes.fromhex(h), "little") for h in inp["a_vec"]])
        b_vecs.append([int.from_bytes(bytes.fromhex(h), "little") for h in inp["b_vec"]])
    return d, generators, instances, a_vecs, b_vecs


class HpTest(absltest.TestCase):
    def test_hp_no_zk_prove_matches_arkworks(self) -> None:
        d, generators, instances, a_vecs, b_vecs = _load()
        instance, _witness, low, high = hp_as.prove_no_zk(
            cv, generators, instances, a_vecs, b_vecs, d["supported_num_elems"], _params()
        )

        proof = hp_as.serialize_proof(cv, low, high)
        self.assertEqual(proof.hex(), d["proof_hex"], f"HP proof:\n got  {proof.hex()}\n want {d['proof_hex']}")
        print(f"  HP no-zk proof byte-matches arkworks ({len(proof)} bytes, "
              f"low={len(low)} high={len(high)})")

        acc = hp_as.serialize_instance(cv, instance)
        self.assertEqual(acc.hex(), d["acc_instance_hex"], (
            f"HP accumulator instance:\n got  {acc.hex()}\n want {d['acc_instance_hex']}"
        ))
        print(f"  HP combined accumulator instance byte-matches arkworks ({len(acc)} bytes)")


if __name__ == "__main__":
    absltest.main()
