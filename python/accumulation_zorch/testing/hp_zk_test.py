"""Slice-6b byte-match: the zk HP-AS prove vs arkworks.

Replays a single hiding HP input through the ported zk prover (which adds the
zero placeholder input, as the make_zk path does) and asserts it reproduces the
crate's serialized accumulator instance + witness + `Proof` (with hiding
commitments) byte-for-byte. The prover's sampled hiding randomness is replayed
from `examples/dump_hp_zk.rs`.

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:hp_zk_test
"""

import json
from pathlib import Path
from typing import Any

from absl.testing import absltest

from accumulation_zorch import curve, hp_as, sponge

cv = curve.PALLAS

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
_HP_ZK = _TESTDATA / "hp_zk_fixtures.json"
_SPONGE = _TESTDATA / "sponge_fixtures.json"


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _point(p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _prove(d: Any) -> Any:
    generators = [_point(g) for g in d["generators"]]
    instance = (_point(d["comm_1"]), _point(d["comm_2"]), _point(d["comm_3"]))
    a_vec = [_fr(h) for h in d["a_vec"]]
    b_vec = [_fr(h) for h in d["b_vec"]]
    input_rands: list[tuple[int, int, int] | None] = [
        (_fr(d["in_rand_1"]), _fr(d["in_rand_2"]), _fr(d["in_rand_3"]))]
    return hp_as.prove_zk(
        cv, generators, _point(d["hiding"]), [instance], [a_vec], [b_vec], input_rands,
        d["supported_num_elems"], _params(),
        _fr(d["hiding_a"]), _fr(d["hiding_b"]),
        _fr(d["hiding_rand_1"]), _fr(d["hiding_rand_2"]), _fr(d["hiding_rand_3"]),
    )


class HpZkTest(absltest.TestCase):
    def test_hp_zk_prove_matches_arkworks(self) -> None:
        d = json.loads(_HP_ZK.read_text())
        instance, witness, low, high, hiding_comms = _prove(d)

        acc_inst = hp_as.serialize_instance(cv, instance)
        self.assertEqual(acc_inst.hex(), d["acc_instance_hex"], (
            f"zk HP acc.instance:\n got  {acc_inst.hex()}\n want {d['acc_instance_hex']}"
        ))
        print(f"  zk HP accumulator instance byte-matches arkworks ({len(acc_inst)} bytes)")

        acc_wit = hp_as.serialize_witness_zk(cv, witness)
        self.assertEqual(acc_wit.hex(), d["acc_witness_hex"], (
            f"zk HP acc.witness:\n got  {acc_wit.hex()}\n want {d['acc_witness_hex']}"
        ))
        print(f"  zk HP accumulator witness byte-matches arkworks ({len(acc_wit)} bytes)")

        proof = hp_as.serialize_proof_zk(cv, low, high, hiding_comms)
        self.assertEqual(proof.hex(), d["proof_hex"], (
            f"zk HP proof:\n got  {proof.hex()}\n want {d['proof_hex']}"
        ))
        print(f"  zk HP proof byte-matches arkworks ({len(proof)} bytes, low={len(low)} high={len(high)})")

    def test_mutation_breaks_match(self) -> None:
        """Perturbing the prover's hiding-a randomness must change the output."""
        d = dict(json.loads(_HP_ZK.read_text()))
        bad = bytearray(bytes.fromhex(d["hiding_a"]))
        bad[0] ^= 0x01
        d["hiding_a"] = bytes(bad).hex()
        _instance, _witness, low, high, hiding_comms = _prove(d)
        proof = hp_as.serialize_proof_zk(cv, low, high, hiding_comms)
        self.assertNotEqual(proof.hex(), json.loads(_HP_ZK.read_text())["proof_hex"], "mutation did not change the proof")
        print("  mutation check: a perturbed hiding randomness diverges from the golden")


if __name__ == "__main__":
    absltest.main()
