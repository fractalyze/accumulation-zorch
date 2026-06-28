"""Slice-4 end-to-end byte-match: the full zk IVC fold (num_addends=3).

Drives `r1cs_nark_as.prove_zk_fold` — one input folded INTO one prior accumulator
— and asserts the serialized folded accumulator (`instance ‖ witness`) and the AS
`Proof` byte-match arkworks. This stitches the two pieces validated in isolation by
`as_fold_zk_test` (the AS-level instance/witness combine) and `as_hp_fold_zk_test`
(the HP-level old-accumulator fold) into the complete fused prove, against the raw
serialized golden (`examples/dump_as_fold_zk.rs`: `golden_{instance,witness,proof}`).

The old accumulator (`acc_prev`) is fed as its parsed instance/witness components
— exactly the materialized form a prior fold (or `prove_zk` init) produces — and
every sampled randomness value (input's NARK, the fold's AS + HP) is replayed from
the dump.

Run (from the repo's `python/` dir, in the accumulation-zorch venv):

    JAX_PLATFORMS=cpu PYTHONPATH=.:<pasta-zorch>/zorch \
      python accumulation_zorch/testing/as_fold_zk_e2e_test.py
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from absl.testing import absltest

from accumulation_zorch import curve, nark, r1cs_nark_as, sponge

cv = curve.PALLAS

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
_FIXTURE = _TESTDATA / "as_fold_zk_fixtures.json"
_SPONGE = _TESTDATA / "sponge_fixtures.json"


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _matrix(rows: Any) -> nark.Matrix:
    return [[(_fr(coeff), idx) for coeff, idx in row] for row in rows]


def _point(p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _fold(d: Any, s: Any, params: Any) -> tuple[bytes, bytes, bytes]:
    a, b, c = (_matrix(d[k]) for k in ("a", "b", "c"))
    generators = [_point(g) for g in d["generators"]]
    hiding = _point(d["hiding"])

    input2 = [_fr(h) for h in s["input2_r1cs_input"]]
    witness2 = [_fr(h) for h in s["input2_witness"]]
    nark_r = [_fr(h) for h in s["r"]]
    nark_blinders = tuple(_fr(s[k]) for k in (
        "a_blinder", "b_blinder", "c_blinder", "r_a_blinder", "r_b_blinder", "r_c_blinder",
        "blinder_1", "blinder_2"))
    as_rand = tuple(_fr(s[k]) for k in ("as_rand_1", "as_rand_2", "as_rand_3"))
    hp_rand = tuple(_fr(s[k]) for k in ("hp_rand_1", "hp_rand_2", "hp_rand_3"))

    acc = s["acc_prev_instance"]
    accw = s["acc_prev_witness"]
    acc_r1cs_input = [_fr(h) for h in acc["r1cs_input"]]
    acc_comms = [np.asarray(_point(acc[k])) for k in (
        "comm_a", "comm_b", "comm_c", "hp_comm_1", "hp_comm_2", "hp_comm_3")]
    acc_blinded_witness = [_fr(h) for h in accw["r1cs_blinded_witness"]]
    acc_sigma_abc = tuple(_fr(accw[k]) for k in ("sigma_a", "sigma_b", "sigma_c"))
    acc_hp_a_vec = [_fr(h) for h in accw["hp_a_vec"]]
    acc_hp_b_vec = [_fr(h) for h in accw["hp_b_vec"]]
    acc_hp_rand = tuple(_fr(accw[k]) for k in ("hp_rand_1", "hp_rand_2", "hp_rand_3"))

    return r1cs_nark_as.prove_zk_fold(
        cv, a, b, c, input2, witness2, generators, hiding, params,
        bytes.fromhex(d["nark_matrices_hash_hex"]), bytes.fromhex(d["as_matrices_hash_hex"]),
        d["supported_num_elems"], nark_r, nark_blinders,
        _fr(s["as_r1cs_r_input"]), _fr(s["as_r1cs_r_witness"]), as_rand,
        _fr(s["hp_hiding_a"]), _fr(s["hp_hiding_b"]), hp_rand,
        acc_r1cs_input, acc_comms, acc_blinded_witness, acc_sigma_abc,
        acc_hp_a_vec, acc_hp_b_vec, acc_hp_rand)


class AsFoldZkE2eTest(absltest.TestCase):
    def test_fold_end_to_end_matches_arkworks(self) -> None:
        d = json.loads(_FIXTURE.read_text())
        params = _params()
        for s in d["seeds"]:
            acc_instance, acc_witness, proof = _fold(d, s, params)
            self.assertEqual(acc_instance.hex(), s["golden_instance_hex"], (
                f"[seed {s['seed']}] folded acc.instance:\n got  {acc_instance.hex()}"
                f"\n want {s['golden_instance_hex']}"))
            self.assertEqual(acc_witness.hex(), s["golden_witness_hex"], (
                f"[seed {s['seed']}] folded acc.witness:\n got  {acc_witness.hex()}"
                f"\n want {s['golden_witness_hex']}"))
            self.assertEqual(proof.hex(), s["golden_proof_hex"], (
                f"[seed {s['seed']}] fold proof:\n got  {proof.hex()}\n want {s['golden_proof_hex']}"))
            print(f"  [seed {s['seed']}] (acc.instance {len(acc_instance)}B ‖ acc.witness "
                  f"{len(acc_witness)}B ‖ proof {len(proof)}B) byte-matches arkworks")

    def test_mutation_breaks_match(self) -> None:
        """Perturbing the old accumulator's HP witness randomness must change the
        folded witness (it feeds the combined HP randomness, not the instance — whose
        HP commitments fold the fixed `acc_comms` inputs)."""
        d = json.loads(_FIXTURE.read_text())
        s = dict(d["seeds"][0])
        accw = dict(s["acc_prev_witness"])
        bad = bytearray(bytes.fromhex(accw["hp_rand_1"]))
        bad[0] ^= 0x01
        accw["hp_rand_1"] = bytes(bad).hex()
        s["acc_prev_witness"] = accw
        _i, acc_witness, _p = _fold(d, s, _params())
        self.assertNotEqual(acc_witness.hex(), d["seeds"][0]["golden_witness_hex"], "mutation did not change the fold")
        print("  mutation check: a perturbed acc HP randomness diverges from the golden witness")


if __name__ == "__main__":
    absltest.main()
