"""Slice-5 byte-match: the **zk fold** of the recursion-verifier circuit, both
cycle directions — the full IVC step (forward / reverse), num_addends=3.

Promotes the toy end-to-end fold (`as_fold_zk_e2e_test`) to the recursion circuit
(~77.5K constraints, the ~2¹⁷-MSM scale): one NARK proof of the cycle-partner
verifier circuit folded INTO a prior accumulator, byte-matched against the crate's
real `ASForR1CSNark::prove`. The same fused `r1cs_nark_as.prove_zk_fold` the toy
validates, here on `VESTA` (forward) and `PALLAS` (reverse) with the on-device
sparse `M·z` (densifying the recursion R1CS is ~15 GB). `acc_prev` is fed as its
materialized instance/witness components; every sampled randomness value (the
input's NARK, the fold's AS + HP) is replayed from the dump.

The fixtures are large so they are generated **off-tree**, not committed:

    ACCUMULATION_ZORCH_ARTIFACTS=<dir> cargo test --features recursion \
      --test recursion_step vesta::dump::dump_recursion_fold_zk    # forward
    ACCUMULATION_ZORCH_ARTIFACTS=<dir> cargo test --features recursion \
      --test recursion_step pallas::dump::dump_recursion_fold_zk   # reverse

Each direction reads its fixture from `$ACCUMULATION_ZORCH_ARTIFACTS` (default
`artifacts/`) and **skips** when absent — the same on-demand contract as the
`#[ignore]` GPU gates.

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:recursion_fold_zk_test
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from absl.testing import absltest

from accumulation_zorch import curve, nark, r1cs_nark_as, sponge

_REPO = Path(__file__).resolve().parents[3]
_ARTIFACTS = Path(os.environ.get("ACCUMULATION_ZORCH_ARTIFACTS", str(_REPO / "artifacts")))
_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

# (label, curve, fixture filename, Poseidon sponge fixture over the constraint field).
# Forward folds on Vesta (constraint field ark_vesta::Fq); reverse on Pallas
# (constraint field ark_pallas::Fq = the toy's sponge_fixtures.json).
_DIRECTIONS = [
    ("vesta forward", curve.VESTA, "recursion_fold_zk_fixtures.json",
     "sponge_vesta_fixtures.json"),
    ("pallas reverse", curve.PALLAS, "recursion_fold_zk_pallas_fixtures.json",
     "sponge_fixtures.json"),
]


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _matrix(rows: Any) -> nark.Matrix:
    return [[(_fr(coeff), idx) for coeff, idx in row] for row in rows]


def _point(cv: Any, p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _params(cv: Any, sponge_file: str) -> Any:
    ark_le = b"".join(bytes.fromhex(h)
                      for h in json.loads((_TESTDATA / sponge_file).read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _fold(cv: Any, d: Any, params: Any) -> tuple[bytes, bytes, bytes]:
    a, b, c = (_matrix(d[k]) for k in ("a", "b", "c"))
    generators = [_point(cv, g) for g in d["generators"]]
    hiding = _point(cv, d["hiding"])

    input2 = [_fr(h) for h in d["input2_r1cs_input"]]
    witness2 = [_fr(h) for h in d["input2_witness"]]
    nark_r = [_fr(h) for h in d["r"]]
    nark_blinders = tuple(_fr(d[k]) for k in (
        "a_blinder", "b_blinder", "c_blinder", "r_a_blinder", "r_b_blinder", "r_c_blinder",
        "blinder_1", "blinder_2"))
    as_rand = tuple(_fr(d[k]) for k in ("as_rand_1", "as_rand_2", "as_rand_3"))
    hp_rand = tuple(_fr(d[k]) for k in ("hp_rand_1", "hp_rand_2", "hp_rand_3"))

    acc = d["acc_prev_instance"]
    accw = d["acc_prev_witness"]
    acc_r1cs_input = [_fr(h) for h in acc["r1cs_input"]]
    acc_comms = [np.asarray(_point(cv, acc[k])) for k in (
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
        _fr(d["as_r1cs_r_input"]), _fr(d["as_r1cs_r_witness"]), as_rand,
        _fr(d["hp_hiding_a"]), _fr(d["hp_hiding_b"]), hp_rand,
        acc_r1cs_input, acc_comms, acc_blinded_witness, acc_sigma_abc,
        acc_hp_a_vec, acc_hp_b_vec, acc_hp_rand)


def _check_direction(label: str, cv: Any, fixture: str, sponge_file: str) -> bool:
    path = _ARTIFACTS / fixture
    if not path.exists():
        print(f"  SKIP [{label}] — no fixture at {path}")
        return False
    d = json.loads(path.read_text())
    acc_instance, acc_witness, proof = _fold(cv, d, _params(cv, sponge_file))
    assert acc_instance.hex() == d["golden_instance_hex"], (
        f"[{label}] folded acc.instance diverged ({len(acc_instance)}B)")
    assert acc_witness.hex() == d["golden_witness_hex"], (
        f"[{label}] folded acc.witness diverged ({len(acc_witness)}B)")
    assert proof.hex() == d["golden_proof_hex"], (
        f"[{label}] fold proof diverged ({len(proof)}B)")
    print(f"  [{label}] recursion zk fold byte-matches arkworks "
          f"({d['num_constraints']} constraints, acc.instance {len(acc_instance)}B ‖ "
          f"acc.witness {len(acc_witness)}B ‖ proof {len(proof)}B)")
    return True


class RecursionFoldZkTest(absltest.TestCase):
    def test_recursion_fold_matches_arkworks(self) -> None:
        ran = [_check_direction(*dirn) for dirn in _DIRECTIONS]
        if not any(ran):
            print("  (no fixtures present — nothing checked)")


if __name__ == "__main__":
    absltest.main()
