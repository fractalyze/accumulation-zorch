"""Slice-2 byte-match: the no-zk NARK prove of the **recursion-verifier circuit**
on Vesta.

The recursion half-step proves the Pasta-cycle AS verifier gadget — a real
~22.5K-constraint × ~21K-var R1CS (but sparse, ~6 non-zeros/row) — as a Vesta
NARK. This replays that R1CS through the curve-generic
`nark.prove_no_zk(VESTA, …)` and byte-matches the golden proof the crate's
real `R1CSNark::prove` emits, confirming the frx NARK core scales from the toy
circuit to the recursion circuit.

The dense `M·z` is infeasible here (`rows × vars` ≈ 471M entries ≈ 15 GB), so
`prove_no_zk` reduces `M·z` **on-device** from the sparse CSR
(`field.sparse_matvec`) and commits with one `lax.msm` — the export-shaped trace.

The fixture is large (~17 MB) so it is generated **off-tree**, not committed, and
`$ACCUMULATION_ZORCH_ARTIFACTS` must name the directory holding it — see
`recursion_artifacts.py` for why there is no default. Run on demand:

    ACCUMULATION_ZORCH_ARTIFACTS=$PWD/artifacts \
      cargo test --features recursion --test recursion_step dump_recursion_nark
    ACCUMULATION_ZORCH_ARTIFACTS=$PWD/artifacts \
      bazel test //python/accumulation_zorch/testing:recursion_nark_test

The target is `manual`, so `bazel test //python/...` does not run it: a clean checkout
has no fixture, and a test that cannot check anything must not report a pass.
"""

import json
from pathlib import Path
from typing import Any

from absl.testing import absltest

import recursion_artifacts
from accumulation_zorch import curve, nark

cv = curve.VESTA  # the forward half-step proves on Vesta

_FIXTURE = "recursion_nark_fixtures.json"
_DUMP = "cargo test --features recursion --test recursion_step dump_recursion_nark"


def _parse_matrix(rows: Any) -> Any:
    return [[(int.from_bytes(bytes.fromhex(coeff), "little"), idx) for coeff, idx in row] for row in rows]


def _fr_list(hexes: Any) -> Any:
    return [int.from_bytes(bytes.fromhex(h), "little") for h in hexes]


def _load(path: Path) -> Any:
    d = json.loads(path.read_text())
    a, b, c = (_parse_matrix(d[k]) for k in ("a", "b", "c"))
    input_ = _fr_list(d["input"])
    witness = _fr_list(d["witness"])
    generators = [
        cv.g1((
            int.from_bytes(bytes.fromhex(g["x_le_hex"]), "little"),
            int.from_bytes(bytes.fromhex(g["y_le_hex"]), "little"),
        ))
        for g in d["generators"]
    ]
    return d, a, b, c, input_, witness, generators


class RecursionNarkTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.fixture = recursion_artifacts.fixture(_FIXTURE, _DUMP)

    def test_recursion_nark_no_zk_fused_proof_matches_arkworks(self) -> None:
        """The fused on-device variant at recursion scale: `prove_no_zk` reduces
        `M·z` **in-trace** via `field.sparse_matvec` (a scatter-free CSR prefix sum
        over the ~140K sparse nonzeros) instead of host-side, and commits with
        `lax.msm`. This is the
        Slice-3 criterion — the export-shaped no-zk NARK core byte-matches the golden
        at the real ~22.5K-constraint scale, so the GPU export (next) reproduces the
        crate's `R1CSNark::prove`."""
        d, a, b, c, input_, witness, generators = _load(self.fixture)
        proof = nark.prove_no_zk(cv, a, b, c, input_, witness, generators)
        self.assertEqual(proof.hex(), d["proof_hex"], (
            f"[vesta] fused recursion NARK proof diverged from arkworks "
            f"(got {len(proof)}B, want {len(d['proof_hex'])//2}B)"
        ))
        print(
            f"  [vesta] fused (on-device sparse M·z) recursion no-zk NARK proof "
            f"byte-matches arkworks ({d['num_constraints']} constraints, {len(proof)} bytes)"
        )


if __name__ == "__main__":
    absltest.main()
