"""Slice-2 byte-match: the no-zk NARK prove of the **recursion-verifier circuit**
on Vesta.

The recursion half-step proves the Pasta-cycle AS verifier gadget — a real
~22.5K-constraint × ~21K-var R1CS (but sparse, ~6 non-zeros/row) — as a Vesta
NARK. This replays that R1CS through the curve-generic
`nark.prove_no_zk(VESTA, …)` and byte-matches the golden proof the crate's
real `R1CSNark::prove` emits, confirming the jax NARK core scales from the toy
circuit to the recursion circuit.

The dense `M·z` is infeasible here (`rows × vars` ≈ 471M entries ≈ 15 GB), so
`prove_no_zk` reduces `M·z` **on-device** from the sparse COO
(`jfield.sparse_matvec`) and commits with one `lax.msm` — the export-shaped trace.

The fixture is large (~17 MB) so it is generated **off-tree**, not committed:

    ACCUMULATION_ZORCH_ARTIFACTS=<dir> \
      cargo test --features recursion --test recursion_step dump_recursion_nark

This test reads it from `$ACCUMULATION_ZORCH_ARTIFACTS` (default `artifacts/`)
and **skips** when absent — the same on-demand contract as the `#[ignore]` GPU
gates.

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:recursion_nark_test
"""

import json
import os
from pathlib import Path
from typing import Any

from absl.testing import absltest

from accumulation_zorch import curve, nark

cv = curve.VESTA  # the forward half-step proves on Vesta

_REPO = Path(__file__).resolve().parents[3]
_ARTIFACTS = Path(os.environ.get("ACCUMULATION_ZORCH_ARTIFACTS", str(_REPO / "artifacts")))
_FIXTURE = _ARTIFACTS / "recursion_nark_fixtures.json"


def _parse_matrix(rows: Any) -> Any:
    return [[(int.from_bytes(bytes.fromhex(coeff), "little"), idx) for coeff, idx in row] for row in rows]


def _fr_list(hexes: Any) -> Any:
    return [int.from_bytes(bytes.fromhex(h), "little") for h in hexes]


def _load() -> Any:
    d = json.loads(_FIXTURE.read_text())
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
        if not _FIXTURE.exists():
            self.skipTest(
                f"no fixture at {_FIXTURE} "
                "(generate it: cargo test --features recursion --test recursion_step dump_recursion_nark)")

    def test_recursion_nark_no_zk_fused_proof_matches_arkworks(self) -> None:
        """The fused on-device variant at recursion scale: `prove_no_zk` reduces
        `M·z` **in-trace** via `jfield.sparse_matvec` (`segment_sum` over the ~140K
        sparse nonzeros) instead of host-side, and commits with `lax.msm`. This is the
        Slice-3 criterion — the export-shaped no-zk NARK core byte-matches the golden
        at the real ~22.5K-constraint scale, so the GPU export (next) reproduces the
        crate's `R1CSNark::prove`."""
        d, a, b, c, input_, witness, generators = _load()
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
