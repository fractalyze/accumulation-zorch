"""Phase-1 slice-1 tracer: a single jit-able Pedersen commitment byte-matches
arkworks.

De-risks the whole jit/GPU-exportable port premise in one minimal path:
the first-round commitment `comm_a = commit(M·z)` is computed by an
`@jax.jit` function that does the `M·z` field reduction in jax (a dense Fr
matmul) and the commitment via `lax.msm` — no `zk_dtypes` numpy field/EC
arithmetic on the prove path — and must reproduce the byte-exact `comm_a` the
CPU port (`nark_test`) already pins to arkworks (`nark_fixtures.json`).

If this can't be made jit-able / byte-exact on CPU, the port stops here.

Run (from the repo's `python/` dir, in the accumulation-zorch venv):

    JAX_PLATFORMS=cpu PYTHONPATH=.:<pasta-zorch>/zorch \
      python accumulation_zorch/testing/jcurve_test.py
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from absl.testing import absltest

from accumulation_zorch import curve, jcurve

cv = curve.PALLAS

_FIXTURES = Path(__file__).resolve().parents[2] / "testdata" / "nark_fixtures.json"


def _fr_list(hexes: Any) -> list[int]:
    return [int.from_bytes(bytes.fromhex(h), "little") for h in hexes]


def _dense_matrix(rows: Any, num_vars: int) -> np.ndarray:
    """A sparse `Matrix<Fr>` row list `[[(coeff_hex, idx), ...], ...]` densified to
    a `(num_rows × num_vars)` fr array — the jit commitment takes the dense form
    (the DummyCircuit matrices are tiny; jagged is a Phase-2 perf concern)."""
    dense = np.zeros((len(rows), num_vars), dtype=cv.fr)
    for r, row in enumerate(rows):
        for coeff_hex, idx in row:
            dense[r, idx] = cv.fr(int.from_bytes(bytes.fromhex(coeff_hex), "little"))
    return dense


def _load() -> Any:
    d = json.loads(_FIXTURES.read_text())
    input_ = _fr_list(d["input"])
    witness = _fr_list(d["witness"])
    z = np.array(input_ + witness, dtype=cv.fr)
    a_dense = _dense_matrix(d["a"], len(z))
    bases = np.frombuffer(
        b"".join(
            cv.g1((
                int.from_bytes(bytes.fromhex(g["x_le_hex"]), "little"),
                int.from_bytes(bytes.fromhex(g["y_le_hex"]), "little"),
            )).tobytes()
            for g in d["generators"]
        ),
        dtype=cv.g1,
    ).copy()
    return d, a_dense, z, bases


class JcurveTest(absltest.TestCase):
    def test_jit_commitment_matches_arkworks_comm_a(self) -> None:
        """`comm_a = commit(A·z)` via the jit `M·z`-then-`lax.msm` core byte-matches
        the arkworks-pinned `comm_a` (the leading 33B of the no-zk NARK proof)."""
        d, a_dense, z, bases = _load()
        want_hex = d["proof_hex"][0:66]  # comm_a = first 33B of the proof
        point = jcurve.commit_dense(a_dense, z, bases)
        got_hex = curve.point_to_bytes(cv, np.asarray(point)).hex()
        self.assertEqual(got_hex, want_hex, f"comm_a: got {got_hex} want {want_hex}")
        print(f"  jit commit(A·z) byte-matches arkworks comm_a ({len(got_hex)//2} bytes)")


if __name__ == "__main__":
    absltest.main()
