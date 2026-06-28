"""Phase-1 slice-2: jit-able Fiat-Shamir challenge squeeze byte-matches arkworks.

The CPU port's `sponge.squeeze_challenges` decodes each squeezed Fq element to a
Python bigint and slices bits in a Python loop (`squeeze_bits` /
`squeeze_nonnative`) — not jit-able. `jsponge.challenges_from_fq` does the same
ark-sponge bit math (low `CAPACITY=254` bits per element, concatenate, window
into `size`-bit challenges, repack LE into Fr) entirely in jax, returning the
challenges as an Fr field-element array (the on-device form the fused core
keeps).

Two gates:
- the arkworks-pinned NARK `gamma` challenge (`absorb_fixtures.json`), the k=1
  / 128-bit path the prover actually squeezes;
- a cross-check vs the CPU `sponge.squeeze_challenges` at k=1,2,3 — k≥2 squeezes
  two Fq elements (256 > 254), so the challenge stream crosses the 254-bit
  element boundary, the case a byte-level (not bit-level) extraction gets wrong.

Run (from the repo's `python/` dir, in the accumulation-zorch venv):

    JAX_PLATFORMS=cpu PYTHONPATH=.:<pasta-zorch>/zorch \
      python accumulation_zorch/testing/jsponge_test.py
"""

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from accumulation_zorch import absorbable, curve, jsponge, nark, sponge

cv = curve.PALLAS

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
_ABSORB = _TESTDATA / "absorb_fixtures.json"
_SPONGE = _TESTDATA / "sponge_fixtures.json"

_SIZE = min(128, sponge.FR_CAPACITY)  # the prover's CHALLENGE_SIZE window


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _point_from_fixture(p: Any) -> Any:
    if p["infinity"]:
        return cv.g1((0, 0))
    return cv.g1((
        int.from_bytes(bytes.fromhex(p["x_le_hex"]), "little"),
        int.from_bytes(bytes.fromhex(p["y_le_hex"]), "little"),
    ))


def _gamma_sponge(params: Any, g: Any) -> Any:
    """Rebuild the no-zk NARK gamma sponge exactly as `nark.compute_challenge`,
    stopped just before the squeeze."""
    matrices_hash = bytes.fromhex(g["matrices_hash_hex"])
    inputs = [int.from_bytes(bytes.fromhex(h), "little") for h in g["inputs_le_hex"]]
    comms = [_point_from_fixture(c) for c in g["comms"]]
    sp = absorbable.fork(cv, sponge.new_sponge(params), nark.PROTOCOL_NAME)
    sp = absorbable.absorb_bytes(cv, sp, matrices_hash)
    sp = absorbable.absorb_bytes(cv, sp, b"".join(int(s).to_bytes(32, "little") for s in inputs))
    arrs = [absorbable.point_to_field_array(cv, c) for c in comms]
    arrs.append(absorbable.option_flag(cv, False))  # randomness = None
    return sp.absorb(jnp.asarray(np.concatenate(arrs)))


def _n_elems(k: int) -> int:
    """ark-sponge `squeeze_bits` element count for `k` challenges: the bit stream
    yields `FQ_CAPACITY` bits per squeezed Fq element."""
    num_bits = k * _SIZE
    return (num_bits + sponge.FQ_CAPACITY - 1) // sponge.FQ_CAPACITY


class JspongeTest(absltest.TestCase):
    def test_gamma_challenge_matches_arkworks(self) -> None:
        g = json.loads(_ABSORB.read_text())["gamma"]
        self.assertIsNone(g["randomness"])
        sp = _gamma_sponge(_params(), g)
        _, elems = sp.squeeze(_n_elems(1))
        fr = jsponge.challenges_from_fq(jnp.asarray(elems), 1, _SIZE, cv)
        got = np.asarray(fr)[0].tobytes().hex()
        self.assertEqual(got, g["gamma_hex"], f"gamma: {got} != {g['gamma_hex']}")
        print("  jit gamma challenge byte-matches R1CSNark::compute_challenge OK")

    def test_matches_cpu_squeeze_across_element_boundary(self) -> None:
        """k=1,2,3 vs the CPU `squeeze_challenges`; k≥2 crosses the 254-bit Fq
        element boundary."""
        params = _params()
        for k in (1, 2, 3):
            base = absorbable.absorb_bytes(cv, sponge.new_sponge(params), bytes([k, 7, 9]))
            _, want = sponge.squeeze_challenges(base, k)  # CPU Python-loop ints
            _, elems = base.squeeze(_n_elems(k))
            fr = jsponge.challenges_from_fq(jnp.asarray(elems), k, _SIZE, cv)
            got = [int.from_bytes(np.asarray(fr)[i].tobytes(), "little") for i in range(k)]
            self.assertEqual(got, want, f"k={k}: {got} != {want}")
            print(f"  k={k} ({_n_elems(k)} Fq elems) matches CPU squeeze_challenges OK")


if __name__ == "__main__":
    absltest.main()
