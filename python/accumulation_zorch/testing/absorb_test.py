"""Slice-2b byte-match: the Absorbable field-element layer, fork() domain
separation, and the NARK gamma challenge, all vs the real ark-sponge.

Replays fixtures dumped from arkworks (`cargo run --example dump_absorb`) and
asserts the zorch `DuplexSponge`, driven through the ported `Absorbable`
packing, reproduces every squeezed element / challenge byte-for-byte. The
fixtures form an isolating ladder so a divergence localizes to one primitive:

* identity point packing ([0, 1, 1], the arkworks `Affine::zero()` trap),
* `fork(domain)` (the double length-prefix + 31-byte CAPACITY/8 chunking),
* raw `&[u8]` absorb at chunk-boundary lengths,
* point absorb (incl. the identity),
* the full `R1CSNark::compute_challenge` (gamma).

The 117 Poseidon ARK constants come from the slice-2 sponge fixtures.

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:absorb_test
"""

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from accumulation_zorch import absorbable, curve, jcurve, nark, sponge

cv = curve.PALLAS

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
_ABSORB = _TESTDATA / "absorb_fixtures.json"
_SPONGE = _TESTDATA / "sponge_fixtures.json"


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _squeeze_hex(sp: Any, n: int) -> Any:
    sp, out = sp.squeeze(n)
    got = np.asarray(out)
    return [got[i].tobytes().hex() for i in range(n)]


def _point_from_fixture(p: Any) -> Any:
    """Rebuild a point from a `{infinity, x_le_hex, y_le_hex}` fixture entry.
    The identity is the all-zero zk_dtypes encoding (x=0, y=0)."""
    if p["infinity"]:
        return cv.g1((0, 0))
    x = int.from_bytes(bytes.fromhex(p["x_le_hex"]), "little")
    y = int.from_bytes(bytes.fromhex(p["y_le_hex"]), "little")
    return cv.g1((x, y))


class AbsorbTest(absltest.TestCase):
    def test_identity_point_packs_as_0_1_1(self) -> None:
        want = json.loads(_ABSORB.read_text())["identity_to_field_elements_le_hex"]
        fes = np.asarray(absorbable.point_to_field_array(cv, cv.g1((0, 0))))
        got = [fes[i].tobytes().hex() for i in range(fes.shape[0])]
        self.assertEqual(got, want, f"identity packing: {got} != {want}")
        print("  identity point packs as [0, 1, 1] (arkworks Affine::zero) OK")

    def test_fork_matches_arkworks(self) -> None:
        data = json.loads(_ABSORB.read_text())
        params = _params()
        for case in data["fork"]:
            sp = absorbable.fork(cv, sponge.new_sponge(params), bytes.fromhex(case["domain_hex"]))
            self.assertEqual(_squeeze_hex(sp, 2), case["squeeze"], f"fork {case['domain_utf8']}")
            print(f"  fork({case['domain_utf8']}) OK")

    def test_bytes_absorb_matches_arkworks(self) -> None:
        data = json.loads(_ABSORB.read_text())
        params = _params()
        for case in data["bytes_absorb"]:
            sp = absorbable.absorb_bytes(cv, sponge.new_sponge(params), bytes.fromhex(case["data_hex"]))
            self.assertEqual(_squeeze_hex(sp, 2), case["squeeze"], f"bytes_absorb len={case['len']}")
            print(f"  &[u8] absorb (len={case['len']}) OK")

    def test_point_absorb_matches_arkworks(self) -> None:
        data = json.loads(_ABSORB.read_text())
        params = _params()
        for case in data["point_absorb"]:
            sp = sponge.new_sponge(params)
            for p in case["points"]:
                sp = absorbable.absorb_point(cv, sp, _point_from_fixture(p))
            self.assertEqual(_squeeze_hex(sp, 2), case["squeeze"], f"point_absorb {case['label']}")
            print(f"  point absorb ({case['label']}) OK")

    def test_point_to_field_array_jax_matches_host(self) -> None:
        """The in-jit batched affine short-Weierstrass point packing reproduces the
        host `point_to_field_array` concatenation byte-for-byte, including the
        arkworks identity `[0, 1, 1]` convention (the all-zero point in the batch)."""
        g = json.loads(_ABSORB.read_text())["gamma"]
        points = [_point_from_fixture(c) for c in g["comms"]] + [cv.g1((0, 0))]
        host = np.concatenate([absorbable.point_to_field_array(cv, p) for p in points])
        pack = jax.jit(lambda pts: absorbable.point_to_field_array_jax(cv, pts))
        got = np.asarray(pack(jcurve.stack_affine(cv, points)))
        self.assertEqual(host.tobytes(), got.tobytes(), "in-jit point packing != host packing")
        print("  point_to_field_array_jax byte-matches host (incl identity) OK")

    def test_gamma_challenge_matches_arkworks(self) -> None:
        g = json.loads(_ABSORB.read_text())["gamma"]
        params = _params()
        matrices_hash = bytes.fromhex(g["matrices_hash_hex"])
        inputs = [int.from_bytes(bytes.fromhex(h), "little") for h in g["inputs_le_hex"]]
        comms = [_point_from_fixture(c) for c in g["comms"]]
        self.assertIsNone(g["randomness"], "this slice ports the no-zk gamma only")
        gamma = nark.compute_challenge(cv, params, matrices_hash, inputs, comms, randomness=None)
        got = gamma.tobytes().hex()
        self.assertEqual(got, g["gamma_hex"], f"gamma: {got} != {g['gamma_hex']}")
        print("  NARK gamma challenge byte-matches R1CSNark::compute_challenge OK")


if __name__ == "__main__":
    absltest.main()
