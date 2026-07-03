"""Slice-2a byte-match: classic Poseidon FS sponge vs the real ark-sponge, over
BOTH Pasta cycle curves (Pallas and Vesta).

Replays the absorb/squeeze schedules dumped from arkworks' actual
`PoseidonSponge<CF>` (`cargo run --example dump_sponge -- <curve>`, where `CF` is
the curve's base field) through zorch's `Poseidon`+`DuplexSponge` over the same
117 ARK constants, and asserts every squeezed field element matches
byte-for-byte. A green run validates the regenerated ARK, the permutation
(params + round structure), and the duplex absorb/squeeze semantics together.

Running the SAME curve-generic sponge (`sponge.poseidon_params(cv, ...)`) against
the Vesta fixture — the constants reduce mod Vesta's base field instead of
Pallas's, so they differ — is the Phase-4 Slice-3 gate for the gamma sponge the
zk recursion NARK draws on Vesta.

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:sponge_test
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from accumulation_zorch import curve, sponge

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

# (curve, fixture) for each cycle curve — the same sponge, two base fields.
_CURVES = [
    (curve.PALLAS, _TESTDATA / "sponge_fixtures.json"),
    (curve.VESTA, _TESTDATA / "sponge_vesta_fixtures.json"),
]


class SpongeTest(absltest.TestCase):
    def test_sponge_schedules_match_arkworks(self) -> None:
        for cv, fixture in _CURVES:
            data = json.loads(fixture.read_text())
            ark_le = b"".join(bytes.fromhex(h) for h in data["ark_le_hex"])
            self.assertEqual(len(data["ark_le_hex"]), 117, "expected 117 ARK constants")
            params = sponge.poseidon_params(cv, ark_le)

            n_squeezes = 0
            for sched in data["schedules"]:
                sp = sponge.new_sponge(params)
                for op in sched["ops"]:
                    if op["op"] == "absorb":
                        arr = jnp.asarray(np.array(op["vals"], dtype=cv.fq))
                        sp = sp.absorb(arr)
                    else:
                        sp, out = sp.squeeze(op["n"])
                        got = np.asarray(out)
                        self.assertEqual(got.shape[0], op["n"], f"[{cv.name}] {sched['name']}: squeeze count")
                        for i, want_hex in enumerate(op["out"]):
                            got_hex = got[i].tobytes().hex()
                            self.assertEqual(got_hex, want_hex, (
                                f"[{cv.name}] {sched['name']} squeeze[{i}]: {got_hex} != {want_hex}"
                            ))
                        n_squeezes += 1
                print(f"  [{cv.name}] {sched['name']} OK")
            print(f"  [{cv.name}] classic Poseidon FS sponge matches ark-sponge "
                  f"({len(data['schedules'])} schedules, {n_squeezes} squeezes)")

    def test_nonnative_truncated_squeeze_matches_arkworks(self) -> None:
        for cv, fixture in _CURVES:
            data = json.loads(fixture.read_text())
            ark_le = b"".join(bytes.fromhex(h) for h in data["ark_le_hex"])
            params = sponge.poseidon_params(cv, ark_le)
            n = 0
            for case in data["nonnative_squeeze"]:
                sp = sponge.new_sponge(params)
                for v in case["absorb"]:
                    sp = sp.absorb(jnp.asarray(np.array([v], dtype=cv.fq)))
                sp, challenges = sponge.squeeze_challenges(sp, case["k"])
                self.assertEqual(len(challenges), len(case["challenges"]))
                for i, want_hex in enumerate(case["challenges"]):
                    got_hex = cv.fr(challenges[i]).tobytes().hex()
                    self.assertEqual(got_hex, want_hex, (
                        f"[{cv.name}] {case['name']} challenge[{i}]: {got_hex} != {want_hex}"
                    ))
                    n += 1
                print(f"  [{cv.name}] {case['name']} (k={case['k']}) OK")
            print(f"  [{cv.name}] nonnative truncated-128 squeeze -> Fr matches ark-sponge "
                  f"({n} challenges)")


if __name__ == "__main__":
    absltest.main()
