"""Byte-match: the in-trace JAX `ArkIpaChallenger` reproduces the NumPy IPA-PC
Fiat-Shamir oracle (`ipa_pc.succinct_check_challenges`) exactly, over BOTH Pasta
cycle curves.

`ipa_challenger.ArkIpaChallenger` is the arkworks-faithful challenger zorch's IPA
fold drives (the `zorch.pcs.ipa.challenger.IpaChallenger` seam): a fresh
`"IPA-PC-2020"`-forked Poseidon sponge per round, the previous challenge
re-absorbed, a nonnative truncated-128 squeeze. This test replays a golden IPA
opening fixture's `(commitment, point, value, l_vec, r_vec)` through the
challenger's `seed` + per-round `challenge` and asserts each round challenge
equals the NumPy oracle's `succinct_check_challenges` round challenge as an exact
Python int — the byte-match the whole zorch-integration refactor hinges on.

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:ipa_challenger_test
"""

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from absl.testing import absltest

from accumulation_zorch import curve, ipa_challenger, ipa_pc, sponge

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

_CURVES = [
    (curve.PALLAS, _TESTDATA / "ipa_fixtures.json", _TESTDATA / "sponge_fixtures.json"),
    (curve.VESTA, _TESTDATA / "ipa_vesta_fixtures.json", _TESTDATA / "sponge_vesta_fixtures.json"),
]


def _fr(h: str) -> int:
    return int.from_bytes(bytes.fromhex(h), "little")


def _point(cv: curve.Curve, p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _params(cv: curve.Curve, sponge_fixture: Path) -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(sponge_fixture.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _load_ipa_fixture(cv: curve.Curve, ipa_fixture: Path) -> tuple:
    d = json.loads(ipa_fixture.read_text())
    return (
        _point(cv, d["commitment"]),
        _fr(d["point"]),
        _fr(d["evaluation"]),
        [_point(cv, p) for p in d["l_vec"]],
        [_point(cv, p) for p in d["r_vec"]],
    )


class IpaChallengerTest(absltest.TestCase):
    def test_ark_challenger_matches_numpy_oracle(self) -> None:
        for cv, ipa_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            comm, point, value, l_vec, r_vec = _load_ipa_fixture(cv, ipa_fixture)

            want = ipa_pc.succinct_check_challenges(cv, params, comm, point, value, l_vec, r_vec)

            ch = ipa_challenger.ark_challenger(cv, params)
            ch, _xi0 = ch.seed(comm, point, value)
            got = []
            for l, r in zip(l_vec, r_vec):
                ch, u = ch.challenge(l, r)
                got.append(int(jnp.asarray(u).reshape(())))

            self.assertEqual(got, want, f"[{cv.name}] ArkIpaChallenger round challenges != oracle")
            print(f"  [{cv.name}] ArkIpaChallenger ({len(want)} rounds) byte-matches "
                  f"ipa_pc.succinct_check_challenges")


if __name__ == "__main__":
    absltest.main()
