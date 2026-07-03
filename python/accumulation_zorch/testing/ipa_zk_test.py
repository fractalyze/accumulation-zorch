"""Slice-5a byte-match: IPA-PC **zk/hiding** `succinct_check` round challenges +
`h(X)` check-poly coefficients vs the real arkworks `ipa_pc::succinct_check`, over
BOTH Pasta cycle curves — the zk twin of `ipa_test.py`.

Replays a hiding IPA opening dumped from the crate's real `InnerProductArgPC`
(`cargo run --example dump_ipa_zk -- <curve>`: the commitment, point + evaluation,
the proof's `l_vec`/`r_vec`, AND the proof's `hiding_comm`/`rand`) through the
ported zk succinct check, and asserts the round challenges + dense `compute_coeffs`
expansion + evaluation match arkworks byte-for-byte. Unlike no-zk, the round
challenges are seeded from the hiding-folded commitment
`commitment + hiding_comm·hiding_challenge − s·rand` (the `IPA-PC-2020` sponge
absorbing `commitment, hiding_comm, to_bytes![point, value]` for the hiding
challenge), so this exercises the hiding block + its group reduction. `coeffs` /
`evaluate` are challenge-only and unchanged from no-zk — fed the golden zk
challenges they isolate the polynomial expansion.

The Poseidon ARK constants come from the per-curve sponge fixture, as in the other
byte-match tests. Running the same curve-generic port against the Vesta fixture is
the curve-generic gate.

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:ipa_zk_test
"""

import json
from pathlib import Path
from typing import Any

from absl.testing import absltest

from accumulation_zorch import curve, ipa_pc, sponge

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

_CURVES = [
    (curve.PALLAS, _TESTDATA / "ipa_zk_fixtures.json", _TESTDATA / "sponge_fixtures.json"),
    (curve.VESTA, _TESTDATA / "ipa_zk_vesta_fixtures.json", _TESTDATA / "sponge_vesta_fixtures.json"),
]


def _fr(h: str) -> int:
    return int.from_bytes(bytes.fromhex(h), "little")


def _point(cv: curve.Curve, p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _params(cv: curve.Curve, sponge_fixture: Path) -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(sponge_fixture.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _load(cv: curve.Curve, ipa_fixture: Path) -> Any:
    d = json.loads(ipa_fixture.read_text())
    return {
        "d": d,
        "commitment": _point(cv, d["commitment"]),
        "point": _fr(d["point"]),
        "value": _fr(d["evaluation"]),
        "l_vec": [_point(cv, p) for p in d["l_vec"]],
        "r_vec": [_point(cv, p) for p in d["r_vec"]],
        "s": _point(cv, d["s"]),
        "hiding_comm": _point(cv, d["hiding_comm"]),
        "rand": _fr(d["rand"]),
    }


def _zk_challenges(cv: curve.Curve, params: Any, f: Any) -> list[int]:
    return ipa_pc.succinct_check_challenges_zk(
        cv, params, f["commitment"], f["point"], f["value"], f["l_vec"], f["r_vec"],
        f["s"], f["hiding_comm"], f["rand"])


class IpaZkTest(absltest.TestCase):
    def test_zk_succinct_check_round_challenges_match_arkworks(self) -> None:
        for cv, ipa_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            f = _load(cv, ipa_fixture)
            got = _zk_challenges(cv, params, f)
            want = f["d"]["round_challenges"]
            self.assertEqual(len(got), len(want), f"[{cv.name}] challenge count")
            for i, want_hex in enumerate(want):
                got_hex = cv.fr(got[i]).tobytes().hex()
                self.assertEqual(got_hex, want_hex, f"[{cv.name}] zk round challenge[{i}]: {got_hex} != {want_hex}")
            print(f"  [{cv.name}] zk succinct_check round challenges (hiding-folded seed) "
                  f"byte-match arkworks ({len(got)} rounds)")

    def test_zk_check_poly_coeffs_and_eval_match_arkworks(self) -> None:
        """`h(X)` coeffs + evaluation fed the golden zk challenges — `compute_coeffs` /
        `evaluate` are challenge-only, so this confirms the zk challenges drive the same
        expansion."""
        for cv, ipa_fixture, sponge_fixture in _CURVES:
            d = json.loads(ipa_fixture.read_text())
            challenges = [_fr(h) for h in d["round_challenges"]]
            point = _fr(d["point"])

            coeffs = ipa_pc.compute_coeffs(cv, challenges)
            for i, want_hex in enumerate(d["coeffs"]):
                got_hex = cv.fr(coeffs[i]).tobytes().hex()
                self.assertEqual(got_hex, want_hex, f"[{cv.name}] zk h(X) coeff[{i}]: {got_hex} != {want_hex}")

            got_eval = cv.fr(ipa_pc.evaluate(cv, challenges, point)).tobytes().hex()
            self.assertEqual(got_eval, d["eval_at_point"], f"[{cv.name}] zk h(point): {got_eval} != {d['eval_at_point']}")
            print(f"  [{cv.name}] zk h(X) compute_coeffs ({len(coeffs)} coeffs) + evaluate "
                  f"byte-match arkworks")

    def test_zk_succinct_check_end_to_end_matches_arkworks(self) -> None:
        """The full zk path: hiding-folded sponge → round challenges → `h(X)` coeffs."""
        for cv, ipa_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            f = _load(cv, ipa_fixture)
            challenges = _zk_challenges(cv, params, f)
            coeffs = ipa_pc.compute_coeffs(cv, challenges)
            for i, want_hex in enumerate(f["d"]["coeffs"]):
                got_hex = cv.fr(coeffs[i]).tobytes().hex()
                self.assertEqual(got_hex, want_hex, f"[{cv.name}] zk end-to-end h(X) coeff[{i}]: {got_hex} != {want_hex}")
            print(f"  [{cv.name}] zk end-to-end (hiding sponge → h(X) coeffs) byte-matches arkworks")


if __name__ == "__main__":
    absltest.main()
