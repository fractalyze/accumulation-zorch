"""Slice-1 byte-match: IPA-PC `succinct_check` round challenges + `h(X)` check-poly
coefficients vs the real arkworks `ipa_pc::succinct_check`, over BOTH Pasta cycle
curves (Pallas and Vesta).

Replays an IPA opening dumped from the crate's real `InnerProductArgPC`
(`cargo run --example dump_ipa -- <curve>`: the commitment, opening point +
evaluation, and the proof's `l_vec`/`r_vec`) through the ported succinct-check
Fiat-Shamir, and asserts the derived `SuccinctCheckPolynomial` round challenges,
its dense `compute_coeffs` expansion, and its evaluation at the point all match
arkworks byte-for-byte. The Poseidon ARK constants come from the per-curve sponge
fixture (`sponge_fixtures.json` / `sponge_vesta_fixtures.json`), the same way the
AS byte-match tests source them.

The three checks localize a divergence: the round-challenge check isolates the
domain-separated sponge + absorb order; the check-poly check (fed the golden
challenges) isolates the polynomial expansion; the end-to-end check confirms the
two compose.

Running the SAME curve-generic port against the Vesta fixture — different
`g1`/`fr`/`fq` dtypes, different base field for the sponge — is the curve-generic
gate (the port is genuinely generic, not just non-breaking on Pallas).

Run (from the repo's `python/` dir, in the accumulation-zorch venv):

    JAX_PLATFORMS=cpu PYTHONPATH=. \
      python accumulation_zorch/testing/ipa_test.py
"""

import json
from pathlib import Path
from typing import Any

from absl.testing import absltest

from accumulation_zorch import curve, ipa_pc, sponge

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

# (curve, IPA fixture, sponge fixture for ARK) per cycle curve — one port, two
# curves. The ARK constants reduce mod the curve's base field, so each curve
# pulls its own sponge fixture.
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


def _load(cv: curve.Curve, ipa_fixture: Path) -> Any:
    d = json.loads(ipa_fixture.read_text())
    commitment = _point(cv, d["commitment"])
    l_vec = [_point(cv, p) for p in d["l_vec"]]
    r_vec = [_point(cv, p) for p in d["r_vec"]]
    return d, commitment, _fr(d["point"]), _fr(d["evaluation"]), l_vec, r_vec


class IpaTest(absltest.TestCase):
    def test_succinct_check_round_challenges_match_arkworks(self) -> None:
        for cv, ipa_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d, commitment, point, value, l_vec, r_vec = _load(cv, ipa_fixture)
            got = ipa_pc.succinct_check_challenges(cv, params, commitment, point, value, l_vec, r_vec)
            self.assertEqual(len(got), len(d["round_challenges"]), f"[{cv.name}] challenge count")
            for i, want_hex in enumerate(d["round_challenges"]):
                got_hex = cv.fr(got[i]).tobytes().hex()
                self.assertEqual(got_hex, want_hex, (
                    f"[{cv.name}] round challenge[{i}]: {got_hex} != {want_hex}"))
            print(f"  [{cv.name}] succinct_check round challenges byte-match arkworks "
                  f"({len(got)} rounds)")

    def test_check_poly_coeffs_and_eval_match_arkworks(self) -> None:
        """`h(X)` dense coefficients + the evaluation at the point, fed the golden
        round challenges — isolates the polynomial expansion from the sponge."""
        for cv, ipa_fixture, sponge_fixture in _CURVES:
            d, _, point, _, _, _ = _load(cv, ipa_fixture)
            challenges = [_fr(h) for h in d["round_challenges"]]

            coeffs = ipa_pc.compute_coeffs(cv, challenges)
            self.assertEqual(len(coeffs), len(d["coeffs"]), f"[{cv.name}] coeff count")
            for i, want_hex in enumerate(d["coeffs"]):
                got_hex = cv.fr(coeffs[i]).tobytes().hex()
                self.assertEqual(got_hex, want_hex, f"[{cv.name}] h(X) coeff[{i}]: {got_hex} != {want_hex}")

            got_eval = cv.fr(ipa_pc.evaluate(cv, challenges, point)).tobytes().hex()
            self.assertEqual(got_eval, d["eval_at_point"], (
                f"[{cv.name}] h(point): {got_eval} != {d['eval_at_point']}"))
            print(f"  [{cv.name}] h(X) compute_coeffs ({len(coeffs)} coeffs) + evaluate "
                  f"byte-match arkworks")

    def test_succinct_check_end_to_end_matches_arkworks(self) -> None:
        """The full Slice-1 path: derive the challenges from the sponge, then expand
        `h(X)` from them — the integrated succinct-check the later decider MSM consumes."""
        for cv, ipa_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d, commitment, point, value, l_vec, r_vec = _load(cv, ipa_fixture)
            challenges = ipa_pc.succinct_check_challenges(cv, params, commitment, point, value, l_vec, r_vec)
            coeffs = ipa_pc.compute_coeffs(cv, challenges)
            for i, want_hex in enumerate(d["coeffs"]):
                got_hex = cv.fr(coeffs[i]).tobytes().hex()
                self.assertEqual(got_hex, want_hex, (
                    f"[{cv.name}] end-to-end h(X) coeff[{i}]: {got_hex} != {want_hex}"))
            print(f"  [{cv.name}] end-to-end (sponge → h(X) coeffs) byte-matches arkworks")


if __name__ == "__main__":
    absltest.main()
