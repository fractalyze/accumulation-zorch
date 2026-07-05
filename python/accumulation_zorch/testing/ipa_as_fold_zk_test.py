"""Byte-match: the IPA-PC accumulation **fold** (zk/hiding) — one no-zk input
folded INTO a prior *hiding* accumulator (`old_accumulators = [acc_prev]`) — vs the
real arkworks `AtomicASForInnerProductArgPC::prove` (`MakeZK::Enabled`), over BOTH
Pasta cycle curves. The zk twin of `ipa_as_fold_test.py`, and the fold twin of
`ipa_as_zk_test.py`.

Replays the fixture from the crate's two-round zk fold
(`cargo run --example dump_ipa_as_fold_zk -- <curve>`): the no-zk new input, the
hiding prior accumulator `acc_prev`, the fold's AS randomness
(`random_linear_polynomial` + commitment + `commitment_randomness`), and the fold's
hiding `IpaPC::open` randomness (`hiding_polynomial` / `hiding_rand`). The port
succinct-checks the input (no-zk) then `acc_prev` (zk — folding the hiding seed),
runs the zk prove over `[input, acc_prev]`, and asserts the golden folded
accumulator (instance + hiding IPA proof) matches arkworks byte-for-byte.

The new logic over the no-zk fold: `acc_prev` carries a hiding IPA opening, so its
succinct check is the zk path (`succinct_check_input` with `s`), while the new input
stays no-zk.

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:ipa_as_fold_zk_test
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

from absl.testing import absltest

from accumulation_zorch import curve, ipa_pc, ipa_pc_as, sponge

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

_CURVES = [
    (curve.PALLAS, _TESTDATA / "ipa_as_fold_zk_fixtures.json", _TESTDATA / "sponge_fixtures.json"),
    (curve.VESTA, _TESTDATA / "ipa_as_fold_zk_vesta_fixtures.json", _TESTDATA / "sponge_vesta_fixtures.json"),
]


def _fr(h: str) -> int:
    return int.from_bytes(bytes.fromhex(h), "little")


def _point(cv: curve.Curve, p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


class _Input(NamedTuple):
    """A parsed input / accumulator instance; `hiding_comm`/`rand` present only for a
    hiding (zk-prove) accumulator."""
    commitment: Any
    point: int
    value: int
    l_vec: list
    r_vec: list
    final_comm_key: Any
    hiding_comm: Any = None
    rand: int = 0


def _parse_input(cv: curve.Curve, d: Any) -> _Input:
    return _Input(
        commitment=_point(cv, d["commitment"]),
        point=_fr(d["point"]),
        value=_fr(d["evaluation"]),
        l_vec=[_point(cv, p) for p in d["l_vec"]],
        r_vec=[_point(cv, p) for p in d["r_vec"]],
        final_comm_key=_point(cv, d["final_comm_key"]),
        hiding_comm=_point(cv, d["hiding_comm"]) if "hiding_comm" in d else None,
        rand=_fr(d["rand"]) if "rand" in d else 0,
    )


def _params(cv: curve.Curve, sponge_fixture: Path) -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(sponge_fixture.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


class IpaAsFoldZkTest(absltest.TestCase):
    def test_fold_zk_full_accumulator_matches_arkworks(self) -> None:
        """The FULL folded accumulator — instance (randomized combined commitment,
        new point, combined evaluation) + the hiding IPA opening proof
        (`l_vec`/`r_vec`/`final_comm_key`/`c` + `hiding_comm`/`rand`) — folding one
        no-zk input into the hiding `acc_prev`."""
        for cv, fold_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(fold_fixture.read_text())
            new_input = _parse_input(cv, d["input"])
            acc_prev = _parse_input(cv, d["acc_prev"])
            svk_h = _point(cv, d["h"])
            s = _point(cv, d["s"])
            generators = [_point(cv, g) for g in d["generators"]]
            rlp_coeffs = [_fr(h) for h in d["random_linear_polynomial"]]
            rlp_commitment = _point(cv, d["random_linear_polynomial_commitment"])
            commitment_randomness = _fr(d["commitment_randomness"])
            hiding_poly = [_fr(h) for h in d["hiding_polynomial"]]
            hiding_rand = _fr(d["hiding_rand"])

            acc = ipa_pc_as.prove_fold(
                cv, params, svk_h, generators, [new_input], [acc_prev],
                ipa_pc_as.Randomness(rlp_coeffs, rlp_commitment, commitment_randomness), s,
                hiding_poly, hiding_rand)
            want = d["accumulator"]

            def _pt(p: Any) -> str:
                return curve.point_to_bytes(cv, p).hex()

            self.assertEqual(_pt(acc.commitment), _pt(_point(cv, want["commitment"])), f"[{cv.name}] commitment")
            self.assertEqual(cv.fr(acc.point).tobytes().hex(), want["point"], f"[{cv.name}] point")
            self.assertEqual(cv.fr(acc.evaluation).tobytes().hex(), want["evaluation"], f"[{cv.name}] evaluation")
            for i, want_l in enumerate(want["l_vec"]):
                got, wnt = _pt(acc.ipa_proof.l_vec[i]), _pt(_point(cv, want_l))
                self.assertEqual(got, wnt, f"[{cv.name}] ipa_proof.l_vec[{i}]: {got} != {wnt}")
            for i, want_r in enumerate(want["r_vec"]):
                got, wnt = _pt(acc.ipa_proof.r_vec[i]), _pt(_point(cv, want_r))
                self.assertEqual(got, wnt, f"[{cv.name}] ipa_proof.r_vec[{i}]: {got} != {wnt}")
            self.assertEqual(_pt(acc.ipa_proof.final_comm_key), _pt(_point(cv, want["final_comm_key"])), f"[{cv.name}] final_comm_key")
            self.assertEqual(cv.fr(acc.ipa_proof.c).tobytes().hex(), want["c"], f"[{cv.name}] c")
            self.assertEqual(_pt(acc.ipa_proof.hiding_comm), _pt(_point(cv, want["hiding_comm"])), f"[{cv.name}] hiding_comm")
            self.assertEqual(cv.fr(acc.ipa_proof.rand).tobytes().hex(), want["rand"], f"[{cv.name}] rand")
            print(f"  [{cv.name}] zk FOLD (1 no-zk input into hiding acc_prev, "
                  f"{len(acc.ipa_proof.l_vec)} fold rounds) byte-matches arkworks")

    def test_fold_zk_decide_size_d_msm_matches_final_comm_key(self) -> None:
        """The zk decider's size-`d` MSM on the FOLDED (hiding) accumulator:
        `final_key = Σ generators_i · compute_coeffs(zk_succinct_check(acc))_i` must
        equal its `final_comm_key`."""
        for cv, fold_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(fold_fixture.read_text())
            generators = [_point(cv, g) for g in d["generators"]]
            s = _point(cv, d["s"])
            acc = _parse_input(cv, d["accumulator"])

            final_key = ipa_pc_as.decide_final_key(cv, params, generators, acc, s)
            got = curve.point_to_bytes(cv, final_key).hex()
            want = curve.point_to_bytes(cv, acc.final_comm_key).hex()
            self.assertEqual(got, want, f"[{cv.name}] folded zk decider size-d MSM != final_comm_key: {got} != {want}")
            print(f"  [{cv.name}] folded (hiding) accumulator zk decider size-d MSM byte-matches its final_comm_key")

    def test_fold_zk_decider_coeffs_fixture_matches_port(self) -> None:
        """The fixture's arkworks-golden `decider_coeffs` equal the port's
        `compute_coeffs(zk_succinct_check(folded accumulator))`."""
        for cv, fold_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(fold_fixture.read_text())
            s = _point(cv, d["s"])
            acc = _parse_input(cv, d["accumulator"])

            check_poly = ipa_pc.succinct_check_challenges_zk(
                cv, params, acc.commitment, acc.point, acc.value, acc.l_vec, acc.r_vec,
                s, acc.hiding_comm, acc.rand)
            coeffs = ipa_pc.compute_coeffs(cv, check_poly)
            got = [c.tobytes().hex() for c in coeffs]
            want = d["decider_coeffs"]
            self.assertEqual(got, want, f"[{cv.name}] zk decider_coeffs: port != fixture")
            print(f"  [{cv.name}] fixture decider_coeffs ({len(want)}) match the port's zk "
                  f"compute_coeffs(succinct_check(folded acc))")


if __name__ == "__main__":
    absltest.main()
