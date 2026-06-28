"""Byte-match: the IPA-PC accumulation **fold** (no-zk) — one input folded INTO a
prior accumulator (`old_accumulators = [acc_prev]`) — vs the real arkworks
`AtomicASForInnerProductArgPC::prove`, over BOTH Pasta cycle curves. The fold twin
of `ipa_as_test.py` (which accumulates inputs with no old accumulators).

Replays the fixture dumped from the crate's two-round fold
(`cargo run --example dump_ipa_as_fold -- <curve>`): the new input, the prior
accumulator `acc_prev`, and the golden folded accumulator. The port succinct-checks
the input then `acc_prev` (inputs first, then accumulators — the
`succinct_check_inputs_and_accumulators` order), folds via the same combine +
`IpaPC::open` machinery, and asserts the resulting accumulator (instance + IPA
proof) matches arkworks byte-for-byte. The crux: `acc_prev` is an `InputInstance`
of the same shape as an input, so the fold is the no-fold prove fed
`[input, acc_prev]`.

Running the SAME curve-generic port against the Vesta fixture is the curve-generic
gate.

Run (from the repo's `python/` dir, in the accumulation-zorch venv):

    JAX_PLATFORMS=cpu PYTHONPATH=python \
      python python/accumulation_zorch/testing/ipa_as_fold_test.py
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

from absl.testing import absltest

from accumulation_zorch import curve, ipa_pc, ipa_pc_as, sponge

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

_CURVES = [
    (curve.PALLAS, _TESTDATA / "ipa_as_fold_fixtures.json", _TESTDATA / "sponge_fixtures.json"),
    (curve.VESTA, _TESTDATA / "ipa_as_fold_vesta_fixtures.json", _TESTDATA / "sponge_vesta_fixtures.json"),
]


def _fr(h: str) -> int:
    return int.from_bytes(bytes.fromhex(h), "little")


def _point(cv: curve.Curve, p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


class _Input(NamedTuple):
    """One parsed input / accumulator instance — the fields the succinct check + AS
    combine read."""
    commitment: Any
    point: int
    value: int
    l_vec: list
    r_vec: list
    final_comm_key: Any


def _parse_input(cv: curve.Curve, d: Any) -> _Input:
    return _Input(
        commitment=_point(cv, d["commitment"]),
        point=_fr(d["point"]),
        value=_fr(d["evaluation"]),
        l_vec=[_point(cv, p) for p in d["l_vec"]],
        r_vec=[_point(cv, p) for p in d["r_vec"]],
        final_comm_key=_point(cv, d["final_comm_key"]),
    )


def _params(cv: curve.Curve, sponge_fixture: Path) -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(sponge_fixture.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


class IpaAsFoldTest(absltest.TestCase):
    def test_fold_no_zk_full_accumulator_matches_arkworks(self) -> None:
        """The FULL folded accumulator — instance (combined commitment, new point,
        combined evaluation) + the IPA opening proof (`l_vec`/`r_vec`/
        `final_comm_key`/`c`) — folding one input into `acc_prev`."""
        for cv, fold_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(fold_fixture.read_text())
            new_input = _parse_input(cv, d["input"])
            acc_prev = _parse_input(cv, d["acc_prev"])
            svk_h = _point(cv, d["h"])
            generators = [_point(cv, g) for g in d["generators"]]

            acc = ipa_pc_as.prove_no_zk_fold(
                cv, params, svk_h, generators, [new_input], [acc_prev])
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
            got_fck, wnt_fck = _pt(acc.ipa_proof.final_comm_key), _pt(_point(cv, want["final_comm_key"]))
            self.assertEqual(got_fck, wnt_fck, f"[{cv.name}] ipa_proof.final_comm_key: {got_fck} != {wnt_fck}")
            got_c = cv.fr(acc.ipa_proof.c).tobytes().hex()
            self.assertEqual(got_c, want["c"], f"[{cv.name}] ipa_proof.c: {got_c} != {want['c']}")

            print(f"  [{cv.name}] no-zk FOLD (1 input into acc_prev, "
                  f"{len(acc.ipa_proof.l_vec)} fold rounds) byte-matches arkworks")

    def test_fold_decide_size_d_msm_matches_final_comm_key(self) -> None:
        """The decider's size-`d` MSM on the FOLDED accumulator:
        `final_key = Σ generators_i · compute_coeffs(succinct_check(acc))_i` must
        equal the folded accumulator's `final_comm_key` — the fused GPU fold core's
        decider gate."""
        for cv, fold_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(fold_fixture.read_text())
            generators = [_point(cv, g) for g in d["generators"]]
            acc = _parse_input(cv, d["accumulator"])

            final_key = ipa_pc_as.decide_final_key(cv, params, generators, acc)
            got = curve.point_to_bytes(cv, final_key).hex()
            want = curve.point_to_bytes(cv, acc.final_comm_key).hex()
            self.assertEqual(got, want, f"[{cv.name}] folded decider size-d MSM != final_comm_key: {got} != {want}")
            print(f"  [{cv.name}] folded accumulator decider size-d MSM byte-matches its final_comm_key")

    def test_fold_decider_coeffs_fixture_matches_port(self) -> None:
        """The fixture's arkworks-golden `decider_coeffs` (the fused GPU fold
        decider MSM's scalar input) equal the port's
        `compute_coeffs(succinct_check(folded accumulator))`."""
        for cv, fold_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(fold_fixture.read_text())
            acc = _parse_input(cv, d["accumulator"])

            check_poly = ipa_pc.succinct_check_challenges(
                cv, params, acc.commitment, acc.point, acc.value, acc.l_vec, acc.r_vec)
            coeffs = ipa_pc.compute_coeffs(cv, check_poly)
            got = [cv.fr(c).tobytes().hex() for c in coeffs]
            want = d["decider_coeffs"]
            self.assertEqual(got, want, f"[{cv.name}] decider_coeffs: port {got} != fixture {want}")
            print(f"  [{cv.name}] fixture decider_coeffs ({len(want)}) match the port's "
                  f"compute_coeffs(succinct_check(folded acc))")


if __name__ == "__main__":
    absltest.main()
