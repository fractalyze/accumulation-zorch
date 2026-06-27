"""Slice-2a byte-match: the IPA-PC accumulation prove's new-accumulator *instance*
(combined commitment, new opening point, combined evaluation) vs the real arkworks
`AtomicASForInnerProductArgPC::prove` (no-zk), over BOTH Pasta cycle curves.

Replays the inputs dumped from the crate's real AS prove
(`cargo run --example dump_ipa_as -- <curve>`) — each input's commitment, point,
evaluation, and IPA proof — runs the Slice-1 succinct check on each, then the AS
combine + new-challenge + combined-evaluation, and asserts the resulting
accumulator instance fields match arkworks byte-for-byte. The accumulator's IPA
*proof* (`l_vec`/`r_vec`/`final_comm_key`/`c`, produced by the `IpaPC::open` fold)
is Slice 2b and is not checked here. The Poseidon ARK constants come from the
per-curve sponge fixture, as in the other byte-match tests.

Running the SAME curve-generic port against the Vesta fixture is the
curve-generic gate.

Run (from the repo's `python/` dir, in the accumulation-zorch venv):

    JAX_PLATFORMS=cpu PYTHONPATH=. \
      python accumulation_zorch/testing/ipa_as_test.py
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

from absl.testing import absltest

from accumulation_zorch import curve, ipa_pc, ipa_pc_as, sponge

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

_CURVES = [
    (curve.PALLAS, _TESTDATA / "ipa_as_fixtures.json", _TESTDATA / "sponge_fixtures.json"),
    (curve.VESTA, _TESTDATA / "ipa_as_vesta_fixtures.json", _TESTDATA / "sponge_vesta_fixtures.json"),
]


def _fr(h: str) -> int:
    return int.from_bytes(bytes.fromhex(h), "little")


def _point(cv: curve.Curve, p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


class _Input(NamedTuple):
    """One parsed input instance — the fields the succinct check + AS combine read."""
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


class IpaAsTest(absltest.TestCase):
    def test_as_prove_no_zk_accumulator_instance_matches_arkworks(self) -> None:
        for cv, as_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(as_fixture.read_text())
            inputs = [_parse_input(cv, inp) for inp in d["inputs"]]

            succinct_checks = [ipa_pc_as.succinct_check_input(cv, params, inp) for inp in inputs]
            got = ipa_pc_as.prove_no_zk_instance(cv, params, succinct_checks)

            acc = d["accumulator"]
            got_comm = curve.point_to_bytes(cv, got.commitment).hex()
            want_comm = curve.point_to_bytes(cv, _point(cv, acc["commitment"])).hex()
            self.assertEqual(got_comm, want_comm, f"[{cv.name}] combined commitment: {got_comm} != {want_comm}")

            got_point = cv.fr(got.point).tobytes().hex()
            self.assertEqual(got_point, acc["point"], f"[{cv.name}] new point: {got_point} != {acc['point']}")

            got_eval = cv.fr(got.evaluation).tobytes().hex()
            self.assertEqual(got_eval, acc["evaluation"], (
                f"[{cv.name}] combined evaluation: {got_eval} != {acc['evaluation']}"))

            print(f"  [{cv.name}] AS no-zk accumulator instance (commitment, point, evaluation) "
                  f"byte-matches arkworks ({d['num_inputs']} inputs)")

    def test_as_prove_no_zk_full_accumulator_matches_arkworks(self) -> None:
        """Slice 2b: the FULL new accumulator, including the IPA opening proof
        (`l_vec`/`r_vec`/`final_comm_key`/`c`) produced by the `IpaPC::open` fold over
        the combined check polynomial."""
        for cv, as_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(as_fixture.read_text())
            inputs = [_parse_input(cv, inp) for inp in d["inputs"]]
            svk_h = _point(cv, d["h"])
            generators = [_point(cv, g) for g in d["generators"]]

            succinct_checks = [ipa_pc_as.succinct_check_input(cv, params, inp) for inp in inputs]
            acc = ipa_pc_as.prove_no_zk_accumulator(cv, params, svk_h, generators, succinct_checks)
            want = d["accumulator"]

            def _pt(p: Any) -> str:
                return curve.point_to_bytes(cv, p).hex()

            assert _pt(acc.commitment) == _pt(_point(cv, want["commitment"])), f"[{cv.name}] commitment"
            assert cv.fr(acc.point).tobytes().hex() == want["point"], f"[{cv.name}] point"
            assert cv.fr(acc.evaluation).tobytes().hex() == want["evaluation"], f"[{cv.name}] evaluation"

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

            print(f"  [{cv.name}] full AS no-zk accumulator (instance + IPA proof: "
                  f"{len(acc.ipa_proof.l_vec)} fold rounds) byte-matches arkworks")

    def test_decide_no_zk_size_d_msm_matches_final_comm_key(self) -> None:
        """Slice 3 (Decide): the decider's size-`d` MSM
        `final_key = Σ generators_i · compute_coeffs(succinct_check(acc))_i` must equal
        the accumulator's `final_comm_key`. This MSM is the IPA accumulation's
        GPU-value work (the fused-core target). Runs on the golden accumulator."""
        for cv, as_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(as_fixture.read_text())
            generators = [_point(cv, g) for g in d["generators"]]
            acc = _parse_input(cv, d["accumulator"])

            final_key = ipa_pc_as.decide_final_key(cv, params, generators, acc)
            got = curve.point_to_bytes(cv, final_key).hex()
            want = curve.point_to_bytes(cv, acc.final_comm_key).hex()
            self.assertEqual(got, want, f"[{cv.name}] decider size-d MSM != final_comm_key: {got} != {want}")
            print(f"  [{cv.name}] decider size-d MSM (= MSM(generators, h(X) coeffs)) "
                  f"byte-matches the accumulator's final_comm_key")

    def test_decider_coeffs_fixture_matches_port(self) -> None:
        """The fixture's arkworks-golden `decider_coeffs` — the scalar input fed to the
        Slice-4 fused GPU decider MSM — are exactly the jax port's
        `compute_coeffs(succinct_check(accumulator))`. This ties the GPU core's runtime
        scalar input to the byte-matched CPU port, so the GPU byte-match
        (`MSM(generators, decider_coeffs) == final_comm_key`) exercises the port's
        coefficients, not just arkworks'."""
        for cv, as_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(as_fixture.read_text())
            acc = _parse_input(cv, d["accumulator"])

            check_poly = ipa_pc.succinct_check_challenges(
                cv, params, acc.commitment, acc.point, acc.value, acc.l_vec, acc.r_vec)
            coeffs = ipa_pc.compute_coeffs(cv, check_poly)
            got = [cv.fr(c).tobytes().hex() for c in coeffs]
            want = d["decider_coeffs"]
            self.assertEqual(got, want, f"[{cv.name}] decider_coeffs: port {got} != fixture {want}")
            print(f"  [{cv.name}] fixture decider_coeffs ({len(want)}) match the port's "
                  f"compute_coeffs(succinct_check(acc)) — the fused GPU MSM's scalar input")


if __name__ == "__main__":
    absltest.main()
