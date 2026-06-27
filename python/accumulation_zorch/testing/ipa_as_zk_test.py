"""Slice-5b byte-match: the IPA-PC accumulation **zk** prove's new-accumulator
*instance* (randomized combined commitment, new opening point, combined
evaluation) vs the real arkworks `AtomicASForInnerProductArgPC::prove`
(`MakeZK::Enabled`), over BOTH Pasta cycle curves — the zk twin of the Slice-2a
instance check in `ipa_as_test.py`.

Replays the no-zk inputs + the AS proof's `random_linear_polynomial` (its two
coefficients), its commitment, and `commitment_randomness` (recovered from the
serialized `Randomness` in `dump_ipa_as_zk.rs`) through the ported zk combine, and
asserts the accumulator instance matches arkworks byte-for-byte. The zk additions
exercised: the random-linear-polynomial absorb in the lc-challenge sponge, the
combined commitment seeded from `rlp_commitment` and blinded by
`s·commitment_randomness`, the `Some(rlp_coeffs)` absorb in `compute_new_challenge`,
and the random poly's `rlp(point)` term in the combined evaluation. The
accumulator's hiding IPA opening proof (the IPA open's replayed hiding polynomial)
is a later sub-step and not checked here.

Run (from the repo's `python/` dir, in the accumulation-zorch venv):

    JAX_PLATFORMS=cpu PYTHONPATH=. \
      python accumulation_zorch/testing/ipa_as_zk_test.py
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

from absl.testing import absltest

from accumulation_zorch import curve, ipa_pc_as, sponge

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

_CURVES = [
    (curve.PALLAS, _TESTDATA / "ipa_as_zk_fixtures.json", _TESTDATA / "sponge_fixtures.json"),
    (curve.VESTA, _TESTDATA / "ipa_as_zk_vesta_fixtures.json", _TESTDATA / "sponge_vesta_fixtures.json"),
]


def _fr(h: str) -> int:
    return int.from_bytes(bytes.fromhex(h), "little")


def _point(cv: curve.Curve, p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


class _Input(NamedTuple):
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


class IpaAsZkTest(absltest.TestCase):
    def test_as_prove_zk_accumulator_instance_matches_arkworks(self) -> None:
        for cv, as_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(as_fixture.read_text())
            inputs = [_parse_input(cv, inp) for inp in d["inputs"]]
            rlp_coeffs = [_fr(h) for h in d["random_linear_polynomial"]]
            rlp_commitment = _point(cv, d["random_linear_polynomial_commitment"])
            s = _point(cv, d["s"])
            commitment_randomness = _fr(d["commitment_randomness"])

            # No-zk inputs ⇒ no-zk succinct check; the AS layer is the zk part.
            succinct_checks = [ipa_pc_as.succinct_check_input(cv, params, inp) for inp in inputs]
            got = ipa_pc_as.prove_zk_instance(
                cv, params, succinct_checks, rlp_coeffs, rlp_commitment, s, commitment_randomness)

            acc = d["accumulator"]
            got_comm = curve.point_to_bytes(cv, got.commitment).hex()
            want_comm = curve.point_to_bytes(cv, _point(cv, acc["commitment"])).hex()
            self.assertEqual(got_comm, want_comm, f"[{cv.name}] randomized combined commitment: {got_comm} != {want_comm}")

            got_point = cv.fr(got.point).tobytes().hex()
            self.assertEqual(got_point, acc["point"], f"[{cv.name}] new point: {got_point} != {acc['point']}")

            got_eval = cv.fr(got.evaluation).tobytes().hex()
            self.assertEqual(got_eval, acc["evaluation"], f"[{cv.name}] combined evaluation: {got_eval} != {acc['evaluation']}")

            print(f"  [{cv.name}] zk AS accumulator instance (randomized commitment, point, evaluation) "
                  f"byte-matches arkworks ({d['num_inputs']} inputs)")

    def test_as_prove_zk_full_accumulator_matches_arkworks(self) -> None:
        """Slice 5c: the FULL new accumulator, including the hiding IPA opening proof
        (`l_vec`/`r_vec`/`final_comm_key`/`c` + `hiding_comm`/`rand`) produced by the
        zk `IpaPC::open` fold over the combined check polynomial `rlp(X) + Σ lc_j·h_j(X)`.
        The open's hiding polynomial + blinder are the fixture's replayed
        `hiding_polynomial` / `hiding_rand`."""
        for cv, as_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(as_fixture.read_text())
            inputs = [_parse_input(cv, inp) for inp in d["inputs"]]
            svk_h = _point(cv, d["h"])
            s = _point(cv, d["s"])
            generators = [_point(cv, g) for g in d["generators"]]
            rlp_coeffs = [_fr(h) for h in d["random_linear_polynomial"]]
            rlp_commitment = _point(cv, d["random_linear_polynomial_commitment"])
            commitment_randomness = _fr(d["commitment_randomness"])
            hiding_poly = [_fr(h) for h in d["hiding_polynomial"]]
            hiding_rand = _fr(d["hiding_rand"])

            succinct_checks = [ipa_pc_as.succinct_check_input(cv, params, inp) for inp in inputs]
            acc = ipa_pc_as.prove_zk_accumulator(
                cv, params, svk_h, s, generators, succinct_checks, rlp_coeffs, rlp_commitment,
                commitment_randomness, hiding_poly, hiding_rand)
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
            assert _pt(acc.ipa_proof.final_comm_key) == _pt(_point(cv, want["final_comm_key"])), f"[{cv.name}] final_comm_key"
            assert cv.fr(acc.ipa_proof.c).tobytes().hex() == want["c"], f"[{cv.name}] c"
            assert _pt(acc.ipa_proof.hiding_comm) == _pt(_point(cv, want["hiding_comm"])), f"[{cv.name}] hiding_comm"
            assert cv.fr(acc.ipa_proof.rand).tobytes().hex() == want["rand"], f"[{cv.name}] rand"
            print(f"  [{cv.name}] full zk AS accumulator (instance + hiding IPA proof: "
                  f"{len(acc.ipa_proof.l_vec)} fold rounds) byte-matches arkworks")

    def test_decide_zk_size_d_msm_matches_final_comm_key(self) -> None:
        """Slice 5d (zk Decide): the decider's size-`d` MSM
        `final_key = Σ generators_i · compute_coeffs(zk_succinct_check(acc))_i` must
        equal the (hiding) accumulator's `final_comm_key` — the zk twin of the no-zk
        decide check, and the fused zk GPU core's target."""
        for cv, as_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(as_fixture.read_text())
            generators = [_point(cv, g) for g in d["generators"]]
            s = _point(cv, d["s"])
            acc = _parse_input(cv, d["accumulator"])

            final_key = ipa_pc_as.decide_final_key_zk(cv, params, generators, acc, s)
            got = curve.point_to_bytes(cv, final_key).hex()
            want = curve.point_to_bytes(cv, acc.final_comm_key).hex()
            self.assertEqual(got, want, f"[{cv.name}] zk decider size-d MSM != final_comm_key: {got} != {want}")
            print(f"  [{cv.name}] zk decider size-d MSM (= MSM(generators, h(X) coeffs)) "
                  f"byte-matches the accumulator's final_comm_key")

    def test_decider_coeffs_fixture_matches_port_zk(self) -> None:
        """The fixture's arkworks-golden `decider_coeffs` — the scalar input fed to the
        Slice-5e fused GPU decider MSM — are exactly the port's
        `compute_coeffs(zk_succinct_check(accumulator))`, tying the GPU core's runtime
        input to the byte-matched CPU port."""
        from accumulation_zorch import ipa_pc
        for cv, as_fixture, sponge_fixture in _CURVES:
            params = _params(cv, sponge_fixture)
            d = json.loads(as_fixture.read_text())
            s = _point(cv, d["s"])
            acc = _parse_input(cv, d["accumulator"])

            check_poly = ipa_pc.succinct_check_challenges_zk(
                cv, params, acc.commitment, acc.point, acc.value, acc.l_vec, acc.r_vec,
                s, acc.hiding_comm, acc.rand)
            coeffs = ipa_pc.compute_coeffs(cv, check_poly)
            got = [cv.fr(c).tobytes().hex() for c in coeffs]
            want = d["decider_coeffs"]
            self.assertEqual(got, want, f"[{cv.name}] zk decider_coeffs: port != fixture")
            print(f"  [{cv.name}] fixture decider_coeffs ({len(want)}) match the port's zk "
                  f"compute_coeffs(succinct_check(acc)) — the fused GPU MSM's scalar input")


if __name__ == "__main__":
    absltest.main()
