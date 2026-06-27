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

from accumulation_zorch import curve, ipa_pc_as, sponge

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


def test_as_prove_no_zk_accumulator_instance_matches_arkworks() -> None:
    for cv, as_fixture, sponge_fixture in _CURVES:
        params = _params(cv, sponge_fixture)
        d = json.loads(as_fixture.read_text())
        inputs = [_parse_input(cv, inp) for inp in d["inputs"]]

        succinct_checks = [ipa_pc_as.succinct_check_input(cv, params, inp) for inp in inputs]
        got = ipa_pc_as.prove_no_zk_instance(cv, params, succinct_checks)

        acc = d["accumulator"]
        got_comm = curve.point_to_bytes(cv, got.commitment).hex()
        want_comm = curve.point_to_bytes(cv, _point(cv, acc["commitment"])).hex()
        assert got_comm == want_comm, f"[{cv.name}] combined commitment: {got_comm} != {want_comm}"

        got_point = cv.fr(got.point).tobytes().hex()
        assert got_point == acc["point"], f"[{cv.name}] new point: {got_point} != {acc['point']}"

        got_eval = cv.fr(got.evaluation).tobytes().hex()
        assert got_eval == acc["evaluation"], (
            f"[{cv.name}] combined evaluation: {got_eval} != {acc['evaluation']}")

        print(f"  [{cv.name}] AS no-zk accumulator instance (commitment, point, evaluation) "
              f"byte-matches arkworks ({d['num_inputs']} inputs)")


def test_as_prove_no_zk_full_accumulator_matches_arkworks() -> None:
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
            assert got == wnt, f"[{cv.name}] ipa_proof.l_vec[{i}]: {got} != {wnt}"
        for i, want_r in enumerate(want["r_vec"]):
            got, wnt = _pt(acc.ipa_proof.r_vec[i]), _pt(_point(cv, want_r))
            assert got == wnt, f"[{cv.name}] ipa_proof.r_vec[{i}]: {got} != {wnt}"
        got_fck, wnt_fck = _pt(acc.ipa_proof.final_comm_key), _pt(_point(cv, want["final_comm_key"]))
        assert got_fck == wnt_fck, f"[{cv.name}] ipa_proof.final_comm_key: {got_fck} != {wnt_fck}"
        got_c = cv.fr(acc.ipa_proof.c).tobytes().hex()
        assert got_c == want["c"], f"[{cv.name}] ipa_proof.c: {got_c} != {want['c']}"

        print(f"  [{cv.name}] full AS no-zk accumulator (instance + IPA proof: "
              f"{len(acc.ipa_proof.l_vec)} fold rounds) byte-matches arkworks")


def main() -> None:
    print("slice-2 IPA-PC accumulation prove byte-match (Pallas + Vesta):")
    test_as_prove_no_zk_accumulator_instance_matches_arkworks()
    test_as_prove_no_zk_full_accumulator_matches_arkworks()
    print("ALL SLICE-2 IPA-PC-AS CHECKS PASSED")


if __name__ == "__main__":
    main()
