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


def test_as_prove_zk_accumulator_instance_matches_arkworks() -> None:
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
        assert got_comm == want_comm, f"[{cv.name}] randomized combined commitment: {got_comm} != {want_comm}"

        got_point = cv.fr(got.point).tobytes().hex()
        assert got_point == acc["point"], f"[{cv.name}] new point: {got_point} != {acc['point']}"

        got_eval = cv.fr(got.evaluation).tobytes().hex()
        assert got_eval == acc["evaluation"], f"[{cv.name}] combined evaluation: {got_eval} != {acc['evaluation']}"

        print(f"  [{cv.name}] zk AS accumulator instance (randomized commitment, point, evaluation) "
              f"byte-matches arkworks ({d['num_inputs']} inputs)")


def main() -> None:
    print("slice-5b IPA-PC zk accumulation prove (instance) byte-match (Pallas + Vesta):")
    test_as_prove_zk_accumulator_instance_matches_arkworks()
    print("ALL SLICE-5b IPA-PC-AS ZK INSTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
