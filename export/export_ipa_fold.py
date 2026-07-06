"""Export the IPA-PC accumulation **fold** open to one StableHLO ``.mlirbc`` — the
Slice-3 fused GPU fold core, the fold twin of ``export_ipa.py`` (the decider MSM).

Unlike the decider (one host-fed MSM), the fold's ``IpaPC::open`` is **sequential**:
each round's Fiat-Shamir challenge is squeezed from that round's ``L_j``/``R_j`` MSM
outputs, so there is no host-challenge shortcut — the fused core must run the
Poseidon sponge on-device, interleaved with the per-round MSMs. That whole
on-device open (zorch's ``_open_one`` ``lax.scan`` fold + the ``final_comm_key``
MSM, driven by the arkworks-faithful ``ipa_challenger``) is
``ipa_open.build_open_no_zk_core``; this script bakes one fixture's combined check
polynomial into it and lowers it, mirroring ``export_fold_zk.py``'s baked-instance
core (``r1cs_nark_as._build_zk_fold_core``).

The combine that produces the ``(combined_commitment, point, coeffs)`` the open
runs on is cheap host field/sponge (already byte-matched on CPU, ``ipa_as_fold_test``),
so it runs here on the host — exactly as ``export_ipa.py`` feeds the host-computed
``decider_coeffs``. The committer-key ``generators`` (the fold basis) is the core's
sole runtime affine input. The open op set dispatches on the basis' element type, so
each curve lowers a distinct module — both are written by default.

Run under Bazel — CPU is enough, lowering needs no GPU:

    bazel run //export:export_ipa_fold [-- pallas|vesta]
"""
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

from accumulation_zorch import curve, ipa_open, ipa_pc_as, sponge
from accumulation_zorch.curve import Curve

_TESTDATA = Path(__file__).resolve().parent.parent / "python" / "testdata"
_FIXTURE = {
    "pallas": (_TESTDATA / "ipa_as_fold_fixtures.json", _TESTDATA / "sponge_fixtures.json"),
    "vesta": (_TESTDATA / "ipa_as_fold_vesta_fixtures.json", _TESTDATA / "sponge_vesta_fixtures.json"),
}
_FIXTURE_ZK = {
    "pallas": (_TESTDATA / "ipa_as_fold_zk_fixtures.json", _TESTDATA / "sponge_fixtures.json"),
    "vesta": (_TESTDATA / "ipa_as_fold_zk_vesta_fixtures.json", _TESTDATA / "sponge_vesta_fixtures.json"),
}

ART = Path(
    os.environ.get(
        "ACCUMULATION_ZORCH_ARTIFACTS",
        str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
)


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _point(cv: Curve, p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


class _Input(NamedTuple):
    """One parsed input / accumulator instance — the fields the succinct check +
    AS combine read (mirrors ``ipa_as_fold{,_zk}_test._Input``). ``hiding_comm`` /
    ``rand`` are present only for a hiding (zk-prove) accumulator."""
    commitment: Any
    point: int
    value: int
    l_vec: list
    r_vec: list
    final_comm_key: Any
    hiding_comm: Any = None
    rand: int = 0


def _parse_input(cv: Curve, d: Any) -> _Input:
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


def _params(cv: Curve, sponge_fixture: Path) -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(sponge_fixture.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def write_bytecode(lowered: Any, path: Path) -> int:
    """Serialize a lowered module to StableHLO bytecode (mirrors
    ``export_ipa.write_bytecode`` / ``export_prove.write_bytecode``)."""
    m = lowered.compiler_ir(dialect="stablehlo")
    try:
        from jax._src.interpreters import mlir as _jmlir

        data = _jmlir.module_to_bytecode(m)
    except Exception:
        buf = io.BytesIO()
        m.operation.write_bytecode(buf)
        data = buf.getvalue()
    path.write_bytes(data)
    return len(data)


def _combine(cv: Curve, params: Any, d: Any) -> tuple:
    """The host-side no-zk fold combine on one fixture: succinct-check the new input
    then the prior accumulator (inputs first, then accumulators — the
    ``succinct_check_inputs_and_accumulators`` order), combine, derive the new
    opening point, and densely expand the combined check polynomial. Returns
    ``(combined_commitment, point, coeffs)`` — the open's baked inputs."""
    new_input = _parse_input(cv, d["input"])
    acc_prev = _parse_input(cv, d["acc_prev"])
    succinct_checks = [
        ipa_pc_as.succinct_check_input(cv, params, new_input),
        ipa_pc_as.succinct_check_input(cv, params, acc_prev),
    ]
    instance, addends = ipa_pc_as._prove_instance(cv, params, succinct_checks, None, None)
    coeffs = ipa_pc_as.combine_check_polynomials(cv, addends, None)
    return instance.commitment, instance.point, coeffs


def export_fold(cv: Curve) -> Path:
    """Lower the fused no-zk IPA fold open core to ``ipa_fold_<curve>.mlirbc``: bake
    one fixture's combined check polynomial into ``ipa_open.build_open_no_zk_core``
    and lower it over the committer-key ``generators`` (the sole runtime input)."""
    fold_fixture, sponge_fixture = _FIXTURE[cv.name]
    d = json.loads(fold_fixture.read_text())
    params = _params(cv, sponge_fixture)
    commitment, point, coeffs = _combine(cv, params, d)
    svk_h = _point(cv, d["h"])
    generators = [_point(cv, g) for g in d["generators"]]

    core = ipa_open.build_open_no_zk_core(cv, params, svk_h, commitment, point, coeffs)
    basis = curve.stack_affine(cv, generators[: len(coeffs)])

    t0 = time.perf_counter()
    lowered = core.lower(basis)
    t_lower = time.perf_counter() - t0
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"ipa_fold_{cv.name}.mlirbc"
    size = write_bytecode(lowered, out)
    print(f"wrote {out} ({size} B); {cv.name} IPA fold open core; "
          f"lower {t_lower:.2f}s; coeffs={len(coeffs)}, bases={basis.shape[0]}, "
          f"rounds={(len(coeffs) - 1).bit_length()}")
    return out


def _combine_zk(cv: Curve, params: Any, d: Any) -> tuple:
    """The host-side **zk** fold combine on one fixture: succinct-check the no-zk new
    input then the *hiding* prior accumulator (`s` folding its `hiding_comm`/`rand`),
    combine with the random linear polynomial, derive the new opening point, and
    densely expand the rlp-seeded combined check polynomial. Returns
    `(randomized_commitment, point, coeffs)` — the hiding open's baked statement."""
    new_input = _parse_input(cv, d["input"])
    acc_prev = _parse_input(cv, d["acc_prev"])
    s = _point(cv, d["s"])
    rlp_coeffs = [_fr(h) for h in d["random_linear_polynomial"]]
    rlp_commitment = _point(cv, d["random_linear_polynomial_commitment"])
    commitment_randomness = _fr(d["commitment_randomness"])
    proof = ipa_pc_as.Randomness(rlp_coeffs, rlp_commitment, commitment_randomness)
    succinct_checks = [
        ipa_pc_as.succinct_check_input(cv, params, new_input),
        ipa_pc_as.succinct_check_input(cv, params, acc_prev, s),
    ]
    instance, addends = ipa_pc_as._prove_instance(cv, params, succinct_checks, proof, s)
    coeffs = ipa_pc_as.combine_check_polynomials(cv, addends, rlp_coeffs)
    return instance.commitment, instance.point, coeffs


def export_fold_zk(cv: Curve) -> Path:
    """Lower the fused **zk** IPA fold open core to ``ipa_fold_zk_<curve>.mlirbc``:
    bake one fixture's rlp-seeded combined check polynomial + the open's hiding
    blinders into ``ipa_open.build_open_zk_core`` and lower over ``generators``."""
    fold_fixture, sponge_fixture = _FIXTURE_ZK[cv.name]
    d = json.loads(fold_fixture.read_text())
    params = _params(cv, sponge_fixture)
    commitment, point, coeffs = _combine_zk(cv, params, d)
    svk_h = _point(cv, d["h"])
    s = _point(cv, d["s"])
    hiding_poly = [_fr(h) for h in d["hiding_polynomial"]]
    hiding_rand = _fr(d["hiding_rand"])
    commitment_randomness = _fr(d["commitment_randomness"])
    generators = [_point(cv, g) for g in d["generators"]]

    core = ipa_open.build_open_zk_core(
        cv, params, svk_h, s, commitment, point, coeffs, hiding_poly, hiding_rand,
        commitment_randomness)
    basis = curve.stack_affine(cv, generators[: len(coeffs)])

    t0 = time.perf_counter()
    lowered = core.lower(basis)
    t_lower = time.perf_counter() - t0
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"ipa_fold_zk_{cv.name}.mlirbc"
    size = write_bytecode(lowered, out)
    print(f"wrote {out} ({size} B); {cv.name} IPA zk fold open core; "
          f"lower {t_lower:.2f}s; coeffs={len(coeffs)}, bases={basis.shape[0]}, "
          f"rounds={(len(coeffs) - 1).bit_length()}")
    return out


def main() -> None:
    # `export_ipa_fold.py [pallas|vesta] [zk]` — exports the named curve's fold open
    # core (no-zk by default, hiding when `zk` is passed), or BOTH curves when no
    # curve arg (the byte-match tests exercise both Pasta curves).
    args = sys.argv[1:]
    curves = {"pallas": curve.PALLAS, "vesta": curve.VESTA}
    zk = "zk" in args
    exp = export_fold_zk if zk else export_fold
    named = [a for a in args if a in curves]
    if named:
        exp(curves[named[0]])
        return
    if [a for a in args if a not in curves and a != "zk"]:
        raise SystemExit("usage: export_ipa_fold.py [pallas|vesta] [zk]")
    for cv in (curve.PALLAS, curve.VESTA):
        exp(cv)


if __name__ == "__main__":
    main()
