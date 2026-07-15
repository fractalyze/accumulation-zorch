"""Export the fused **zk NARK** prove of the recursion circuit to one StableHLO
``.mlirbc`` — the Vesta half-step (make_zk path).

The whole zk NARK prove (six sparse ``M·z`` / ``M·z_r`` reduces, eight ``lax.msm``
blinded commitments, the in-trace gamma sponge, and the gamma-blinded responses)
is one ``@frx.jit`` core (``nark.build_zk_core``) that takes the committer key +
hiding base (``bases_h``) as its sole affine argument and closes over the circuit
and the prover's sampled randomness (the ``r`` blinders + 8 sigma blinders), baked
as constants. This lowers that core to one module: the single PJRT call the GPU
byte-match runs against ``recursion_step_proves_on_vesta`` (make_zk=true).

The gamma sponge is **unforked** (``fork=False``): the standalone half-step's
subject passes a plain ``Sponge::new()``, not the AS ``nark_sponge`` fork. The
``M·z`` is reduced on-device from the **sparse** COO (``field.sparse_matvec`` →
``stablehlo.scatter``): densifying the recursion R1CS is infeasible. The fixture
is the off-tree Slice-3 zk recursion dump.

Run under Bazel (CPU is enough — lowering needs no GPU):

    ACCUMULATION_ZORCH_ARTIFACTS=<dir> \\
      bazel run //export:export_nark_zk
"""
import json
import os
import time
from pathlib import Path
from typing import Any

from accumulation_zorch import curve, nark, sponge

# Sibling script (both run with `export/` on sys.path[0] via direct invocation);
# `write_bytecode` is the shared lowered-module → StableHLO-bytecode serializer.
from export_prove import write_bytecode

# The recursion half-step proves on Vesta (ark_pallas::Fq == ark_vesta::Fr).
_CURVE = {"pallas": curve.PALLAS, "vesta": curve.VESTA}[
    os.environ.get("PROVE_CURVE", "vesta")
]

# The recursion zk NARK fixture is large (~147 MB), so it lives off-tree under
# $ACCUMULATION_ZORCH_ARTIFACTS (default `artifacts/`) — the same dir the Rust
# dumper writes it to and the GPU consumer loads the `.mlirbc` from.
ART = Path(
    os.environ.get(
        "ACCUMULATION_ZORCH_ARTIFACTS",
        str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
)
_FIXTURE = ART / "recursion_nark_zk_fixtures.json"
_SPONGE = Path(__file__).resolve().parent.parent / "python" / "testdata" / "sponge_vesta_fixtures.json"
_MLIRBC = ART / f"nark_zk_{_CURVE.name}.mlirbc"


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _matrix(rows: Any) -> nark.Matrix:
    return [[(_fr(coeff), idx) for coeff, idx in row] for row in rows]


def _point(p: Any) -> Any:
    return _CURVE.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(_CURVE, ark_le)


def main() -> None:
    if not _FIXTURE.exists():
        raise SystemExit(
            f"no fixture at {_FIXTURE}\n"
            "  (generate it: ACCUMULATION_ZORCH_ARTIFACTS=<dir> "
            "cargo test --features recursion --test recursion_step dump_recursion_nark_zk)"
        )
    d = json.loads(_FIXTURE.read_text())
    a, b, c = (_matrix(d[k]) for k in ("a", "b", "c"))
    input_ = [_fr(h) for h in d["input"]]
    witness = [_fr(h) for h in d["witness"]]
    generators = [_point(g) for g in d["generators"]]
    hiding = _point(d["hiding"])

    core_fn, bases_h = nark.build_zk_core(
        _CURVE, a, b, c, input_, witness, generators, hiding, _params(),
        bytes.fromhex(d["nark_matrices_hash_hex"]),
        [_fr(h) for h in d["r"]],
        _fr(d["a_blinder"]), _fr(d["b_blinder"]), _fr(d["c_blinder"]),
        _fr(d["r_a_blinder"]), _fr(d["r_b_blinder"]), _fr(d["r_c_blinder"]),
        _fr(d["blinder_1"]), _fr(d["blinder_2"]),
        fork=False,  # standalone half-step: unforked gamma sponge
    )
    t0 = time.perf_counter()
    lowered = core_fn.lower(bases_h)
    t_lower = time.perf_counter() - t0
    ART.mkdir(parents=True, exist_ok=True)
    size = write_bytecode(lowered, _MLIRBC)
    print(
        f"wrote {_MLIRBC} ({size} B); curve={_CURVE.name}; "
        f"lower {t_lower:.2f}s; bases_h={bases_h.shape[0]}; "
        f"{d['num_constraints']} constraints, {d['num_vars']} vars"
    )


if __name__ == "__main__":
    main()
