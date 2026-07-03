"""Export the fused **no-zk NARK** prove of the recursion circuit to one
StableHLO ``.mlirbc`` — the Vesta half-step.

The whole no-zk NARK prove (three sparse ``M·z`` reduces + three ``lax.msm``
commitments) is one ``@jax.jit`` core (``nark.build_no_zk_core``) that takes the
committer key (``bases``) as its sole affine argument and closes over the circuit
— the sparse-COO matrices and ``z = input ‖ witness``, baked as constants. This
lowers that core to one module: the single PJRT call the GPU byte-match runs
against ``recursion_step_proves_on_vesta``.

Unlike the toy AS export (``export_prove.py``, which densifies), the recursion
circuit's ``M·z`` is reduced on-device from the **sparse** COO
(``jfield.sparse_matvec`` → ``stablehlo.scatter``): densifying it (``rows × vars``
≈ 471M ≈ 15 GB) is infeasible. The fixture is the off-tree Slice-2 recursion dump.

Run under Bazel (CPU is enough — lowering needs no GPU):

    ACCUMULATION_ZORCH_ARTIFACTS=<dir> \\
      bazel run //export:export_nark
"""
import json
import os
import time
from pathlib import Path
from typing import Any

from accumulation_zorch import curve, nark

# Sibling script (carried as a helper src of each export py_binary, so `export/`
# is on the import path); `write_bytecode` is the shared lowered-module →
# StableHLO-bytecode serializer.
from export_prove import write_bytecode

# The recursion half-step proves on Vesta (ark_pallas::Fq == ark_vesta::Fr).
_CURVE = {"pallas": curve.PALLAS, "vesta": curve.VESTA}[
    os.environ.get("PROVE_CURVE", "vesta")
]

# The recursion NARK fixture is large (~17 MB), so it lives off-tree under
# $ACCUMULATION_ZORCH_ARTIFACTS (default `artifacts/`) — the same dir the Rust
# dumper writes it to and the GPU consumer loads the `.mlirbc` from.
ART = Path(
    os.environ.get(
        "ACCUMULATION_ZORCH_ARTIFACTS",
        str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
)
_FIXTURE = ART / "recursion_nark_fixtures.json"
_MLIRBC = ART / f"nark_no_zk_{_CURVE.name}.mlirbc"


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _matrix(rows: Any) -> nark.Matrix:
    return [[(_fr(coeff), idx) for coeff, idx in row] for row in rows]


def _point(p: Any) -> Any:
    return _CURVE.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def main() -> None:
    if not _FIXTURE.exists():
        raise SystemExit(
            f"no fixture at {_FIXTURE}\n"
            "  (generate it: ACCUMULATION_ZORCH_ARTIFACTS=<dir> "
            "cargo test --features recursion --test recursion_step dump_recursion_nark)"
        )
    d = json.loads(_FIXTURE.read_text())
    a, b, c = (_matrix(d[k]) for k in ("a", "b", "c"))
    input_ = [_fr(h) for h in d["input"]]
    witness = [_fr(h) for h in d["witness"]]
    generators = [_point(g) for g in d["generators"]]

    core_fn, bases = nark.build_no_zk_core(_CURVE, a, b, c, input_, witness, generators)
    t0 = time.perf_counter()
    lowered = core_fn.lower(bases)
    t_lower = time.perf_counter() - t0
    ART.mkdir(parents=True, exist_ok=True)
    size = write_bytecode(lowered, _MLIRBC)
    print(
        f"wrote {_MLIRBC} ({size} B); curve={_CURVE.name}; "
        f"lower {t_lower:.2f}s; bases={bases.shape[0]}; "
        f"{d['num_constraints']} constraints, {d['num_vars']} vars"
    )


if __name__ == "__main__":
    main()
