"""Export the IPA-PC accumulation decider's size-`d` MSM to one StableHLO
``.mlirbc`` — the Slice-4 fused GPU core, à la ``export_prove.py`` /
bellman-zorch's exporter.

The IPA-AS decider's only GPU-value work is the size-`d` MSM
``final_key = Σ generators_i · coeffs_i``, where ``coeffs`` is the dense
``compute_coeffs(succinct_check(accumulator))`` of the accumulator's check
polynomial; the decider accepts iff ``final_key == accumulator.final_comm_key``
(``ipa_pc_as.decide_final_key`` / ``IpaPC::check``'s final equality). Per the
issue's profile note the heavy MSM lives here (not in the field-heavy prover), so
this lowers exactly that one MSM (``lax.msm``) to a single PJRT
call: the committer-key ``generators`` AND the check-poly ``coeffs`` are runtime
inputs, so one general core decides any accumulator (the fixture supplies only the
runtime shapes — degree-`d` for both Pasta curves here).

The challenge-derivation + ``compute_coeffs`` that produce ``coeffs`` stay
host-side (cheap field/sponge work, already byte-matched on CPU in slices 1-3);
the Rust consumer feeds the arkworks-golden ``decider_coeffs`` from the fixture
(tied to the jax port by ``testing/ipa_as_test.py``). The MSM op dispatches on the
bases' element type, so each curve lowers a distinct module — both are written by
default.

Run under Bazel — CPU is enough, lowering needs no GPU:

    bazel run //export:export_ipa [-- pallas|vesta]
"""
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from accumulation_zorch import curve
from accumulation_zorch.curve import Curve

_TESTDATA = Path(__file__).resolve().parent.parent / "python" / "testdata"
_FIXTURE = {
    "pallas": _TESTDATA / "ipa_as_fixtures.json",
    "vesta": _TESTDATA / "ipa_as_vesta_fixtures.json",
}

# Artifacts live next to the crate so the Rust consumer can find them via a stable
# default; override with the env var for out-of-tree builds.
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


def write_bytecode(lowered: Any, path: Path) -> int:
    """Serialize a lowered module to StableHLO bytecode (the format the plugin's
    ``PJRT_Client_Compile`` consumes). Mirrors ``export_prove.write_bytecode``."""
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


def build_decider_core(cv: Curve) -> tuple:
    """Build the general decider-MSM inputs for curve ``cv`` from its ``ipa_as``
    fixture: return the example ``(scalars, bases)`` — the ``decider_coeffs`` /
    ``generators`` arrays carrying the runtime shapes. Both are runtime inputs to
    the lowered ``lax.msm`` core, so it decides any accumulator of this degree."""
    d = json.loads(_FIXTURE[cv.name].read_text())
    generators = [_point(cv, g) for g in d["generators"]]
    coeffs = [_fr(h) for h in d["decider_coeffs"]]
    bases = curve.stack_affine(cv, generators)
    scalars = jnp.asarray(np.array(coeffs, dtype=cv.fr))
    return scalars, bases


def export_decider(cv: Curve) -> Path:
    """Lower the general decider-MSM core to ``ipa_decider_msm_<curve>.mlirbc``.
    The committer-key ``generators`` and the check-poly ``coeffs`` are BOTH runtime
    inputs (``_core(scalars, bases) = lax.msm(scalars, bases)``), so one lowered
    core decides any accumulator at this degree."""
    scalars, bases = build_decider_core(cv)
    t0 = time.perf_counter()
    # `lax.msm` is a plain function (the per-leaf `@jit` was dropped); jit it
    # here so the lowered core is the single fused MSM call the plugin consumes.
    lowered = jax.jit(lax.msm).lower(scalars, bases)
    t_lower = time.perf_counter() - t0
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"ipa_decider_msm_{cv.name}.mlirbc"
    size = write_bytecode(lowered, out)
    print(f"wrote {out} ({size} B); {cv.name} decider size-d MSM core; "
          f"lower {t_lower:.2f}s; coeffs={scalars.shape[0]}, bases={bases.shape[0]}")
    return out


def export_decider_bench(cv: Curve, n: int) -> Path:
    """Lower the general decider-MSM core at bench size ``n`` to
    ``ipa_decider_msm_bench.mlirbc``. The example arrays carry only the runtime
    shapes — ``n`` scalars + ``n`` bases — so their values are irrelevant to the
    traced module; the scale bench feeds real random inputs at run time. The bases
    are one fixture generator replicated (a valid affine point; the MSM kernel
    dispatches on its element type)."""
    d = json.loads(_FIXTURE[cv.name].read_text())
    bases = curve.stack_affine(cv, [_point(cv, d["generators"][0])] * n)
    scalars = jnp.asarray(np.zeros(n, dtype=cv.fr))
    t0 = time.perf_counter()
    lowered = jax.jit(lax.msm).lower(scalars, bases)
    t_lower = time.perf_counter() - t0
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "ipa_decider_msm_bench.mlirbc"
    size = write_bytecode(lowered, out)
    print(f"wrote {out} ({size} B); {cv.name} decider MSM bench core; "
          f"lower {t_lower:.2f}s; size={n}")
    return out


def main() -> None:
    # `export_ipa.py [pallas|vesta]` — exports the named curve's decider MSM core,
    # or BOTH (the byte-match test exercises both Pasta curves) when no arg given.
    # `IPA_DECIDER_SIZE=<n>` instead lowers the size-`n` bench core for `PROVE_CURVE`
    # (default Pallas), for the scale benchmark.
    bench_size = os.environ.get("IPA_DECIDER_SIZE")
    if bench_size:
        cv = {"pallas": curve.PALLAS, "vesta": curve.VESTA}[
            os.environ.get("PROVE_CURVE", "pallas").lower()]
        export_decider_bench(cv, int(bench_size))
        return

    args = sys.argv[1:]
    curves = {"pallas": curve.PALLAS, "vesta": curve.VESTA}
    if args:
        if args[0] not in curves:
            raise SystemExit(f"unknown curve {args[0]!r} (expected pallas|vesta)")
        export_decider(curves[args[0]])
        return
    for cv in (curve.PALLAS, curve.VESTA):
        export_decider(cv)


if __name__ == "__main__":
    main()
