"""Export the R1CS-NARK accumulation decider to one fused StableHLO ``.mlirbc``
per curve — the GPU decide core (#10), à la ``export_ipa.py`` / ``export_prove.py``.

The R1CS-NARK-AS decider's GPU-value work is the **six size-`n` MSMs** it runs
(`ASForR1CSNark::decide` + `ASForHadamardProducts::decide`): over
`z = r1cs_input ‖ r1cs_blinded_witness`,

    comm_a = commit(A·z, σ_a)   comm_b = commit(B·z, σ_b)   comm_c = commit(C·z, σ_c)
    test_comm_1 = commit(A·z, ρ₁)   test_comm_2 = commit(B·z, ρ₂)   test_comm_3 = commit(A·z∘B·z, ρ₃)

and the decider accepts iff these equal the accumulator's stored commitments.
This lowers exactly that — the three `M·z` reduces (`commit_dense` = matvec +
`lax.msm`) and the HP Hadamard product, then the six `commit_hiding` MSMs over
`generators ‖ hiding` — to a single PJRT call. The committer key (`bases_h`), the
assignment (`z`), and the six randomizers (`rand6`, all 0 on the no-zk path,
where the hiding term vanishes) are runtime inputs, so ONE lowered core decides
any accumulator of the baked circuit shape (zk and no-zk alike).

The matrices `A/B/C` are the circuit shape (baked, dense `n×num_vars`); the
challenge/randomness derivation that yields `z` / `rand6` stays host-side (already
byte-matched on CPU by ``testing/as_decide_test.py``). The MSM dispatches on the
bases' element type, so each curve lowers a distinct module.

Run under Bazel — CPU is enough, lowering needs no GPU:

    bazel run //export:export_as_decide [-- pallas|vesta]

    # bench core (a size-`n` pure 6-MSM decider load, for the scale benchmark):
    AS_DECIDE_SIZE=65536 PROVE_CURVE=pallas bazel run //export:export_as_decide
"""
import io
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from accumulation_zorch import curve, jcurve, jfield, nark
from accumulation_zorch.curve import Curve

_TESTDATA = Path(__file__).resolve().parent.parent / "python" / "testdata"
_FIXTURE = {
    "pallas": _TESTDATA / "as_fixtures.json",
    "vesta": _TESTDATA / "as_vesta_fixtures.json",
}

ART = Path(
    os.environ.get(
        "ACCUMULATION_ZORCH_ARTIFACTS",
        str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
)

_CURVES = {"pallas": curve.PALLAS, "vesta": curve.VESTA}


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _point(cv: Curve, p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _matrix(rows: Any) -> nark.Matrix:
    return [[(_fr(coeff), idx) for coeff, idx in row] for row in rows]


def write_bytecode(lowered: Any, path: Path) -> int:
    """Serialize a lowered module to StableHLO bytecode (mirrors
    ``export_ipa.write_bytecode``)."""
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


def build_decider_core(cv: Curve, a: nark.Matrix, b: nark.Matrix, c: nark.Matrix,
                       num_vars: int) -> Any:
    """The general decider core: `_core(bases_h, z, hp_a, hp_b, rand6)` recomputes
    the six decider commitments. `A/B/C` (baked dense `n×num_vars`) are the circuit
    shape; the runtime inputs are `bases_h = generators ‖ hiding`, the assignment
    `z` (for `comm_{a,b,c} = commit(M·z, σ)`), the HP witness vectors `hp_a`/`hp_b`
    (for the HP decide check `commit(a_vec, ρ₁), commit(b_vec, ρ₂),
    commit(a_vec∘b_vec, ρ₃)`), and `rand6 = (σ_a,σ_b,σ_c,ρ₁,ρ₂,ρ₃)` (= 0 ⇒ the
    no-zk non-hiding commitments).

    The HP witness is a *runtime* input, not `M·z`: on the zk path the accumulator's
    folded `hp_a`/`hp_b` differ from `A·z`/`B·z` of the folded `z` (by the AS
    cross-term `c·M·z_r`), so the decider reads them from the witness (matching
    arkworks `ASForHadamardProducts::decide`). On the no-zk path they coincide with
    `A·z`/`B·z`. Output leaves: `comm_a, comm_b, comm_c, test_comm_1, test_comm_2,
    test_comm_3`."""
    a_dense, b_dense, c_dense = (jnp.asarray(nark.to_dense(cv, m, num_vars)) for m in (a, b, c))

    @jax.jit
    def _core(bases_h: jax.Array, z: jax.Array, hp_a: jax.Array, hp_b: jax.Array,
              rand6: jax.Array) -> tuple:
        av = jfield.matvec(a_dense, z)
        bv = jfield.matvec(b_dense, z)
        cv_vec = jfield.matvec(c_dense, z)
        product = hp_a * hp_b  # Hadamard a_vec ∘ b_vec (fr element-wise)
        return (
            jcurve.commit_hiding(cv, av, rand6[0], bases_h),
            jcurve.commit_hiding(cv, bv, rand6[1], bases_h),
            jcurve.commit_hiding(cv, cv_vec, rand6[2], bases_h),
            jcurve.commit_hiding(cv, hp_a, rand6[3], bases_h),
            jcurve.commit_hiding(cv, hp_b, rand6[4], bases_h),
            jcurve.commit_hiding(cv, product, rand6[5], bases_h),
        )

    return _core


def export_decider(cv: Curve) -> Path:
    """Lower the general decider core to ``as_decider_<curve>.mlirbc`` from the
    curve's no-zk fixture (the circuit shape + committer key; runtime inputs decide
    any zk/no-zk accumulator of that shape)."""
    import json

    d = json.loads(_FIXTURE[cv.name].read_text())
    a, b, c = _matrix(d["a"]), _matrix(d["b"]), _matrix(d["c"])
    generators = [_point(cv, g) for g in d["generators"]]
    hiding = _point(cv, d["hiding"])
    rows = len(a)
    num_vars = len(d["seeds"][0]["r1cs_input"]) + len(d["seeds"][0]["blinded_witness"])

    core = build_decider_core(cv, a, b, c, num_vars)
    bases_h = jcurve.stack_affine(cv, generators[:rows] + [hiding])
    z = jnp.asarray(np.zeros(num_vars, dtype=cv.fr))
    hp_vec = jnp.asarray(np.zeros(rows, dtype=cv.fr))
    rand6 = jnp.asarray(np.zeros(6, dtype=cv.fr))

    t0 = time.perf_counter()
    lowered = core.lower(bases_h, z, hp_vec, hp_vec, rand6)
    t_lower = time.perf_counter() - t0
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"as_decider_{cv.name}.mlirbc"
    size = write_bytecode(lowered, out)
    print(f"wrote {out} ({size} B); {cv.name} R1CS-NARK decider core; "
          f"lower {t_lower:.2f}s; rows={rows}, num_vars={num_vars}")
    return out


def build_decider_bench_core(cv: Curve) -> Any:
    """The scale-bench core: `_core(bases_h, av, bv, cv_vec, rand6)` — the
    decider's six size-`n` MSMs over runtime vectors `A·z`, `B·z`, `C·z` (the cheap
    matvec is excluded; the MSM is the GPU-value op). `test_comm_3` commits the
    Hadamard `A·z ∘ B·z` (on device). One lowering per size, like ``export_ipa``'s
    bench core."""

    @jax.jit
    def _core(bases_h: jax.Array, av: jax.Array, bv: jax.Array, cv_vec: jax.Array,
              rand6: jax.Array) -> tuple:
        product = av * bv
        return (
            jcurve.commit_hiding(cv, av, rand6[0], bases_h),
            jcurve.commit_hiding(cv, bv, rand6[1], bases_h),
            jcurve.commit_hiding(cv, cv_vec, rand6[2], bases_h),
            jcurve.commit_hiding(cv, av, rand6[3], bases_h),
            jcurve.commit_hiding(cv, bv, rand6[4], bases_h),
            jcurve.commit_hiding(cv, product, rand6[5], bases_h),
        )

    return _core


def export_decider_bench(cv: Curve, n: int) -> Path:
    """Lower the size-`n` decider bench core to ``as_decider_bench.mlirbc``. The
    example arrays carry only the runtime shapes — `n+1` bases, `(3, n)` vectors,
    6 randomizers — so the bench feeds real random inputs at run time. The bases
    are one fixture generator replicated (a valid affine point; the MSM dispatches
    on its element type)."""
    import json

    d = json.loads(_FIXTURE[cv.name].read_text())
    g0 = _point(cv, d["generators"][0])
    core = build_decider_bench_core(cv)
    bases_h = jcurve.stack_affine(cv, [g0] * (n + 1))
    zeros_n = jnp.asarray(np.zeros(n, dtype=cv.fr))
    rand6 = jnp.asarray(np.zeros(6, dtype=cv.fr))

    t0 = time.perf_counter()
    lowered = core.lower(bases_h, zeros_n, zeros_n, zeros_n, rand6)
    t_lower = time.perf_counter() - t0
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "as_decider_bench.mlirbc"
    size = write_bytecode(lowered, out)
    print(f"wrote {out} ({size} B); {cv.name} R1CS-NARK decider bench core; "
          f"lower {t_lower:.2f}s; size={n}")
    return out


def main() -> None:
    # `AS_DECIDE_SIZE=<n>` lowers the size-`n` bench core for `PROVE_CURVE`
    # (default Pallas). Otherwise `export_as_decide.py [pallas|vesta]` exports the
    # named curve's decider core, or BOTH when no arg is given.
    bench_size = os.environ.get("AS_DECIDE_SIZE")
    if bench_size:
        cv = _CURVES[os.environ.get("PROVE_CURVE", "pallas").lower()]
        export_decider_bench(cv, int(bench_size))
        return

    args = sys.argv[1:]
    if args:
        if args[0] not in _CURVES:
            raise SystemExit(f"unknown curve {args[0]!r} (expected pallas|vesta)")
        export_decider(_CURVES[args[0]])
        return
    for cv in (curve.PALLAS, curve.VESTA):
        export_decider(cv)


if __name__ == "__main__":
    main()
