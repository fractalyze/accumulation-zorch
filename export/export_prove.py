"""Export the fused jit zk prove (the whole R1CS-NARK-AS make_zk prove) to one
StableHLO ``.mlirbc``, à la ``export_pasta_msm.py`` / bellman-zorch's exporter.

The whole prove is one ``@jax.jit`` core (zorch#317 Stage A) that takes the
committer key (``bases_h = generators ‖ hiding``) and the HP placeholder identity
as its affine arguments and closes over the circuit / witness / replayed
randomness; everything else — the NARK + HP cores, every commitment / fold, and
all three Fiat-Shamir sponges (gamma / mu-nu / beta) — is inside the trace. This
lowers that core for a fixture seed to one StableHLO module: the single PJRT call
the GPU byte-match (Stage B2) and the Rust consumer run.

zorch#330 lifted the assignment + all replayed randomness (NARK/AS/HP) to runtime
inputs, so this lowers ONE seed-independent ``prove_zk_general.mlirbc`` (the fixture
supplies only the runtime shapes) — no per-seed baking. The two ``u8_batch``
Fiat-Shamir absorbs (``r1cs_input`` / ``r1cs_r_input``) are fed pre-encoded as fq
runtime inputs: the in-trace ``fr→u8`` rechunk the zkx GPU plugin mis-lowers is done
consumer-side.

Run with the Pasta jax-fork venv — CPU is enough, lowering needs no GPU:

    JAX_PLATFORMS=cpu PYTHONPATH=python:<pasta-zorch>/zorch \\
      <venv>/bin/python export/export_prove.py [no-zk]
"""
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from accumulation_zorch import curve, r1cs_nark_as, sponge

# The Pasta cycle curve to export for (`PROVE_CURVE=pallas|vesta`, default Pallas).
# The fixture + Poseidon constants must be the matching curve's; the recursion
# half-step (zorch#326 Slice 3) exports Vesta.
_CURVE = {"pallas": curve.PALLAS, "vesta": curve.VESTA}[
    os.environ.get("PROVE_CURVE", "pallas")
]

_TESTDATA = Path(__file__).resolve().parent.parent / "python" / "testdata"
# Fixture path is overridable (env `AS_ZK_FIXTURE`) so a larger off-tree
# scale/bench fixture can be exported (its assignment shapes flow into the one
# general core) without clobbering the committed size-10 `as_zk_fixtures.json`.
_AS_ZK = Path(os.environ.get("AS_ZK_FIXTURE", str(_TESTDATA / "as_zk_fixtures.json")))
_AS_NO_ZK = _TESTDATA / "as_fixtures.json"
_SPONGE = _TESTDATA / "sponge_fixtures.json"

# Artifacts live next to the crate so the Rust consumer can find them via a stable
# default; override with the env var for out-of-tree builds.
ART = Path(
    os.environ.get(
        "ACCUMULATION_ZORCH_ARTIFACTS",
        str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
)


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(_CURVE, ark_le)


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _matrix(rows: Any) -> r1cs_nark_as.nark.Matrix:
    return [[(_fr(coeff), idx) for coeff, idx in row] for row in rows]


def _point(p: Any) -> Any:
    return _CURVE.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def write_bytecode(lowered: Any, path: Path) -> int:
    """Serialize a lowered module to StableHLO bytecode (the format the plugin's
    ``PJRT_Client_Compile`` consumes). Mirrors ``export_pasta_msm.write_bytecode``
    / bellman-zorch's exporter."""
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


def _seed_entry(d: Any, seed: int) -> Any:
    for s in d["seeds"]:
        if s["seed"] == seed:
            return s
    raise SystemExit(f"seed {seed} not in fixture (have {[s['seed'] for s in d['seeds']]})")


def build_core(seed: int) -> tuple:
    """Build the GENERAL fused zk-prove core (zorch#330) from fixture ``seed``: load
    the circuit + the example assignment / replayed randomness, then return
    ``(core_fn, bases_h, id_pt, <12 runtime arrays>)`` from
    ``r1cs_nark_as._build_zk_core``. ``seed`` supplies only the example arrays for
    the runtime shapes; the lowered core is seed-independent (proves any seed)."""
    d = json.loads(_AS_ZK.read_text())
    a, b, c = _matrix(d["a"]), _matrix(d["b"]), _matrix(d["c"])
    generators = [_point(g) for g in d["generators"]]
    s = _seed_entry(d, seed)
    return r1cs_nark_as._build_zk_core(
        _CURVE, a, b, c, [_fr(h) for h in s["r1cs_input"]], [_fr(h) for h in s["witness"]],
        generators, _point(d["hiding"]), _params(),
        bytes.fromhex(d["nark_matrices_hash_hex"]), bytes.fromhex(d["as_matrices_hash_hex"]),
        d["supported_num_elems"],
        [_fr(h) for h in s["r"]],
        (_fr(s["a_blinder"]), _fr(s["b_blinder"]), _fr(s["c_blinder"]),
         _fr(s["r_a_blinder"]), _fr(s["r_b_blinder"]), _fr(s["r_c_blinder"]),
         _fr(s["blinder_1"]), _fr(s["blinder_2"])),
        _fr(s["as_r1cs_r_input"]), _fr(s["as_r1cs_r_witness"]),
        (_fr(s["as_rand_1"]), _fr(s["as_rand_2"]), _fr(s["as_rand_3"])),
        _fr(s["hp_hiding_a"]), _fr(s["hp_hiding_b"]),
        (_fr(s["hp_rand_1"]), _fr(s["hp_rand_2"]), _fr(s["hp_rand_3"])),
    )


def export_zk_general() -> Path:
    """Lower the GENERAL fused zk prove core (zorch#330) to one StableHLO module
    ``prove_zk_general.mlirbc``. The committer key + placeholder identity AND the
    assignment + all replayed randomness (NARK/AS/HP, plus the two pre-encoded
    ``u8_batch`` fq packings) are ALL runtime inputs — the seed-0 fixture supplies
    only the runtime shapes, so the lowered core proves any seed (no per-seed
    ``.mlirbc``)."""
    (core_fn, bases_h, id_pt, ex_in, ex_wit, ex_r, ex_blinders, ex_r_in, ex_r_wit,
     ex_as_rand, ex_hp_rand, ex_in_u8b, ex_r_in_u8b) = build_core(0)
    t0 = time.perf_counter()
    lowered = core_fn.lower(bases_h, id_pt, ex_in, ex_wit, ex_r, ex_blinders, ex_r_in,
                            ex_r_wit, ex_as_rand, ex_hp_rand, ex_in_u8b, ex_r_in_u8b)
    t_lower = time.perf_counter() - t0
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "prove_zk_general.mlirbc"
    size = write_bytecode(lowered, out)
    print(f"wrote {out} ({size} B); GENERAL zk core; lower {t_lower:.2f}s; "
          f"bases_h={bases_h.shape[0]}, input={ex_in.shape[0]}, witness={ex_wit.shape[0]}")
    return out


def build_no_zk_core(seed: int) -> tuple:
    """Build the GENERAL fused no-zk-prove core (zorch#330) from
    ``as_fixtures.json``: return ``(core_fn, bases, r1cs_input_arr,
    blinded_witness_arr)`` from ``r1cs_nark_as._build_no_zk_core``. The committer
    key ``bases`` AND the assignment (``r1cs_input`` / ``blinded_witness``) are
    runtime inputs; ``seed`` only supplies the example assignment for the runtime
    shapes (the lowered core is seed-independent)."""
    d = json.loads(_AS_NO_ZK.read_text())
    a, b, c = _matrix(d["a"]), _matrix(d["b"]), _matrix(d["c"])
    generators = [_point(g) for g in d["generators"]]
    s = _seed_entry(d, seed)
    return r1cs_nark_as._build_no_zk_core(
        _CURVE, a, b, c, [_fr(h) for h in s["r1cs_input"]],
        [_fr(h) for h in s["blinded_witness"]], generators, d["supported_num_elems"], _params())


def export_no_zk_general() -> Path:
    """Lower the GENERAL fused no-zk prove core (zorch#330) to one StableHLO module
    ``prove_no_zk_general.mlirbc``. The committer key ``bases``, the public input
    ``r1cs_input``, and the ``blinded_witness`` are ALL runtime inputs — the seed-0
    assignment supplies only the runtime shapes, so the lowered core proves any
    assignment (no per-seed ``.mlirbc``)."""
    core_fn, bases, r1cs_input_arr, blinded_witness_arr = build_no_zk_core(0)
    t0 = time.perf_counter()
    lowered = core_fn.lower(bases, r1cs_input_arr, blinded_witness_arr)
    t_lower = time.perf_counter() - t0
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "prove_no_zk_general.mlirbc"
    size = write_bytecode(lowered, out)
    print(f"wrote {out} ({size} B); GENERAL no-zk core; lower {t_lower:.2f}s; "
          f"bases={bases.shape[0]}, input={r1cs_input_arr.shape[0]}, witness={blinded_witness_arr.shape[0]}")
    return out


def main() -> None:
    # `export_prove.py [no-zk]` — exports the GENERAL no-zk or zk core once
    # (zorch#330; assignment + randomness are runtime inputs, so one seed-independent
    # artifact each, no per-seed baking).
    args = sys.argv[1:]
    if bool(args) and args[0] == "no-zk":
        export_no_zk_general()
        return
    export_zk_general()


if __name__ == "__main__":
    main()
