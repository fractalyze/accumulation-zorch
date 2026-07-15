"""Export the fused **zk fold** prove of the recursion circuit to one StableHLO
``.mlirbc`` — the full IVC fold step (forward / reverse, num_addends=3).

The whole multi-addend fold (input's NARK + the fold's AS/HP commitments on the
on-device sparse ``M·z``, the ``num_addends=3`` AS-level fold over
``[acc, input, proof_randomness]``, and the HP-level fold INTO the old
accumulator's HP input) is one ``@frx.jit`` core
(``r1cs_nark_as._build_zk_fold_core``) that takes **three** affine arguments —
the committer key ``bases_h = generators[:rows] ‖ hiding``, the HP placeholder
identity ``id_pt``, and the old accumulator's ``(6,)`` commitments ``acc_comms``
``[comm_a, comm_b, comm_c, hp_1, hp_2, hp_3]`` — and closes over both inputs' fr
components + the prover's replayed randomness, baked as constants. This lowers
that core to one module: the single PJRT call the GPU byte-match
(``gpu_fused_fold_zk_byte_match``) runs against ``ASForR1CSNark::prove`` (the same
golden ``recursion_fold_zk_test.py`` byte-matches on CPU).

The fold runs on **both** cycle directions: forward folds on Vesta (constraint
field ``ark_vesta::Fq``), reverse on Pallas. ``PROVE_CURVE`` selects the
direction, which also picks the off-tree fold fixture + the Poseidon sponge
fixture over that direction's constraint field. The ``M·z`` is reduced on-device
from the **sparse** COO (densifying the recursion R1CS is ~15 GB).

Run under Bazel (CPU is enough — lowering needs no GPU):

    ACCUMULATION_ZORCH_ARTIFACTS=<dir> PROVE_CURVE=vesta \\
      bazel run //export:export_fold_zk
"""
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from accumulation_zorch import curve, nark, r1cs_nark_as, sponge

# Sibling script (both run with `export/` on sys.path[0] via direct invocation);
# `write_bytecode` is the shared lowered-module → StableHLO-bytecode serializer.
from export_prove import write_bytecode

# The fold runs both cycle directions; PROVE_CURVE picks one. Each direction has
# its own off-tree fixture and Poseidon sponge over its constraint field — Vesta
# forward over ark_vesta::Fq, Pallas reverse over ark_pallas::Fq (the toy sponge).
_CURVE_NAME = os.environ.get("PROVE_CURVE", "vesta")
_CURVE = {"pallas": curve.PALLAS, "vesta": curve.VESTA}[_CURVE_NAME]
_DIRECTION = {
    "vesta": ("recursion_fold_zk_fixtures.json", "sponge_vesta_fixtures.json"),
    "pallas": ("recursion_fold_zk_pallas_fixtures.json", "sponge_fixtures.json"),
}
_FIXTURE_NAME, _SPONGE_NAME = _DIRECTION[_CURVE_NAME]

# The recursion fold fixtures are large (~165 MB), so they live off-tree under
# $ACCUMULATION_ZORCH_ARTIFACTS (default `artifacts/`) — the same dir the Rust
# dumper writes them to and the GPU consumer loads the `.mlirbc` from.
ART = Path(
    os.environ.get(
        "ACCUMULATION_ZORCH_ARTIFACTS",
        str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
)
_FIXTURE = ART / _FIXTURE_NAME
_SPONGE = Path(__file__).resolve().parent.parent / "python" / "testdata" / _SPONGE_NAME
_MLIRBC = ART / f"fold_zk_{_CURVE.name}.mlirbc"


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
            "  (generate it: ACCUMULATION_ZORCH_ARTIFACTS=<dir> cargo test "
            f"--features recursion --test recursion_step {_CURVE_NAME}::dump::dump_recursion_fold_zk)"
        )
    d = json.loads(_FIXTURE.read_text())
    a, b, c = (_matrix(d[k]) for k in ("a", "b", "c"))
    generators = [_point(g) for g in d["generators"]]
    hiding = _point(d["hiding"])

    input2 = [_fr(h) for h in d["input2_r1cs_input"]]
    witness2 = [_fr(h) for h in d["input2_witness"]]
    nark_r = [_fr(h) for h in d["r"]]
    nark_blinders = tuple(_fr(d[k]) for k in (
        "a_blinder", "b_blinder", "c_blinder", "r_a_blinder", "r_b_blinder", "r_c_blinder",
        "blinder_1", "blinder_2"))
    as_rand = tuple(_fr(d[k]) for k in ("as_rand_1", "as_rand_2", "as_rand_3"))
    hp_rand = tuple(_fr(d[k]) for k in ("hp_rand_1", "hp_rand_2", "hp_rand_3"))

    acc = d["acc_prev_instance"]
    accw = d["acc_prev_witness"]
    acc_r1cs_input = [_fr(h) for h in acc["r1cs_input"]]
    acc_comms = [np.asarray(_point(acc[k])) for k in (
        "comm_a", "comm_b", "comm_c", "hp_comm_1", "hp_comm_2", "hp_comm_3")]
    acc_blinded_witness = [_fr(h) for h in accw["r1cs_blinded_witness"]]
    acc_sigma_abc = tuple(_fr(accw[k]) for k in ("sigma_a", "sigma_b", "sigma_c"))
    acc_hp_a_vec = [_fr(h) for h in accw["hp_a_vec"]]
    acc_hp_b_vec = [_fr(h) for h in accw["hp_b_vec"]]
    acc_hp_rand = tuple(_fr(accw[k]) for k in ("hp_rand_1", "hp_rand_2", "hp_rand_3"))

    core_fn, bases_h, id_pt, acc_comms_arr = r1cs_nark_as._build_zk_fold_core(
        _CURVE, a, b, c, input2, witness2, generators, hiding, _params(),
        bytes.fromhex(d["nark_matrices_hash_hex"]), bytes.fromhex(d["as_matrices_hash_hex"]),
        d["supported_num_elems"], nark_r, nark_blinders,
        _fr(d["as_r1cs_r_input"]), _fr(d["as_r1cs_r_witness"]), as_rand,
        _fr(d["hp_hiding_a"]), _fr(d["hp_hiding_b"]), hp_rand,
        acc_r1cs_input, acc_comms, acc_blinded_witness, acc_sigma_abc,
        acc_hp_a_vec, acc_hp_b_vec, acc_hp_rand)
    t0 = time.perf_counter()
    lowered = core_fn.lower(bases_h, id_pt, acc_comms_arr)
    t_lower = time.perf_counter() - t0
    ART.mkdir(parents=True, exist_ok=True)
    size = write_bytecode(lowered, _MLIRBC)
    print(
        f"wrote {_MLIRBC} ({size} B); curve={_CURVE.name}; "
        f"lower {t_lower:.2f}s; bases_h={bases_h.shape[0]}; acc_comms={acc_comms_arr.shape[0]}; "
        f"{d['num_constraints']} constraints, {d['num_vars']} vars"
    )


if __name__ == "__main__":
    main()
