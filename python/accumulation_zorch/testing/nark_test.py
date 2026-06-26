"""Slice-3 byte-match: the no-zk NARK prove vs the real arkworks prover, over
BOTH Pasta cycle curves (Pallas and Vesta).

Replays a constructed R1CS (the `DummyCircuit` arkworks' `r1cs_nark_as` tests
use) dumped from the crate's real `R1CSNark::prove` (no-zk) — matrices, instance
+ witness assignments, committer-key generators, and the golden serialized
`Proof` (`cargo run --example dump_nark -- <curve>`) — and asserts the ported
prove reproduces the proof byte-for-byte. The no-zk proof is `commit(z_a/z_b/z_c)`
over `z = input ‖ witness` plus the raw witness, with no randomness, no gamma.

Running the SAME curve-generic prover (`nark.prove_no_zk(cv, ...)`) against the
Vesta golden — different `g1`/`fr`/`fq` dtypes, different committer key — is the
Phase-4 Slice-1 gate: it proves the curve abstraction is genuinely generic, not
just non-breaking on Pallas. Per-commitment anchors (the leading 3×33B of the
proof) localize a divergence to a single matrix's `matrix_vec_mul` + commit.

Run (from the repo's `python/` dir, in the accumulation-zorch venv):

    JAX_PLATFORMS=cpu PYTHONPATH=.:<pasta-zorch>/zorch \
      python accumulation_zorch/testing/nark_test.py
"""

import json
from pathlib import Path
from typing import Any

from accumulation_zorch import curve, nark

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

# (curve, golden fixture) for each cycle curve — the same prover, two curves.
_CURVES = [
    (curve.PALLAS, _TESTDATA / "nark_fixtures.json"),
    (curve.VESTA, _TESTDATA / "nark_vesta_fixtures.json"),
]


def _parse_matrix(rows: Any) -> Any:
    return [[(int.from_bytes(bytes.fromhex(coeff), "little"), idx) for coeff, idx in row] for row in rows]


def _fr_list(hexes: Any) -> Any:
    return [int.from_bytes(bytes.fromhex(h), "little") for h in hexes]


def _load(cv: curve.Curve, fixture: Path) -> Any:
    d = json.loads(fixture.read_text())
    a, b, c = (_parse_matrix(d[k]) for k in ("a", "b", "c"))
    input_ = _fr_list(d["input"])
    witness = _fr_list(d["witness"])
    generators = [
        cv.g1((
            int.from_bytes(bytes.fromhex(g["x_le_hex"]), "little"),
            int.from_bytes(bytes.fromhex(g["y_le_hex"]), "little"),
        ))
        for g in d["generators"]
    ]
    return d, a, b, c, input_, witness, generators


def test_nark_no_zk_proof_matches_arkworks() -> None:
    for cv, fixture in _CURVES:
        d, a, b, c, input_, witness, generators = _load(cv, fixture)
        proof = nark.prove_no_zk(cv, a, b, c, input_, witness, generators)
        got = proof.hex()
        assert got == d["proof_hex"], f"[{cv.name}] NARK proof:\n got  {got}\n want {d['proof_hex']}"
        assert len(proof) == len(d["proof_hex"]) // 2
        print(f"  [{cv.name}] no-zk NARK proof byte-matches arkworks ({len(proof)} bytes)")


def test_first_round_commitments_match_arkworks() -> None:
    """Per-commitment anchors: each first-round commitment is one 33B field of
    the proof, so a mismatch localizes to that matrix's matrix_vec_mul + commit."""
    for cv, fixture in _CURVES:
        d, a, b, c, input_, witness, generators = _load(cv, fixture)
        want = d["proof_hex"]
        comms = {
            "comm_a": (a, want[0:66]),
            "comm_b": (b, want[66:132]),
            "comm_c": (c, want[132:198]),
        }
        for name, (matrix, want_hex) in comms.items():
            z = nark.matrix_vec_mul(cv, matrix, input_, witness)
            got_hex = curve.point_to_bytes(cv, curve.pedersen_commit(cv, generators, z)).hex()
            assert got_hex == want_hex, f"[{cv.name}] {name}: {got_hex} != {want_hex}"
            print(f"  [{cv.name}] {name} = commit(M·z) byte-matches OK")


def test_no_zk_fused_proof_matches_arkworks() -> None:
    """The fused on-device variant (`prove_no_zk_fused`) reduces `M·z` in-trace
    from the sparse COO (`jfield.sparse_matvec`) instead of host-side, so this is
    the toy-scale regression that the on-device sparse reduce is byte-correct
    before scaling it to the recursion circuit."""
    for cv, fixture in _CURVES:
        d, a, b, c, input_, witness, generators = _load(cv, fixture)
        proof = nark.prove_no_zk_fused(cv, a, b, c, input_, witness, generators)
        assert proof.hex() == d["proof_hex"], (
            f"[{cv.name}] fused no-zk NARK proof diverged from host-side")
        print(f"  [{cv.name}] fused (on-device sparse M·z) no-zk NARK proof byte-matches arkworks")


def main() -> None:
    print("slice-3 NARK no-zk prove byte-match (Pallas + Vesta):")
    test_first_round_commitments_match_arkworks()
    test_nark_no_zk_proof_matches_arkworks()
    test_no_zk_fused_proof_matches_arkworks()
    print("ALL SLICE-3 NARK CHECKS PASSED")


if __name__ == "__main__":
    main()
