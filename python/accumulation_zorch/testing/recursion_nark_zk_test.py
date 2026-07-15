"""Slice-3 byte-match: the **zk** NARK prove of the recursion-verifier circuit on
Vesta — the half-step make_zk path.

The make_zk recursion circuit (~77.5K constraints × ~76.2K vars — larger than the
no-zk circuit, which carries no randomness commitments) is proven as a Vesta zk
NARK. This replays the prover's sampled randomness (the `r` witness blinders + 8
sigma blinders, dumped from the crate's real `R1CSNark::prove`) through the ported
`nark.prove_zk(VESTA, …)` and byte-matches the golden the crate emits.

Three Slice-3 pieces converge here vs the no-zk half-step:
  * the **on-device sparse M·z** (step: the six z_M / r_M reduces run as
    `segment_sum`, not densified — the recursion R1CS densified is ~15 GB),
  * the **Vesta Poseidon constants** (`sponge_vesta_fixtures.json`), feeding the
    gamma sponge over Vesta's base field,
  * the **unforked gamma sponge** (`fork=False`): the standalone half-step's
    subject (`recursion_step_proves_on_vesta`) passes a plain `Sponge::new()`, not
    the AS `nark_sponge` fork. gamma's `FirstRoundMessage` point absorb (the
    issue's flagged byte-match trap) goes live here — the no-zk path masked it.

This is the host-side `prove_zk` gate; the fused on-device export is the next slice.

The fixture is large (~147 MB) so it is generated **off-tree**, not committed:

    ACCUMULATION_ZORCH_ARTIFACTS=<dir> \
      cargo test --features recursion --test recursion_step dump_recursion_nark_zk

This test reads it from `$ACCUMULATION_ZORCH_ARTIFACTS` (default `artifacts/`)
and **skips** when absent — the same on-demand contract as the `#[ignore]` GPU
gates.

Run under Bazel:

    bazel test //python/accumulation_zorch/testing:recursion_nark_zk_test
"""

import json
import os
from pathlib import Path
from typing import Any

from absl.testing import absltest

from accumulation_zorch import curve, nark, sponge

cv = curve.VESTA  # the forward half-step proves on Vesta

_REPO = Path(__file__).resolve().parents[3]
_ARTIFACTS = Path(os.environ.get("ACCUMULATION_ZORCH_ARTIFACTS", str(_REPO / "artifacts")))
_FIXTURE = _ARTIFACTS / "recursion_nark_zk_fixtures.json"
_SPONGE = Path(__file__).resolve().parents[2] / "testdata" / "sponge_vesta_fixtures.json"


def _fr(hex_le: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_le), "little")


def _matrix(rows: Any) -> Any:
    return [[(_fr(coeff), idx) for coeff, idx in row] for row in rows]


def _point(p: Any) -> Any:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _params() -> Any:
    ark_le = b"".join(bytes.fromhex(h) for h in json.loads(_SPONGE.read_text())["ark_le_hex"])
    return sponge.poseidon_params(cv, ark_le)


def _load() -> Any:
    return json.loads(_FIXTURE.read_text())


def _prove(d: Any) -> nark.NarkZkProof:
    a, b, c = _matrix(d["a"]), _matrix(d["b"]), _matrix(d["c"])
    return nark.prove_zk(
        cv, a, b, c,
        [_fr(h) for h in d["input"]], [_fr(h) for h in d["witness"]],
        [_point(g) for g in d["generators"]], _point(d["hiding"]), _params(),
        bytes.fromhex(d["nark_matrices_hash_hex"]),
        [_fr(h) for h in d["r"]],
        _fr(d["a_blinder"]), _fr(d["b_blinder"]), _fr(d["c_blinder"]),
        _fr(d["r_a_blinder"]), _fr(d["r_b_blinder"]), _fr(d["r_c_blinder"]),
        _fr(d["blinder_1"]), _fr(d["blinder_2"]),
        fork=False,  # standalone half-step: unforked gamma sponge
    )


class RecursionNarkZkTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        if not _FIXTURE.exists():
            self.skipTest(
                f"no fixture at {_FIXTURE} "
                "(generate it: cargo test --features recursion --test recursion_step dump_recursion_nark_zk)")

    def test_recursion_nark_zk_proof_matches_arkworks(self) -> None:
        d = _load()
        proof = nark.serialize_zk_proof(cv, _prove(d))
        self.assertEqual(proof.hex(), d["proof_hex"], (
            f"[vesta] recursion zk NARK proof diverged "
            f"(got {len(proof)}B, want {len(d['proof_hex'])//2}B)"
        ))
        print(
            f"  [vesta] recursion zk NARK proof byte-matches arkworks "
            f"({d['num_constraints']} constraints, {d['num_vars']} vars, {len(proof)} bytes)"
        )


if __name__ == "__main__":
    absltest.main()
