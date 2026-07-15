"""Consumer parity test for the zorch `pcs/ipa` **zk/hiding** seam (zorch#348).

Drives ZORCH's `pcs/ipa` zk seam (`commit_zk` / `_open_one_zk` /
`reduce_opening_zk` / `settle`) over the Pasta cycle with an arkworks-faithful
`ZkIpaChallenger`, and byte-matches the arkworks `ipa_pc` **zk** oracle over BOTH
Pasta curves, two ways (the zk twin of the no-zk `ipa_test.py` gate). This is the
committed home of the zorch#347 T4 gate: it pins the seam this repo now consumes as
its single IPA-PC math impl.

* VERIFIER — `reduce_opening_zk` over the static hiding proof in
  `ipa_zk_fixtures.json`: the round challenges (hiding-folded seed) byte-match
  arkworks' `round_challenges`, `final_comm_key = <h(u), G>` byte-matches, and
  `settle()` accepts.
* PROVER — `_open_one_zk` over the fixture's witness polynomial + hiding blinders,
  byte-matched against the arkworks golden zk proof in `ipa_zk_fixtures.json`:
  `commit_zk` == the golden hiding commitment, then l_vec/r_vec/c + hiding_comm +
  rand + final_comm_key == the golden proof.

zorch ships only the running-transcript default challenger (`TranscriptChallenger`,
not arkworks-byte-exact); the byte-exact Fiat-Shamir is the consumer's job (see
`zorch/pcs/ipa/challenger.py`). Both gates inject `ark_challenger`
(`accumulation_zorch.ipa_challenger`), the arkworks-faithful `IPA-PC-2020`
challenger — a jax-traceable `register_dataclass` pytree that rides the fold's
`lax.scan` carry, so the prover's on-device fold derives byte-exact challenges.

Basis from `ipa_as_zk_*fixtures.json` `generators` (same deterministic
`IpaPC::setup(7, test_rng())`; h/s byte-match across both fixtures).

Both gates run on either backend. The prover gate folds EC points affine<->jacobian
(`lax.convert_element_type`); the CPU XLA backend lowers that convert as of
fractalyze/xla#195 (jaxlib >= 0.10.0.dev20260705083732), so it no longer needs the
Pasta GPU plugin. On CPU:

    JAX_PLATFORMS=cpu PYTHONPATH=python \
      python python/accumulation_zorch/testing/ipa_seam_zk_test.py

On GPU (Pasta plugin):

    SO=$HOME/Workspace/envs/pasta-zorch/zkx/bazel-bin/zkx/pjrt/c/pjrt_c_api_gpu_plugin.so
    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
      PJRT_NAMES_AND_LIBRARY_PATHS="cuda:$SO" PYTHONPATH=python \
      python python/accumulation_zorch/testing/ipa_seam_zk_test.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import lax

from accumulation_zorch import curve, sponge
from accumulation_zorch.curve import Curve
from accumulation_zorch.ipa_challenger import ark_challenger

from zorch.pcs.ipa.config import IpaZkProof
from zorch.pcs.ipa.math import challenge_vector
from zorch.pcs.ipa.prover import IpaProver, _open_one_zk
from zorch.pcs.ipa.setup import setup
from zorch.pcs.ipa.verifier import reduce_opening_zk, settle

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

# (name, curve, ipa-zk fixture, ipa-as-zk fixture, sponge fixture)
_CURVES = [
    ("pallas", curve.PALLAS,
     "ipa_zk_fixtures.json", "ipa_as_zk_fixtures.json", "sponge_fixtures.json"),
    ("vesta", curve.VESTA,
     "ipa_zk_vesta_fixtures.json", "ipa_as_zk_vesta_fixtures.json", "sponge_vesta_fixtures.json"),
]


def _fr(h: str) -> int:
    return int.from_bytes(bytes.fromhex(h), "little")


def _np_point(cv: Curve, p: dict) -> np.ndarray:
    return cv.g1((_fr(p["x_le_hex"]), _fr(p["y_le_hex"])))


def _jax_point(cv: Curve, p: dict) -> jax.Array:
    return jnp.asarray(_np_point(cv, p))


def _params(cv: Curve, sponge_fixture: str) -> Any:
    ark_le = b"".join(
        bytes.fromhex(h)
        for h in json.loads((_TESTDATA / sponge_fixture).read_text())["ark_le_hex"]
    )
    return sponge.poseidon_params(cv, ark_le)


def _host_point(p: jax.Array) -> np.ndarray:
    """zorch g1-affine Array (0-d) -> host np point (standard affine)."""
    return np.asarray(p)


def _hexs(cv: Curve, arr: jax.Array) -> str:
    return cv.fr(int(np.asarray(arr))).tobytes().hex()


def _point_hex(p: jax.Array) -> tuple[str, str]:
    raw = _host_point(p).tobytes()
    return raw[:32].hex(), raw[32:64].hex()


class IpaSeamZkTest(absltest.TestCase):
    """zorch `pcs/ipa` zk seam ≡ arkworks `ipa_pc` zk, driven by the consumer's
    arkworks-faithful `ark_challenger`, over Pallas + Vesta."""

    def test_verifier_reduce_opening_zk_byte_matches_arkworks(self) -> None:
        """`reduce_opening_zk` + `settle` over the static arkworks zk proof: round
        challenges (hiding-folded seed) + `final_comm_key` byte-match, and accept."""
        for name, cv, ipa_zk_f, ipa_as_zk_f, sponge_f in _CURVES:
            z = json.loads((_TESTDATA / ipa_zk_f).read_text())
            a = json.loads((_TESTDATA / ipa_as_zk_f).read_text())
            params = _params(cv, sponge_f)
            basis = jnp.stack([_jax_point(cv, g) for g in a["generators"]])
            key = setup(basis, _jax_point(cv, z["h"]), _jax_point(cv, z["s"]))

            proof = IpaZkProof(
                jnp.stack([_jax_point(cv, p) for p in z["l_vec"]]),
                jnp.stack([_jax_point(cv, p) for p in z["r_vec"]]),
                jnp.asarray(np.array(_fr(z["c"]), dtype=cv.fr)),
                _jax_point(cv, z["hiding_comm"]),
                jnp.asarray(np.array(_fr(z["rand"]), dtype=cv.fr)),
            )
            commitment = _jax_point(cv, z["commitment"])
            point = jnp.asarray(np.array(_fr(z["point"]), dtype=cv.fr))
            value = jnp.asarray(np.array(_fr(z["evaluation"]), dtype=cv.fr))

            _, claim = reduce_opening_zk(
                key, commitment, point, value, proof,
                ark_challenger(cv, params))

            for j, want in enumerate(z["round_challenges"]):
                self.assertEqual(_hexs(cv, claim.u[j]), want,
                                 f"[{name}] zk round challenge[{j}]")

            s_vec = challenge_vector(claim.u)
            g_final = lax.msm(s_vec, basis[: s_vec.shape[0]])
            want = z["final_comm_key"]
            self.assertEqual(_point_hex(g_final),
                             (want["x_le_hex"], want["y_le_hex"]),
                             f"[{name}] final_comm_key")

            self.assertTrue(bool(settle(key, claim)),
                            f"[{name}] settle() rejected a valid zk opening")
            print(f"  [{name}] verifier: round_challenges[{len(z['round_challenges'])}] "
                  f"(hiding-folded seed) + final_comm_key byte-match arkworks; settle accepts")

    def test_prover_open_one_zk_byte_matches_port(self) -> None:
        """`commit_zk` + `_open_one_zk` over the fixture's witness polynomial and
        hiding blinders, byte-matched against the arkworks golden zk proof in
        `ipa_zk_fixtures.json`: commit_zk == the golden hiding commitment, then
        l_vec/r_vec/c + hiding_comm + rand + final_comm_key == the golden proof."""
        for name, cv, ipa_zk_f, ipa_as_zk_f, sponge_f in _CURVES:
            z = json.loads((_TESTDATA / ipa_zk_f).read_text())
            a = json.loads((_TESTDATA / ipa_as_zk_f).read_text())
            params = _params(cv, sponge_f)
            # IPA-PC committer generators are hash-derived (deterministic), so the
            # ipa-as fixture's `generators` are the same set this zk proof used.
            basis = jnp.stack([_jax_point(cv, g) for g in a["generators"]])
            key = setup(basis, _jax_point(cv, z["h"]), _jax_point(cv, z["s"]))

            coeffs = jnp.asarray(np.array([_fr(c) for c in z["polynomial"]], dtype=cv.fr))
            crand_j = jnp.asarray(np.array(_fr(z["commitment_randomness"]), dtype=cv.fr))
            point = jnp.asarray(np.array(_fr(z["point"]), dtype=cv.fr))
            hiding_j = jnp.asarray(
                np.array([_fr(h) for h in z["hiding_polynomial"]], dtype=cv.fr))
            hrand_j = jnp.asarray(np.array(_fr(z["hiding_rand"]), dtype=cv.fr))

            # `commit_zk` reproduces the golden (hiding) commitment.
            commitment_z, _ = IpaProver(key).commit_zk([coeffs], [crand_j])
            self.assertEqual(_host_point(commitment_z[0]).tobytes(),
                             _np_point(cv, z["commitment"]).tobytes(),
                             f"[{name}] commit_zk != arkworks hiding commitment")

            _, _, pz, final_comm_key, _ = _open_one_zk(
                key, commitment_z[0], coeffs, point, hiding_j, hrand_j, crand_j,
                ark_challenger(cv, params))

            for j, (lw, rw) in enumerate(zip(z["l_vec"], z["r_vec"])):
                self.assertEqual(_host_point(pz.l[j]).tobytes(), _np_point(cv, lw).tobytes(),
                                 f"[{name}] L[{j}]")
                self.assertEqual(_host_point(pz.r[j]).tobytes(), _np_point(cv, rw).tobytes(),
                                 f"[{name}] R[{j}]")
            self.assertEqual(_hexs(cv, pz.a), z["c"], f"[{name}] c")
            self.assertEqual(_host_point(pz.hiding_comm).tobytes(),
                             _np_point(cv, z["hiding_comm"]).tobytes(), f"[{name}] hiding_comm")
            self.assertEqual(_hexs(cv, pz.rand), z["rand"], f"[{name}] rand")
            self.assertEqual(
                _point_hex(final_comm_key),
                (z["final_comm_key"]["x_le_hex"], z["final_comm_key"]["y_le_hex"]),
                f"[{name}] final_comm_key")
            print(f"  [{name}] prover: commit_zk + l_vec[{len(z['l_vec'])}] + r_vec + c + "
                  f"hiding_comm + rand + final_comm_key byte-match arkworks golden")


if __name__ == "__main__":
    absltest.main()
