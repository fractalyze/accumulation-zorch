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
* PROVER — `_open_one_zk` vs the port's `ipa_pc.open_zk` (the repo's
  arkworks-faithful zk oracle) on a witness poly with the fixture's blinding poly /
  randomizers: `commit_zk` + l_vec/r_vec/c + hiding_comm + rand byte-match.

zorch ships only the running-transcript default challenger (`TranscriptChallenger`,
not arkworks-byte-exact); the byte-exact Fiat-Shamir is the consumer's job (see
`zorch/pcs/ipa/challenger.py`). `ArkZkIpaChallenger` below is that consumer
challenger, wrapping this repo's `sponge`/`absorbable`/`ipa_pc` primitives: a fresh
`IPA-PC-2020` sponge per challenge, the previous challenge re-absorbed, a
truncated-128 squeeze, plus the extra pre-fold hiding squeeze.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import zk_dtypes as zk
from absl.testing import absltest
from jax import lax

from accumulation_zorch import absorbable, curve, ipa_pc, sponge
from accumulation_zorch.curve import Curve

from zorch.pcs.ipa.config import IpaZkProof
from zorch.pcs.ipa.math import challenge_vector
from zorch.pcs.ipa.prover import IpaProver, _open_one_zk
from zorch.pcs.ipa.setup import setup
from zorch.pcs.ipa.verifier import reduce_opening_zk, settle

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"

# A deterministic degree-7 witness poly for the prover gate (any poly works).
_POLY = [3, 1, 4, 1, 5, 9, 2, 6]

# (name, curve, scalar mont dtype, ipa-zk fixture, ipa-as-zk fixture, sponge fixture)
_CURVES = [
    ("pallas", curve.PALLAS, zk.pallas_sf_mont,
     "ipa_zk_fixtures.json", "ipa_as_zk_fixtures.json", "sponge_fixtures.json"),
    ("vesta", curve.VESTA, zk.vesta_sf_mont,
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


def _canon_int(x: jax.Array) -> int:
    """Canonical int of a zorch scalar Array — `int()` decodes regardless of the
    Montgomery storage. The arkworks FS absorbs canonical 32B LE."""
    return int(np.asarray(x))


def _host_point(p: jax.Array) -> np.ndarray:
    """zorch g1-affine Array (0-d) -> host np point (standard affine)."""
    return np.asarray(p)


def _hexs(cv: Curve, arr: jax.Array) -> str:
    return cv.fr(int(np.asarray(arr))).tobytes().hex()


def _point_hex(p: jax.Array) -> tuple[str, str]:
    raw = _host_point(p).tobytes()
    return raw[:32].hex(), raw[32:64].hex()


@dataclass(frozen=True)
class ArkZkIpaChallenger:
    """arkworks `ipa_pc` **zk** Fiat-Shamir as a zorch `ZkIpaChallenger`. `seed` /
    `challenge` are the no-zk chain (a fresh `IPA-PC-2020` sponge per challenge, the
    previous challenge re-absorbed, a truncated-128 squeeze); `hiding_challenge` is
    the extra pre-fold squeeze over `(commitment, hiding_comm, point, value)`.
    Wraps the accumulation-zorch port's primitives; threads only the prev challenge."""

    cv: Curve
    params: Any
    dtype: Any
    prev: int | None = None

    def _wrap(self, ch: int) -> tuple[ArkZkIpaChallenger, jax.Array]:
        return (ArkZkIpaChallenger(self.cv, self.params, self.dtype, ch),
                jnp.asarray(np.array(ch, dtype=self.dtype)))

    def hiding_challenge(self, commitment: jax.Array, hiding_comm: jax.Array,
                         point: jax.Array, value: jax.Array
                         ) -> tuple[ArkZkIpaChallenger, jax.Array]:
        sp = ipa_pc._new(self.cv, self.params)
        sp = absorbable.absorb_point(self.cv, sp, _host_point(commitment))
        sp = absorbable.absorb_point(self.cv, sp, _host_point(hiding_comm))
        sp = absorbable.absorb_bytes(
            self.cv, sp,
            ipa_pc._fr32(_canon_int(point)) + ipa_pc._fr32(_canon_int(value)))
        return self._wrap(ipa_pc._squeeze_challenge(self.cv, sp))

    def seed(self, commitment: jax.Array, point: jax.Array,
             value: jax.Array) -> tuple[ArkZkIpaChallenger, jax.Array]:
        sp = ipa_pc._new(self.cv, self.params)
        sp = absorbable.absorb_point(self.cv, sp, _host_point(commitment))
        sp = absorbable.absorb_bytes(
            self.cv, sp,
            ipa_pc._fr32(_canon_int(point)) + ipa_pc._fr32(_canon_int(value)))
        return self._wrap(ipa_pc._squeeze_challenge(self.cv, sp))

    def challenge(self, l: jax.Array, r: jax.Array) -> tuple[ArkZkIpaChallenger, jax.Array]:
        assert self.prev is not None, "challenge() before seed()"
        sp = ipa_pc._new(self.cv, self.params)
        sp = absorbable.absorb_bytes(
            self.cv, sp, int(self.prev).to_bytes(ipa_pc._CHALLENGE_BYTES, "little"))
        sp = absorbable.absorb_point(self.cv, sp, _host_point(l))
        sp = absorbable.absorb_point(self.cv, sp, _host_point(r))
        return self._wrap(ipa_pc._squeeze_challenge(self.cv, sp))


class IpaSeamZkTest(absltest.TestCase):
    """zorch `pcs/ipa` zk seam ≡ arkworks `ipa_pc` zk, driven by the consumer's
    arkworks-faithful `ArkZkIpaChallenger`, over Pallas + Vesta."""

    def test_verifier_reduce_opening_zk_byte_matches_arkworks(self) -> None:
        """`reduce_opening_zk` + `settle` over the static arkworks zk proof: round
        challenges (hiding-folded seed) + `final_comm_key` byte-match, and accept."""
        for name, cv, mont, ipa_zk_f, ipa_as_zk_f, sponge_f in _CURVES:
            z = json.loads((_TESTDATA / ipa_zk_f).read_text())
            a = json.loads((_TESTDATA / ipa_as_zk_f).read_text())
            params = _params(cv, sponge_f)
            basis = jnp.stack([_jax_point(cv, g) for g in a["generators"]])
            key = setup(basis, _jax_point(cv, z["h"]), _jax_point(cv, z["s"]))

            proof = IpaZkProof(
                jnp.stack([_jax_point(cv, p) for p in z["l_vec"]]),
                jnp.stack([_jax_point(cv, p) for p in z["r_vec"]]),
                jnp.asarray(np.array(_fr(z["c"]), dtype=mont)),
                _jax_point(cv, z["hiding_comm"]),
                jnp.asarray(np.array(_fr(z["rand"]), dtype=mont)),
            )
            commitment = _jax_point(cv, z["commitment"])
            point = jnp.asarray(np.array(_fr(z["point"]), dtype=mont))
            value = jnp.asarray(np.array(_fr(z["evaluation"]), dtype=mont))

            _, claim = reduce_opening_zk(
                key, commitment, point, value, proof,
                ArkZkIpaChallenger(cv, params, mont))

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
        """`commit_zk` + `_open_one_zk` on a witness poly vs the port's `open_zk`
        (arkworks-faithful zk oracle): commit_zk + l_vec/r_vec/c + hiding_comm + rand."""
        for name, cv, mont, ipa_zk_f, ipa_as_zk_f, sponge_f in _CURVES:
            z = json.loads((_TESTDATA / ipa_zk_f).read_text())
            a = json.loads((_TESTDATA / ipa_as_zk_f).read_text())
            params = _params(cv, sponge_f)
            basis = jnp.stack([_jax_point(cv, g) for g in a["generators"]])
            key = setup(basis, _jax_point(cv, z["h"]), _jax_point(cv, z["s"]))
            point_int = _fr(z["point"])

            generators_np = [_np_point(cv, g) for g in a["generators"]]
            svk_h_np = _np_point(cv, a["h"])
            s_np = _np_point(cv, a["s"])
            hiding_poly = [_fr(h) for h in a["hiding_polynomial"]]
            hiding_rand = _fr(a["hiding_rand"])
            crand = _fr(a["commitment_randomness"])

            # Hiding commit: zorch commit_zk == the port's hiding pedersen_commit.
            coeffs = jnp.asarray(np.array(_POLY, dtype=mont))
            crand_j = jnp.asarray(np.array(crand, dtype=mont))
            commitment_z, _ = IpaProver(key).commit_zk([coeffs], [crand_j])
            comm_np = curve.pedersen_commit(cv, generators_np, _POLY, hiding=s_np,
                                            randomizer=crand)
            self.assertEqual(_host_point(commitment_z[0]).tobytes(), comm_np.tobytes(),
                             f"[{name}] commit_zk != port hiding pedersen_commit")

            point = jnp.asarray(np.array(point_int, dtype=mont))
            hiding_j = jnp.asarray(np.array(hiding_poly, dtype=mont))
            hrand_j = jnp.asarray(np.array(hiding_rand, dtype=mont))

            _, _, pz, _, _ = _open_one_zk(
                key, commitment_z[0], coeffs, point, hiding_j, hrand_j, crand_j,
                ArkZkIpaChallenger(cv, params, mont))
            pp = ipa_pc.open_zk(cv, params, svk_h_np, s_np, generators_np, comm_np,
                                point_int, _POLY, hiding_poly, hiding_rand, crand)

            for j in range(len(pp.l_vec)):
                self.assertEqual(_host_point(pz.l[j]).tobytes(), pp.l_vec[j].tobytes(),
                                 f"[{name}] L[{j}]")
                self.assertEqual(_host_point(pz.r[j]).tobytes(), pp.r_vec[j].tobytes(),
                                 f"[{name}] R[{j}]")
            self.assertEqual(cv.fr(int(np.asarray(pz.a))).tobytes(),
                             cv.fr(int(pp.c)).tobytes(), f"[{name}] c")
            self.assertEqual(_host_point(pz.hiding_comm).tobytes(),
                             pp.hiding_comm.tobytes(), f"[{name}] hiding_comm")
            self.assertEqual(cv.fr(int(np.asarray(pz.rand))).tobytes(),
                             cv.fr(int(pp.rand)).tobytes(), f"[{name}] rand")
            print(f"  [{name}] prover: commit_zk + l_vec[{len(pp.l_vec)}] + r_vec + c + "
                  f"hiding_comm + rand == port open_zk (arkworks)")


if __name__ == "__main__":
    absltest.main()
