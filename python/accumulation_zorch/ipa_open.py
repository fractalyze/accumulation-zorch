"""Adapter: produce accumulation-zorch's `ipa_pc.IpaProof` by driving zorch's
`pcs/ipa` fold — the replacement for accumulation-zorch's former local port of
arkworks' `ipa_pc::open_individual_opening_challenges` (Task 4 / Phase 2; the
local port was deleted in Task 5 once this adapter was byte-matching through
the standing gate — `ipa_as_test.py`/`ipa_as_zk_test.py`, the full accumulator
vs golden arkworks bytes).

The accumulation prover (`ipa_pc_as`) needs an `IpaProof(l_vec, r_vec,
final_comm_key, c[, hiding_comm, rand])`. zorch's `zorch.pcs.ipa.prover._open_one`
/ `_open_one_zk` produce the *same* proof from a uniform per-round basis fold (a
`lax.scan`), byte-identical to arkworks' deferred (BCLMS) even/odd fold. This
module is the thin translation between the two shapes:

* Inputs map onto zorch's `IpaKey` as ``basis = generators[:len(coeffs)]``,
  ``u = svk_h`` (arkworks' inner-product generator `h`; the fold scales it to
  ``h' = U·ξ₀`` itself), and ``s = None`` (no-zk) / the hiding generator (zk). The
  fold is driven by the arkworks-faithful `ipa_challenger.ark_challenger`, so every
  squeezed challenge — hence every `L_j`/`R_j` and the collapsed `a` — is
  byte-identical to arkworks' fold.
* zorch's `IpaProof` carries no folded generator (the fold collapses the basis
  inside its `lax.scan` and drops it), so the accumulator's `final_comm_key` is
  recovered the way zorch's verifier settles a claim (`verifier.settle`): re-derive
  the round challenges through the challenger over the proof's own `L_j`/`R_j`, then
  pay the one size-`n` MSM ``G_final = ⟨challenge_vector(u), G⟩``.

The two `open_*` functions here mirror arkworks' `ipa_pc::open_individual_opening_challenges`
(no-zk / zk) signatures, so `ipa_pc_as` calls them directly — this is the sole
IPA-fold implementation left in the tree.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from zorch.pcs.ipa.math import challenge_vector
from zorch.pcs.ipa.prover import _open_one, _open_one_zk
from zorch.pcs.ipa.setup import IpaKey

from . import curve, ipa_challenger, ipa_pc
from .curve import Curve


def _affine(cv: Curve, point: Any) -> np.ndarray:
    """Normalize a (possibly jacobian) `lax.msm`/fold result to an affine point
    array — the form `ipa_pc.IpaProof`'s fields and the byte comparison expect."""
    return np.asarray(point, dtype=cv.g1)


def _basis(cv: Curve, generators: list[np.ndarray], n: int) -> Array:
    """Stack the first `n` generators into zorch's `IpaKey.basis` (`generators`
    matches `coeffs` in length in the working path, so this slice is a no-op there;
    kept per the mapping `basis = generators[:len(coeffs)]`)."""
    return jnp.asarray(np.stack([_affine(cv, g) for g in generators[:n]]))


def _fr_scalar(cv: Curve, value: Any) -> Array:
    """A host scalar (int or `fr` value) as a 0-d `cv.fr` jax array."""
    return jnp.asarray(np.array([int(value)], dtype=cv.fr))[0]


def _fr_vec(cv: Curve, values: list[int]) -> Array:
    """A list of `fr` ints as a 1-d `cv.fr` jax array."""
    return jnp.asarray(np.array([int(v) for v in values], dtype=cv.fr))


def _c_int(cv: Curve, a: Array) -> int:
    """A 0-d `fr` jax scalar as a canonical `fr` int (the `IpaProof.c`/`rand`
    form)."""
    return int.from_bytes(np.asarray(a, dtype=cv.fr).tobytes(), "little")


def _final_comm_key(
    cv: Curve, params: Any, key: IpaKey, seed_commitment: Any, x: Array, value: Array,
    l: Array, r: Array,
) -> np.ndarray:
    """The fully-folded generator `final_comm_key`, recovered the way zorch's
    verifier settles (`verifier.settle`): re-derive the round challenges by driving
    a fresh `ark_challenger` over the proof's own `L_j`/`R_j` (seeded from the same
    `(seed_commitment, x, value)` the fold used), then pay the one size-`n` MSM
    ``G_final = ⟨challenge_vector(u), G⟩``. `seed_commitment` is the combined
    commitment (no-zk) or the blinding-folded `mod_commitment` (zk) the fold
    actually seeded from."""
    ch = ipa_challenger.ark_challenger(cv, params)
    ch, _xi0 = ch.seed(jnp.asarray(seed_commitment), x, value)
    us = []
    for j in range(l.shape[0]):
        ch, uj = ch.challenge(l[j], r[j])
        us.append(uj)
    s = challenge_vector(jnp.stack(us))
    return _affine(cv, lax.msm(s, key.basis[: s.shape[0]]))


def open_no_zk(
    cv: Curve, params: Any, svk_h: np.ndarray, combined_commitment: np.ndarray,
    point: int, coeffs: list[int], generators: list[np.ndarray],
) -> ipa_pc.IpaProof:
    """Drop-in for arkworks' `ipa_pc::open_individual_opening_challenges` (no-zk),
    driving zorch's `_open_one`: the IPA fold producing `(l_vec, r_vec,
    final_comm_key, c)` byte-identical to arkworks' fold.

    `coeffs` is the combined check polynomial (length `d+1 = 2^log_d`),
    `generators` the committer key `comm_key`, `combined_commitment` seeds the
    Fiat-Shamir, `svk_h` the inner-product generator."""
    n = len(coeffs)
    key = IpaKey(basis=_basis(cv, generators, n), u=jnp.asarray(_affine(cv, svk_h)), s=None)
    coeffs_arr = _fr_vec(cv, coeffs)
    x = _fr_scalar(cv, point)
    commitment = jnp.asarray(_affine(cv, combined_commitment))

    fs = ipa_challenger.ark_challenger(cv, params)
    _fs, value, proof = _open_one(key, commitment, coeffs_arr, x, fs)

    final = _final_comm_key(cv, params, key, commitment, x, value, proof.l, proof.r)
    l_vec = [_affine(cv, proof.l[j]) for j in range(proof.l.shape[0])]
    r_vec = [_affine(cv, proof.r[j]) for j in range(proof.r.shape[0])]
    return ipa_pc.IpaProof(l_vec, r_vec, final, _c_int(cv, proof.a))


def open_zk(
    cv: Curve, params: Any, svk_h: np.ndarray, s: np.ndarray, generators: list[np.ndarray],
    combined_commitment: np.ndarray, point: int, coeffs: list[int],
    hiding_poly_raw: list[int], hiding_rand: int, commitment_randomness: int,
) -> ipa_pc.IpaProof:
    """Drop-in for arkworks' `ipa_pc::open_individual_opening_challenges` (zk),
    driving zorch's `_open_one_zk`: the hiding prelude + shared fold producing
    `(l_vec, r_vec, final_comm_key, c, hiding_comm, rand)` byte-identical to
    arkworks' zk fold.

    `combined_commitment` is the (randomized) accumulator commitment the hiding
    open randomizes; `s` the succinct verifier key's hiding generator;
    `hiding_poly_raw` / `hiding_rand` / `commitment_randomness` the open's replayed
    blinders. The blinding polynomial is padded to length `d+1` (arkworks resizes it
    to `d+1` before the vanish-at-`point` shift)."""
    n = len(coeffs)
    s_pt = jnp.asarray(_affine(cv, s))
    key = IpaKey(basis=_basis(cv, generators, n), u=jnp.asarray(_affine(cv, svk_h)), s=s_pt)
    coeffs_arr = _fr_vec(cv, coeffs)
    x = _fr_scalar(cv, point)
    commitment = jnp.asarray(_affine(cv, combined_commitment))

    # arkworks resizes `P::rand(d)` to `d+1` (zeros for the missing high terms)
    # before the vanish-at-point shift; `_open_one_zk` expects the full length-`n`
    # blinding polynomial.
    raw = [int(c) for c in hiding_poly_raw] + [0] * (n - len(hiding_poly_raw))
    hiding_poly = _fr_vec(cv, raw)
    hiding_rand_s = _fr_scalar(cv, hiding_rand)
    commitment_randomness_s = _fr_scalar(cv, commitment_randomness)

    fs = ipa_challenger.ark_challenger(cv, params)
    _fs, value, zkp = _open_one_zk(
        key, commitment, coeffs_arr, x, hiding_poly, hiding_rand_s,
        commitment_randomness_s, fs)

    hiding_comm = _affine(cv, zkp.hiding_comm)
    combined_rand_int = _c_int(cv, zkp.rand)

    # Recover the fold's seed `mod_commitment` (the blinding-folded commitment
    # `combined_commitment + hc·hiding_comm − s·combined_rand`) to re-derive the
    # round challenges for `final_comm_key`: re-squeeze the hiding challenge `hc`
    # (byte-exact to the one `_open_one_zk` used), then reduce as `_open_one_zk`
    # does. The result is affine, byte-identical to the fold's internal seed.
    # NOTE: this `pedersen_commit` must stay affine-normalization-identical to
    # zorch's `_open_one_zk`'s internal `lax.msm` fold (both reduce the same
    # 3-term hiding fold); if the two MSM primitives ever diverged on point
    # normalization, the re-seeded round challenges here would silently mismatch.
    ch = ipa_challenger.ark_challenger(cv, params)
    _ch, hc = ch.hiding_challenge(commitment, jnp.asarray(hiding_comm), x, value)
    neg_rand = (-combined_rand_int) % cv.fr_modulus
    mod_commitment = curve.pedersen_commit(
        cv, [_affine(cv, combined_commitment), hiding_comm, _affine(cv, s)],
        [1, _c_int(cv, hc), neg_rand])

    final = _final_comm_key(cv, params, key, jnp.asarray(mod_commitment), x, value, zkp.l, zkp.r)
    l_vec = [_affine(cv, zkp.l[j]) for j in range(zkp.l.shape[0])]
    r_vec = [_affine(cv, zkp.r[j]) for j in range(zkp.r.shape[0])]
    return ipa_pc.IpaProof(
        l_vec, r_vec, final, _c_int(cv, zkp.a),
        hiding_comm=hiding_comm, rand=combined_rand_int)
