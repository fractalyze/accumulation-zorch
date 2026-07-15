"""Adapter: produce accumulation-zorch's `IpaProof` by driving zorch's
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
* zorch's `_open_one` / `_open_one_zk` fold the committer generator in place and
  return its collapsed head as `final_comm_key` (``g[0]``, equal to the
  ``G_final = ⟨challenge_vector(u), G⟩`` the verifier recomputes in `settle`;
  zorch#371) — the zk opener also returns the blinded `mod_commitment` it opened.
  This adapter reads those straight from the fold, so recovering `final_comm_key`
  needs no challenger replay and no second size-`n` MSM.

The two `open_*` functions here mirror arkworks' `ipa_pc::open_individual_opening_challenges`
(no-zk / zk) signatures, so `ipa_pc_as` calls them directly — this is the sole
IPA-fold implementation left in the tree.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import frx
import frx.numpy as jnp
import numpy as np
from frx import Array, lax

from zorch.pcs.ipa.prover import _open_one, _open_one_zk
from zorch.pcs.ipa.setup import IpaKey

from . import ipa_challenger
from .curve import Curve, FrVec


class IpaProof(NamedTuple):
    """The IPA opening proof: the per-round fold commitments, the fully-folded
    generator and coefficient, and (zk only) the hiding commitment + combined
    blinder. No-zk leaves `hiding_comm`/`rand` as `None`. Produced here by driving
    zorch's fold; the AS combinator (`ipa_pc_as.Accumulator`) carries it."""
    l_vec: list[np.ndarray]
    r_vec: list[np.ndarray]
    final_comm_key: np.ndarray
    c: int
    hiding_comm: np.ndarray | None = None
    rand: int | None = None


def _affine(cv: Curve, point: Any) -> np.ndarray:
    """Normalize a (possibly jacobian) `lax.msm`/fold result to an affine point
    array — the form `IpaProof`'s fields and the byte comparison expect."""
    return np.asarray(point, dtype=cv.g1)


def _basis(cv: Curve, generators: list[np.ndarray], n: int) -> Array:
    """Stack the first `n` generators into zorch's `IpaKey.basis` (`generators`
    matches `coeffs` in length in the working path, so this slice is a no-op there;
    kept per the mapping `basis = generators[:len(coeffs)]`)."""
    return jnp.asarray(np.stack([_affine(cv, g) for g in generators[:n]]))


def _fr_scalar(cv: Curve, value: Any) -> Array:
    """A host scalar (int or `fr` value) as a 0-d `cv.fr` frx array."""
    return jnp.asarray(np.array([int(value)], dtype=cv.fr))[0]


def _fr_vec(cv: Curve, values: FrVec) -> Array:
    """`fr` scalars as a 1-d `cv.fr` frx array — an `fr` array (the combined check
    polynomial) or an int list; `np.asarray(_, dtype=cv.fr)` normalizes both, so no
    `fr` value round-trips through a python int."""
    return jnp.asarray(np.asarray(values, dtype=cv.fr))


def _c_int(cv: Curve, a: Array) -> int:
    """A 0-d `fr` frx scalar as a canonical `fr` int (the `IpaProof.c`/`rand`
    form)."""
    return int.from_bytes(np.asarray(a, dtype=cv.fr).tobytes(), "little")


def _pad_hiding_poly(cv: Curve, hiding_poly_raw: list[int], n: int) -> Array:
    """The open's blinding polynomial as a length-`n` `cv.fr` vector: arkworks resizes
    `P::rand(d)` to `d+1` (zeros for the missing high terms) before the
    vanish-at-`point` shift, and `_open_one_zk` expects the full length-`n` poly."""
    return _fr_vec(cv, [int(c) for c in hiding_poly_raw] + [0] * (n - len(hiding_poly_raw)))


def open_no_zk(
    cv: Curve, params: Any, svk_h: np.ndarray, combined_commitment: np.ndarray,
    point: int, coeffs: np.ndarray, generators: list[np.ndarray],
) -> IpaProof:
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
    _fs, _value, proof, final_comm_key = _open_one(key, commitment, coeffs_arr, x, fs)

    final = _affine(cv, final_comm_key)
    l_vec = [_affine(cv, proof.l[j]) for j in range(proof.l.shape[0])]
    r_vec = [_affine(cv, proof.r[j]) for j in range(proof.r.shape[0])]
    return IpaProof(l_vec, r_vec, final, _c_int(cv, proof.a))


def build_open_no_zk_core(
    cv: Curve, params: Any, svk_h: np.ndarray, combined_commitment: np.ndarray,
    point: int, coeffs: np.ndarray,
):  # type: ignore[no-untyped-def]
    """The fused GPU **fold** core: a `@frx.jit` device twin of :func:`open_no_zk`.

    Bakes the combined check polynomial (`coeffs`), the Fiat-Shamir seed
    `combined_commitment`, and the opening `point` as constants; the committer-key
    `basis` (`generators[:len(coeffs)]`, `cv.g1` affine) is the sole runtime input.
    Runs the whole sequential open on device — zorch's `_open_one` `lax.scan` fold
    (Poseidon squeezed on-device per round from that round's `L_j`/`R_j` via the
    arkworks-faithful `ark_challenger`), reading the fold's collapsed generator
    `g[0]` as `final_comm_key` (zorch#371) — and returns the opening proof leaves
    `(l, r, final_comm_key, c)` as `cv.g1`/`cv.fr` frx
    arrays for the Rust consumer to serialize into the folded accumulator's
    `IpaProof`.

    The host combine that produces `combined_commitment` / `point` / `coeffs` stays
    on the host (cheap field/sponge, already byte-matched on CPU), exactly as the
    decider core feeds host-computed `decider_coeffs`; the export bakes them per
    fixture (mirroring `r1cs_nark_as._build_zk_fold_core`). Its CPU byte-match gate
    is `ipa_as_fold_test`, which this reproduces on GPU."""
    u = jnp.asarray(_affine(cv, svk_h))
    coeffs_arr = _fr_vec(cv, coeffs)
    x = _fr_scalar(cv, point)
    commitment = jnp.asarray(_affine(cv, combined_commitment))

    @frx.jit
    def _core(basis: Array) -> tuple[Array, Array, Array, Array]:
        key = IpaKey(basis=basis, u=u, s=None)
        fs = ipa_challenger.ark_challenger(cv, params)
        _fs, _value, proof, final = _open_one(key, commitment, coeffs_arr, x, fs)
        return proof.l, proof.r, final, proof.a

    return _core


def build_open_zk_core(
    cv: Curve, params: Any, svk_h: np.ndarray, s: np.ndarray, combined_commitment: np.ndarray,
    point: int, coeffs: np.ndarray, hiding_poly_raw: list[int], hiding_rand: int,
    commitment_randomness: int,
):  # type: ignore[no-untyped-def]
    """The fused GPU **zk fold** core: a `@frx.jit` device twin of :func:`open_zk`.

    Bakes the combined check polynomial and the open's replayed hiding blinders
    (`hiding_poly` / `hiding_rand` / `commitment_randomness`); the committer-key
    `basis` is the sole runtime input. Runs the hiding open on device — zorch's
    `_open_one_zk` (the `_hiding_commit` Pedersen commitment, the on-device
    `hiding_challenge`, the blinded `lax.scan` fold) — and reads the fold's collapsed
    generator `g[0]` as `final_comm_key` straight from it (zorch#371). Returns the six
    hiding-proof leaves `(l, r, final_comm_key, c, hiding_comm, rand)`.

    The zk combine that produces `combined_commitment` / `point` / `coeffs` (the
    randomized commitment, the rlp-seeded check polynomial) stays host-side; the
    export bakes them per fixture. CPU byte-match gate: `ipa_as_fold_zk_test`."""
    n = len(coeffs)
    u = jnp.asarray(_affine(cv, svk_h))
    s_pt = jnp.asarray(_affine(cv, s))
    coeffs_arr = _fr_vec(cv, coeffs)
    x = _fr_scalar(cv, point)
    commitment = jnp.asarray(_affine(cv, combined_commitment))
    hiding_poly = _pad_hiding_poly(cv, hiding_poly_raw, n)
    hiding_rand_s = _fr_scalar(cv, hiding_rand)
    commitment_randomness_s = _fr_scalar(cv, commitment_randomness)

    @frx.jit
    def _core(basis: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
        key = IpaKey(basis=basis, u=u, s=s_pt)
        fs = ipa_challenger.ark_challenger(cv, params)
        _fs, _value, zkp, final, _mod_commitment = _open_one_zk(
            key, commitment, coeffs_arr, x, hiding_poly, hiding_rand_s,
            commitment_randomness_s, fs)
        hcomm = lax.convert_element_type(zkp.hiding_comm, key.basis.dtype)
        return zkp.l, zkp.r, final, zkp.a, hcomm, zkp.rand

    return _core


def open_zk(
    cv: Curve, params: Any, svk_h: np.ndarray, s: np.ndarray, generators: list[np.ndarray],
    combined_commitment: np.ndarray, point: int, coeffs: np.ndarray,
    hiding_poly_raw: list[int], hiding_rand: int, commitment_randomness: int,
) -> IpaProof:
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

    hiding_poly = _pad_hiding_poly(cv, hiding_poly_raw, n)
    hiding_rand_s = _fr_scalar(cv, hiding_rand)
    commitment_randomness_s = _fr_scalar(cv, commitment_randomness)

    fs = ipa_challenger.ark_challenger(cv, params)
    _fs, _value, zkp, final_comm_key, _mod_commitment = _open_one_zk(
        key, commitment, coeffs_arr, x, hiding_poly, hiding_rand_s,
        commitment_randomness_s, fs)

    final = _affine(cv, final_comm_key)
    l_vec = [_affine(cv, zkp.l[j]) for j in range(zkp.l.shape[0])]
    r_vec = [_affine(cv, zkp.r[j]) for j in range(zkp.r.shape[0])]
    return IpaProof(
        l_vec, r_vec, final, _c_int(cv, zkp.a),
        hiding_comm=_affine(cv, zkp.hiding_comm), rand=_c_int(cv, zkp.rand))
