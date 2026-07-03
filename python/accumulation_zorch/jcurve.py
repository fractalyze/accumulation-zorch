"""Jit-able Pasta-G1 Pedersen commitments — the jit/GPU-exportable replacement
for `curve.py`'s `zk_dtypes` CPU group ops (the jit/GPU-exportable port).

`curve.pedersen_commit` sums `g * cv.fr(s)` over the affine dtype — CPU-only group
ops that don't lower to a jit/GPU kernel. Here the commitment is `lax.msm` (the
Pasta G1 MSM kernel), and any field reduction feeding it (`M·z`)
runs as vectorized jax over the `fr` dtype. The op is GPU-ready for Phase 2;
Phase 1 gates it on CPU (`JAX_PLATFORMS=cpu`).

A curve appears only where a host-side array is built from the curve's dtypes
(`stack_affine` over `cv.g1`, the `cv.fr` randomizer/challenge arrays). The
`commit_dense` / `msm` kernels are dtype-agnostic — the dtype rides on the input
arrays — so they name no curve.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import jfield
from .curve import Curve


def stack_affine(cv: Curve, points: list[np.ndarray]) -> jax.Array:
    """Stack a list of affine points into one `(n,)` G1 array — the `bases` layout
    `commit_dense`/`lax.msm` consume. `dtype=cv.g1` normalizes each point (a
    `cv.g1((x, y))` scalar, or a jacobian from a CPU group op) to the affine form
    before the byte concat."""
    raw = b"".join(np.asarray(p, dtype=cv.g1).tobytes() for p in points)
    return jnp.asarray(np.frombuffer(raw, dtype=cv.g1).copy())


def commit_dense(coeffs: jax.Array, z: jax.Array, bases: jax.Array) -> jax.Array:
    """`commit(M·z) = Σ_i (Σ_j coeffs[i,j]·z[j]) · bases[i]` — the first-round
    NARK commitment, computed entirely in jax.

    `coeffs` is a dense `(rows × vars)` `fr` matrix (a sparse `Matrix<Fr>`
    densified host-side), `z` the `(vars,)` `fr` vector (`r1cs_input ‖ witness`),
    and `bases` the `(rows,)` G1 affine generators. The `M·z` reduction is a
    broadcast multiply-and-sum over `fr` (not `@`/`einsum`, which the jax fork
    doesn't lower for the field dtype); the commitment is one `lax.msm` →
    a single affine point, byte-identical to `PedersenCommitment::commit`.
    """
    return lax.msm(jfield.matvec(coeffs, z), bases)


def msm(scalars: jax.Array, bases: jax.Array) -> jax.Array:
    """`Σ scalars[i]·bases[i]` — commit a coefficient/scalar vector, or fold a set
    of points under challenges. The one trusted jit scalar-mul / point-fold
    primitive: jit `point × scalar` is byte-wrong, and jit `affine + affine` is an
    invalid EC type combination, so every scalar-mul AND point-add recasts to a
    `lax.msm` (a single `s·P` is `msm([s],[P])`; `A + B` is `msm([1,1],[A,B])`)."""
    return lax.msm(scalars, bases)


def commit_hiding(cv: Curve, scalars: jax.Array, randomizer: int | jax.Array,
                  bases_h: jax.Array) -> jax.Array:
    """`Σ scalars[i]·bases_h[i] + randomizer·bases_h[-1]` — a Pedersen commitment
    with a hiding term, as one `lax.msm`. `bases_h` is the pre-stacked generators
    with the hiding base appended last (so the randomizer rides as the extra MSM
    term); `scalars` an `(n,)` `fr` array threaded straight from the `M·z` /
    cross-term compute. Returns the on-device point — `bases_h` is a jit argument
    (an affine-typed constant doesn't lower), the export-correct shape since the
    committer key is a runtime input.

    `randomizer` is a host int (the baked half-step / fold path) or a runtime `fr`
    device scalar (the general prover, where the randomness is a runtime
    input); either rides as the trailing MSM term, byte-identically."""
    if isinstance(randomizer, (int, np.integer)):
        rand = jnp.asarray(np.array([randomizer], dtype=cv.fr))
    else:
        rand = jnp.asarray(randomizer).reshape(1)
    full = jnp.concatenate([scalars, rand])
    return msm(full, bases_h)


def combine(cv: Curve, points: list[np.ndarray], challenges: list[int]) -> np.ndarray | None:
    """`combine_commitments`: `Σ points[i]·challenges[i]` as one `lax.msm`. `None`
    for an empty input (the additive identity ark starts the fold from), so the
    single-input case — where `low`/`high` are empty — needs no identity point."""
    if not points:
        return None
    scalars = jnp.asarray(np.array(challenges[: len(points)], dtype=cv.fr))
    return np.asarray(msm(scalars, stack_affine(cv, points)))
