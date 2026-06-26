"""Jit-able ark-sponge challenge squeeze — the jit/GPU-exportable replacement
for `sponge.squeeze_bits`/`squeeze_nonnative`/`squeeze_challenges` (the
jit/GPU-exportable port).

Those decode each squeezed Fq element to a Python bigint and slice bits in a
Python loop. Here the same ark-sponge bit math runs in jax: bitcast the squeezed
Fq elements to their canonical LE bytes (`bitcast_convert_type` returns the
standard-form bytes — verified equal to `.tobytes()`), expand to a per-element
low-`CAPACITY` bit stream, window into `size`-bit challenges, and repack each
window LE into an Fr field element. The Poseidon squeeze itself is already jax
(`zorch.hash.DuplexSponge.squeeze`), so the whole challenge derivation composes
under one jit.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax

import numpy as np

from .curve import Curve
from .sponge import FQ_CAPACITY

_FE_BYTES = 32  # BigInteger256 field repr
_LIMB_BITS = 32


@functools.partial(jax.jit, static_argnames=("k", "size", "cv"))
def challenges_from_fq(fq_elems: jax.Array, k: int, size: int, cv: Curve) -> jax.Array:
    """`k` truncated challenges as an `(k,)` ``cv.fr`` array, from the squeezed
    ``cv.fq`` elements `fq_elems`.

    Faithful to ark-sponge `squeeze_nonnative_field_elements_with_sizes` for the
    uniform-`size` case the accumulation prover uses (`squeeze_challenges`): the
    bit stream is the low `FQ_CAPACITY` bits of each Fq element concatenated, and
    each challenge is the next `size` bits packed LE (`size <= fr_capacity`, so no
    reduction). `fq_elems` must hold `ceil(k*size / FQ_CAPACITY)` elements, and
    `size` is a multiple of the 32-bit limb width (the prover's 128 is). `cv` is a
    static jit arg (a `Curve` is hashable), so its `fr`/`fr_modulus` are trace
    constants.
    """
    # (n, 32) canonical LE bytes -> (n, 256) LE bits -> low FQ_CAPACITY per element.
    byts = lax.bitcast_convert_type(fq_elems, jnp.uint8)
    bit_pos = jnp.arange(8, dtype=jnp.uint8)
    bits = ((byts[:, :, None] >> bit_pos) & 1).astype(jnp.uint8)
    bits = bits.reshape(byts.shape[0], 8 * _FE_BYTES)[:, :FQ_CAPACITY]
    stream = bits.reshape(-1)[: k * size]

    # Window into k challenges and pack each `size`-bit LE window into uint32
    # limbs, then recombine in fr by place value: `Σ_i limb_i · (2^32)^i`. (The
    # field dtype has no limbs->field bitcast, so the lift is field arithmetic;
    # each limb < 2^32 < r is canonical.)
    used = size // _LIMB_BITS
    limb_pos = jnp.arange(_LIMB_BITS, dtype=jnp.uint32)
    limb_val = jnp.sum(
        stream.reshape(k, used, _LIMB_BITS).astype(jnp.uint32) << limb_pos, axis=2
    )
    place = jnp.asarray(
        np.array([pow(1 << _LIMB_BITS, i, cv.fr_modulus) for i in range(used)], dtype=cv.fr)
    )
    return jnp.sum(limb_val.astype(cv.fr) * place[jnp.newaxis, :], axis=1)


def squeeze_challenges(sp, k, size, cv):  # type: ignore[no-untyped-def]
    """ark-sponge `squeeze_challenges`: squeeze the `ceil(k*size / FQ_CAPACITY)`
    Fq elements the bit stream needs, extract `k` `size`-bit challenges as an
    `(k,)` ``cv.fr`` array. Returns `(sponge, challenges)` (the sponge squeeze is
    the already-jax `DuplexSponge.squeeze`)."""
    n_elems = (k * size + FQ_CAPACITY - 1) // FQ_CAPACITY
    sp, elems = sp.squeeze(n_elems)
    return sp, challenges_from_fq(jnp.asarray(elems), k, size, cv)
