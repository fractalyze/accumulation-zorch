"""Classic Poseidon Fiat-Shamir sponge over Fq, faithful to ark-sponge.

Builds zorch's `Poseidon` permutation + `DuplexSponge` (add-absorb) over the
default ark-sponge `PoseidonSponge<ark_pallas::Fq>` parameters — full=8,
partial=31, alpha=17, mds=[[1,0,1],[1,1,0],[0,1,1]], rate=2, capacity=1,
width=3. The 117 round constants are arkworks' (drawn from
`ChaChaRng::seed_from_u64(123456789)`); they ride in as raw canonical LE bytes
so there is no encoding ambiguity.
"""

import frx
import frx.numpy as jnp
import numpy as np
from frx import lax
from zorch.hash.duplex_sponge import DuplexSponge
from zorch.hash.poseidon.params import PoseidonParams
from zorch.hash.poseidon.poseidon import Poseidon

from .curve import PALLAS, Curve

WIDTH = 3
ALPHA = 17
FULL_ROUNDS = 8
PARTIAL_ROUNDS = 31
RATE = 2
MDS = ((1, 0, 1), (1, 1, 0), (0, 1, 1))

# Usable bits per squeezed element = field CAPACITY = MODULUS_BITS - 1. Both Pasta
# primes are 255-bit, so the fq/fr capacities (= 254) are identical for Pallas and
# Vesta — the squeeze bit-math below is curve-invariant. (Read off PALLAS; VESTA
# gives the same values.)
FQ_CAPACITY = PALLAS.fq_capacity
FR_CAPACITY = PALLAS.fr_capacity
CHALLENGE_SIZE = 128  # ark r1cs_nark_as / hp_as CHALLENGE_SIZE


def poseidon_params(cv: Curve, ark_le: bytes) -> PoseidonParams:
    """Build the ``cv.fq`` Poseidon params from the 117 ARK constants as raw LE
    bytes (row-major over ``(full+partial, width)``). The constants are the curve's
    base-field elements, so the caller supplies the Pallas-Fq or Vesta-Fq set."""
    total_rounds = FULL_ROUNDS + PARTIAL_ROUNDS
    rc = np.frombuffer(ark_le, dtype=cv.fq).reshape(total_rounds, WIDTH).copy()
    mds = np.array(MDS, dtype=cv.fq)
    return PoseidonParams(
        width=WIDTH,
        dtype=cv.fq,
        alpha=ALPHA,
        full_rounds=FULL_ROUNDS,
        partial_rounds=PARTIAL_ROUNDS,
        round_constants=rc,
        mds=mds,
    )


def new_sponge(params: PoseidonParams) -> DuplexSponge:
    """A fresh add-absorb duplex sponge over the classic Poseidon permutation."""
    return DuplexSponge(Poseidon(params), rate=RATE)


def squeeze_bits(sp: DuplexSponge, num_bits: int) -> tuple[DuplexSponge, list[int]]:
    """ark-sponge `squeeze_bits`: squeeze `ceil(num_bits/CAPACITY)` Fq elements,
    take each one's low `CAPACITY` LE bits, concatenate, truncate to `num_bits`."""
    usable = FQ_CAPACITY
    n_elems = (num_bits + usable - 1) // usable
    sp, elems = sp.squeeze(n_elems)
    vals = np.asarray(elems)
    bits: list[int] = []
    for i in range(n_elems):
        e = int.from_bytes(vals[i].tobytes(), "little")  # canonical (non-Montgomery)
        bits.extend((e >> b) & 1 for b in range(usable))
    return sp, bits[:num_bits]


def squeeze_nonnative(
    sp: DuplexSponge, sizes: list[int]
) -> tuple[DuplexSponge, list[int]]:
    """ark-sponge `squeeze_nonnative_field_elements_with_sizes`: one bit stream of
    `sum(sizes)` bits, sliced into per-element LE windows. Each `sizes[i]` is a
    bit count (already capped at the target field's CAPACITY); the window packs
    LE into the integer value of an Fr element (no reduction — sizes < modulus)."""
    sp, bits = squeeze_bits(sp, sum(sizes))
    out: list[int] = []
    off = 0
    for nb in sizes:
        v = 0
        for i in range(nb):
            if bits[off + i]:
                v += 1 << i
        off += nb
        out.append(v)
    return sp, out


def squeeze_challenges(
    sp: DuplexSponge, k: int, size: int = CHALLENGE_SIZE
) -> tuple[DuplexSponge, list[int]]:
    """`k` truncated challenges as Fr-valued ints (each `min(size, FR_CAPACITY)` bits).

    The host (eager, Python-loop) squeeze — used by the host Fiat-Shamir combines
    that produce constants baked into the fused cores (`ipa_pc`, `ipa_pc_as`). It
    handles arbitrary bit widths, including the 184-bit AS opening point that
    :func:`squeeze_challenges_frx` cannot (see below)."""
    return squeeze_nonnative(sp, [min(size, FR_CAPACITY)] * k)


# --- jit-able / device squeeze -------------------------------------------------
# The same ark-sponge bit math as `squeeze_bits`/`squeeze_nonnative`, but as frx
# array ops returning an `fr` array — so the challenge derivation composes under
# the fused prove cores' outermost `@frx.jit` and lowers to GPU. NOT a superset of
# the host squeeze: `challenges_from_fq` only handles `size` a multiple of the
# 32-bit limb width (the prover's 128 is), so the arbitrary-width host squeeze
# stays for e.g. the 184-bit AS opening point.
_FE_BYTES = 32  # BigInteger256 field repr
_LIMB_BITS = 32


def challenges_from_fq(fq_elems: frx.Array, k: int, size: int, cv: Curve) -> frx.Array:
    """`k` truncated challenges as an `(k,)` ``cv.fr`` array, from the squeezed
    ``cv.fq`` elements `fq_elems`.

    Faithful to ark-sponge `squeeze_nonnative_field_elements_with_sizes` for the
    uniform-`size` case the accumulation prover uses (`squeeze_challenges_frx`): the
    bit stream is the low `FQ_CAPACITY` bits of each Fq element concatenated, and
    each challenge is the next `size` bits packed LE (`size <= fr_capacity`, so no
    reduction). `fq_elems` must hold `ceil(k*size / FQ_CAPACITY)` elements, and
    `size` is a multiple of the 32-bit limb width (the prover's 128 is). `cv` is
    captured as a trace constant when this leaf inlines into the boundary jit, so
    its `fr`/`fr_modulus` are trace constants.
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


def squeeze_challenges_frx(sp, k, size, cv):  # type: ignore[no-untyped-def]
    """ark-sponge `squeeze_challenges` as frx: squeeze the `ceil(k*size / FQ_CAPACITY)`
    Fq elements the bit stream needs, extract `k` `size`-bit challenges as an
    `(k,)` ``cv.fr`` array. Returns `(sponge, challenges)` (the sponge squeeze is
    the already-frx `DuplexSponge.squeeze`). The device twin of
    :func:`squeeze_challenges`, for the in-trace prover challenges (all 128-bit)."""
    n_elems = (k * size + FQ_CAPACITY - 1) // FQ_CAPACITY
    sp, elems = sp.squeeze(n_elems)
    return sp, challenges_from_fq(jnp.asarray(elems), k, size, cv)
