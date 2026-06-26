"""Classic Poseidon Fiat-Shamir sponge over Fq, faithful to ark-sponge.

Builds zorch's `Poseidon` permutation + `DuplexSponge` (add-absorb) over the
default ark-sponge `PoseidonSponge<ark_pallas::Fq>` parameters — full=8,
partial=31, alpha=17, mds=[[1,0,1],[1,1,0],[0,1,1]], rate=2, capacity=1,
width=3. The 117 round constants are arkworks' (drawn from
`ChaChaRng::seed_from_u64(123456789)`); they ride in as raw canonical LE bytes
so there is no encoding ambiguity.
"""

import numpy as np
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
    """`k` truncated challenges as Fr-valued ints (each `min(size, FR_CAPACITY)` bits)."""
    return squeeze_nonnative(sp, [min(size, FR_CAPACITY)] * k)
