"""ark-sponge `Absorbable` field-element packing + `fork()`, over the curve's fq.

`PoseidonSponge::absorb(x)` consumes `x.to_sponge_field_elements()` — so every
absorbed object is first packed into a list of fq elements. This ports that
packing for the types the classic accumulation Fiat-Shamir layer absorbs:

* **bytes** (`&[u8]` / `Vec<u8>`): ark-ff `[u8]::to_field_elements()` chunks the
  bytes into `CAPACITY/8 = 31`-byte little-endian groups, each zero-padded to a
  full field repr (32B) and read as a canonical LE integer → fq. The `u8`
  `batch_to_sponge_field_elements` first prepends `(len as u64).to_le_bytes()`.
* **SW-affine point**: `[x, y, infinity]`. arkworks' `Affine::zero()` is
  `(x=0, y=1, infinity=true)` — packed as `[0, 1, 1]`, NOT the all-zero
  zk_dtypes encoding. This is the one point→field trap.
* **fork(domain)**: absorbs `(len(domain) as u64).LE ‖ domain` as a `Vec<u8>`
  Absorbable — i.e. a *double* length-prefix (the inner `to_sponge_bytes(usize)`
  then the outer `u8` batch prefix) before the 31-byte chunking.

Field elements ride in as canonical LE bytes (the same path that loads the 117
Poseidon ARK constants), so there is no Montgomery-encoding ambiguity. Both Pasta
fields are 254-cap, so the 31-byte chunk size is curve-invariant; the curve only
sets which `fq` dtype the bytes are read as.
"""

import struct

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from zorch.hash.duplex_sponge import DuplexSponge

from . import curve, jcurve
from .curve import Curve

# ark-ff `[u8]::to_field_elements` chunk size = CAPACITY / 8 (both Pasta fq are
# CAPACITY 254 → 31). Curve-invariant.
_BYTES_PER_FE = 31
_FE_REPR_BYTES = 32  # BigInteger256: 4 limbs × 8 bytes


def bytes_to_field_array(cv: Curve, data: bytes) -> np.ndarray:
    """ark-ff `[u8]::to_field_elements()` for fq: 31-byte LE chunks, each
    zero-padded to a 32-byte field repr and read as a canonical LE integer."""
    if len(data) == 0:
        return np.frombuffer(b"", dtype=cv.fq).copy()
    buf = bytearray()
    for i in range(0, len(data), _BYTES_PER_FE):
        chunk = data[i : i + _BYTES_PER_FE]
        buf += chunk + b"\x00" * (_FE_REPR_BYTES - len(chunk))
    return np.frombuffer(bytes(buf), dtype=cv.fq).copy()


def u8_batch_field_array(cv: Curve, data: bytes) -> np.ndarray:
    """`u8::batch_to_sponge_field_elements`: prepend `(len as u64).LE`, then chunk."""
    return bytes_to_field_array(cv, struct.pack("<Q", len(data)) + data)


def absorb_bytes(cv: Curve, sp: DuplexSponge, data: bytes) -> DuplexSponge:
    """Absorb a `&[u8]` / `Vec<u8>` Absorbable into the sponge."""
    return sp.absorb(jnp.asarray(u8_batch_field_array(cv, data)))


def point_to_field_array(cv: Curve, point: np.ndarray) -> np.ndarray:
    """SW-affine `to_field_elements()` = `[x, y, infinity]`. The identity packs
    as `[0, 1, 1]` (arkworks `Affine::zero()`), not the all-zero coords."""
    if curve.is_infinity(cv, point):
        x_le, y_le, inf = (0).to_bytes(32, "little"), (1).to_bytes(32, "little"), 1
    else:
        x, y = curve.point_coords(cv, point)
        x_le, y_le, inf = x.to_bytes(32, "little"), y.to_bytes(32, "little"), 0
    return np.frombuffer(x_le + y_le + inf.to_bytes(32, "little"), dtype=cv.fq).copy()


def absorb_point(cv: Curve, sp: DuplexSponge, point: np.ndarray) -> DuplexSponge:
    """Absorb an SW-affine point Absorbable into the sponge (packed in-jit)."""
    return sp.absorb(point_to_field_array_jax(cv, jcurve.stack_affine(cv, [point])))


def fork(cv: Curve, sp: DuplexSponge, domain: bytes) -> DuplexSponge:
    """`CryptographicSponge::fork(domain)`: domain separation by absorbing
    `(len(domain) as u64).LE ‖ domain` as a `Vec<u8>` Absorbable."""
    inp = struct.pack("<Q", len(domain)) + domain
    return absorb_bytes(cv, sp, inp)


def _fe_array(cv: Curve, values: list[int]) -> np.ndarray:
    """fq array from canonical integer values (each ``< p``, via 32-byte LE repr);
    ``to_bytes(32)`` raises on a wider int rather than silently reducing."""
    return np.frombuffer(b"".join(int(v).to_bytes(32, "little") for v in values), dtype=cv.fq)


def point_to_field_array_jax(cv: Curve, points: jax.Array) -> jax.Array:
    """In-jit `point_to_field_array` for a batch of SW-affine points.

    `points` is an ``(N,)`` ``cv.g1`` jax array (e.g. a stacked set of commitments
    straight off ``lax.msm``); returns the ``(3N,)`` fq packing
    ``[x0, y0, inf0, x1, y1, inf1, …]`` — the same bytes as concatenating the host
    :func:`point_to_field_array` over the points, but without leaving the device,
    so the Fiat-Shamir point absorbs trace into the prove instead of forcing a host
    hop at every commitment.

    The coordinate reinterpret is ``lax.bitcast_convert_type(affine → fq)``;
    identity is the all-zero zk_dtypes encoding — both coordinates
    equal to ``fq(0)`` — packed as arkworks' ``[0, 1, 1]`` via ``select``, the one
    point→field trap, kept identical to the host path. Plain (un-jitted) so it
    inlines into the single ``@jax.jit`` prove.

    Identity is detected by fq field equality (``coord == 0``), NOT a second
    ``fq → uint8`` bitcast: the zkx GPU plugin mis-lowers the field→bytes bitcast on
    a rank-2 field tensor (``'tensor.extract' op incorrect number of indices``),
    whereas the field comparison lowers cleanly on both CPU and GPU and is
    byte-identical (``fq(0)`` is the all-zero canonical encoding).
    """
    fq_zero = jnp.asarray(np.array([0], dtype=cv.fq))[0]
    fq_one = jnp.asarray(np.array([1], dtype=cv.fq))[0]
    coords = lax.bitcast_convert_type(points, cv.fq)  # (N, 2): [x, y]
    inf = jnp.all(coords == fq_zero, axis=-1)  # (N,): both coords zero ⇒ identity
    x = jnp.where(inf, fq_zero, coords[:, 0])
    y = jnp.where(inf, fq_one, coords[:, 1])
    flag = jnp.where(inf, fq_one, fq_zero)
    return jnp.stack([x, y, flag], axis=-1).reshape(-1)  # (3N,)


def absorb_u64(cv: Curve, sp: DuplexSponge, value: int) -> DuplexSponge:
    """Absorb a `u64` Absorbable: `to_sponge_field_elements` = `[F::from(v)]`."""
    return sp.absorb(jnp.asarray(_fe_array(cv, [value])))


def absorb_none(cv: Curve, sp: DuplexSponge) -> DuplexSponge:
    """Absorb an `Option::None` Absorbable: `[F::from(false)]` = `[0]`."""
    return sp.absorb(jnp.asarray(_fe_array(cv, [0])))


def option_flag(cv: Curve, is_some: bool) -> np.ndarray:
    """The leading element of `Option<_>`'s `to_sponge_field_elements`:
    `[F::from(is_some)]` — a single fq (`0` for `None`, `1` for `Some`). Used when
    an Option is packed inline into a larger field-element vector (e.g. the
    `FirstRoundMessage` absorb in the gamma challenge)."""
    return _fe_array(cv, [1 if is_some else 0])


def absorb_option_points(cv: Curve, sp: DuplexSponge, points: list[np.ndarray]) -> DuplexSponge:
    """Absorb an `Option<_>` whose inner Absorbable is a batch of points, in the
    `Some` case: a single `F::from(true)` flag, then each point's `[x, y,
    infinity]` — all in one absorb (e.g. `Some(ProofHidingCommitments)`)."""
    arr = jnp.concatenate([jnp.asarray(option_flag(cv, True)),
                           point_to_field_array_jax(cv, jcurve.stack_affine(cv, points))])
    return sp.absorb(arr)


def absorb_points(cv: Curve, sp: DuplexSponge, points: list[np.ndarray]) -> DuplexSponge:
    """Absorb a batch of SW-affine points in one call (e.g. a `Vec<InputInstance>`
    flattened to its commitments, or a `ProductPolynomialCommitment`'s low‖high):
    the concatenation of each point's `[x, y, infinity]`."""
    if not points:
        return sp
    return sp.absorb(point_to_field_array_jax(cv, jcurve.stack_affine(cv, points)))


def absorb_points_jax(cv: Curve, sp: DuplexSponge, points: jax.Array) -> DuplexSponge:
    """`absorb_points` for a pre-stacked `(N,)` jax affine array — the commitments
    threaded straight off `lax.msm` (no host re-stack), so the absorb traces into
    the single-`@jax.jit` prove."""
    return sp.absorb(point_to_field_array_jax(cv, points))


def absorb_option_points_jax(cv: Curve, sp: DuplexSponge, points: jax.Array) -> DuplexSponge:
    """`absorb_option_points` for a pre-stacked `(N,)` jax affine array (the `Some`
    case): the `F::from(true)` flag then each point's `[x, y, infinity]`, all in one
    absorb — the in-trace twin of `absorb_option_points`."""
    return sp.absorb(jnp.concatenate([jnp.asarray(option_flag(cv, True)),
                                      point_to_field_array_jax(cv, points)]))
