"""Pasta cycle curves as a pure dtype surface, plus arkworks point serialization.

A :class:`Curve` bundles the three zk_dtypes scalar types that differ between
Pallas and Vesta — the affine G1 dtype and the scalar (``fr``) / base (``fq``)
field dtypes — and the field facts derived from them. The dtypes ARE the
constructors, so there is no wrapper layer:

* ``cv.fr(v)`` builds a scalar field element. It is the canonical (non-Montgomery)
  dtype, so ``cv.fr(v).tobytes()`` is the 32-byte LE wire form arkworks'
  ``CanonicalSerialize`` emits.
* ``cv.g1((x, y))`` builds an affine point — raw 64-byte ``x ‖ y`` (each a 32-byte
  LE ``fq`` coordinate); the identity is all-zero coords (arkworks
  ``Affine::zero()``).

Operations are free functions over those arrays — ``lax.msm`` consumes them
directly — so a curve only appears at host-side construction / serialization; the
jit kernels never name one.

The Pasta cycle is a type-level swap: ``vesta.fr == pallas.fq`` and
``vesta.fq == pallas.fr`` (ark-vesta re-exports ark-pallas's fields). The mapping
is pinned by **modulus** (and a generator byte-match), NOT by the zk_dtypes
``_sf`` names, which are inverted relative to arkworks:

* ``fr`` = scalar field: r1cs witness, blinded witness, squeezed challenges.
  Pallas ``fr`` is ``pallas_sf`` (``ark_pallas::Fr``).
* ``fq`` = base field: point coordinates and the Poseidon / Fiat-Shamir
  constraint field. Pallas ``fq`` is ``vesta_sf`` (``ark_pallas::Fq``).

(``ecinfo(pallas_g1_affine).base_field`` reports ``pallas_sf`` — a red herring;
trust the modulus.)

arkworks (ark-serialize 0.2) ``CanonicalSerialize`` for a short-Weierstrass affine
point is **compressed = 33 bytes**: the 32-byte LE x-coordinate followed by one
flag byte (``0x40`` infinity, ``0x80`` when ``y > p - y``, else ``0x00``). The
2-bit SW flag needs its own trailing byte because the Pasta base modulus top byte
(``0x40``) leaves only one spare high bit.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import zk_dtypes as zk

# SW flag byte values (ark-serialize 0.2 `SWFlags`).
_FLAG_INFINITY = 0x40
_FLAG_NEG_Y = 0x80
_FLAG_POS_Y = 0x00


@dataclass(frozen=True)
class Curve:
    """A Pasta cycle curve as its zk_dtypes scalar-type triple (the constructors)
    plus the field facts they imply. Data only — every operation is a free
    function taking a ``Curve``. ``g1``/``fr``/``fq`` are callable: ``cv.fr(v)``,
    ``cv.g1((x, y))``; they also serve as the numpy dtype for array construction
    (``np.array(values, dtype=cv.fr)``)."""

    name: str
    g1: Any  # affine G1 dtype: cv.g1((x, y)) -> 64B x‖y; identity = all-zero coords
    fr: Any  # scalar field dtype: cv.fr(v) canonical, .tobytes() = 32B LE
    fq: Any  # base field dtype: point coords + Poseidon / Fiat-Shamir constraint field

    @property
    def fr_modulus(self) -> int:
        return int(zk.pfinfo(self.fr).modulus)

    @property
    def fq_modulus(self) -> int:
        return int(zk.pfinfo(self.fq).modulus)

    @property
    def fr_capacity(self) -> int:
        """Usable bits per squeezed scalar = ``MODULUS_BITS - 1`` (ark field CAPACITY)."""
        return self.fr_modulus.bit_length() - 1

    @property
    def fq_capacity(self) -> int:
        """Usable bits per squeezed base-field element = ``MODULUS_BITS - 1``."""
        return self.fq_modulus.bit_length() - 1


# Pinned by modulus / a generator byte-match (NOT the zk_dtypes `_sf` names).
PALLAS = Curve("pallas", zk.pallas_g1_affine, zk.pallas_sf, zk.vesta_sf)
VESTA = Curve("vesta", zk.vesta_g1_affine, zk.vesta_sf, zk.pallas_sf)


def point_coords(cv: Curve, point: np.ndarray) -> tuple[int, int]:
    """The ``(x, y)`` integer coordinates of an affine point. ``dtype=cv.g1``
    normalizes a projective (``*``/``+``) result back to affine before the read —
    zk_dtypes' numpy group ops return a 96-byte jacobian, and the affine cast is
    where the prove path's serialization recovers the ``x ‖ y`` form."""
    raw = np.asarray(point, dtype=cv.g1).tobytes()
    return int.from_bytes(raw[:32], "little"), int.from_bytes(raw[32:64], "little")


def is_infinity(cv: Curve, point: np.ndarray) -> bool:
    """True for the arkworks identity — all-zero coords (``Affine::zero()``), the
    convention :func:`point_to_bytes` and the sponge point-packing rely on. NB
    ``0·G`` in zk_dtypes has a *different*, non-all-zero encoding, so this must not
    be expressed as ``point == 0·G``. ``dtype=cv.g1`` normalizes a jacobian to
    affine first."""
    return np.asarray(point, dtype=cv.g1).tobytes() == b"\x00" * 64


def point_to_bytes(cv: Curve, point: np.ndarray) -> bytes:
    """arkworks ``CanonicalSerialize`` (compressed, 33 bytes) of an affine point."""
    if is_infinity(cv, point):
        return b"\x00" * 32 + bytes([_FLAG_INFINITY])
    x, y = point_coords(cv, point)
    # y is canonical (< fq_modulus), so fq_modulus - y needs no reduction: for
    # y == 0 it gives fq_modulus, but `0 > fq_modulus` is false → POS flag, the
    # same outcome arkworks' `y > (p - y) mod p` produces.
    neg_y = cv.fq_modulus - y
    flag = _FLAG_NEG_Y if y > neg_y else _FLAG_POS_Y
    return x.to_bytes(32, "little") + bytes([flag])


def pedersen_commit(
    cv: Curve,
    generators: list[np.ndarray],
    elems: Any,
    hiding: np.ndarray | None = None,
    randomizer: Any | None = None,
) -> np.ndarray:
    """``Σ generatorsᵢ·elemsᵢ (+ randomizer·hiding)`` as a CPU group reduction —
    the byte-match oracle (NOT the jit/``lax.msm`` prove path, which is
    :mod:`jcurve`). Byte-identical to arkworks ``PedersenCommitment::commit`` (the
    sum is associative, so the hiding term folds in as one extra ``(base, scalar)``
    pair). ``generators``/``hiding`` are affine point arrays.

    ``elems`` / ``randomizer`` are ``fr`` scalars: an ``fr`` array (a jit kernel's
    output or a host field array) or an int list (e.g. sponge-squeezed challenges)
    — ``np.asarray(_, dtype=cv.fr)`` normalizes both to canonical ``fr`` and the
    element multiplies the base directly, so no ``fr`` value round-trips through a
    python int here.

    zk_dtypes' ``point * fr`` / ``point + point`` produce a jacobian; the running
    sum stays jacobian and is normalized back to affine on return (the affine
    cast :func:`point_coords` would also do), so the result serializes as ``x ‖ y``.
    """
    scalars = np.asarray(elems, dtype=cv.fr)
    terms = [g * s for g, s in zip(generators, scalars)]
    if randomizer is not None:
        # `.reshape(-1)[0]` is the single `fr` scalar element `point * fr` accepts —
        # the same element type iterating `scalars` yields (a bare 0-d array is not).
        terms.append(hiding * np.asarray(randomizer, dtype=cv.fr).reshape(-1)[0])
    acc = terms[0]
    for t in terms[1:]:
        acc = acc + t
    return np.asarray(acc, dtype=cv.g1)
