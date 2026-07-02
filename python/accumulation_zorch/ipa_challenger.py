"""`ArkIpaChallenger` — the arkworks-faithful IPA-PC Fiat-Shamir, in JAX, in-trace.

zorch's IPA fold (`zorch.pcs.ipa.challenger.IpaChallenger`) derives its round
challenges through a challenger it carries as a `lax.scan` carry. This is the
accumulation-consumer's challenger: a byte-exact re-derivation of
`ipa_pc.succinct_check_challenges` (ark-poly-commit `ipa_pc::succinct_check`, no-zk)
built entirely on the jit-able `jsponge` / `absorbable` primitives, so the same
challenger drives the CPU port and the fused GPU core and yields byte-identical
challenges.

The reproduced Fiat-Shamir (faithful to `ipa_pc._round_challenges_from_seed`):

* Every challenge is squeezed from a **fresh** ``"IPA-PC-2020"``-forked
  ``DomainSeparatedSponge`` — the seed round AND each per-round L/R squeeze — NOT
  one running sponge. The only state carried between rounds is the previous
  challenge value.
* ``seed(commitment, point, value)``: absorb the (combined) commitment point, then
  ``to_bytes![point, value]`` (each a 32-byte LE ``fr``), squeeze the seed
  challenge ξ₀ (the inner-product generator scale ``h' = svk.h·ξ₀``). ξ₀ becomes
  the first ``prev``.
* ``challenge(l, r)``: absorb the previous challenge (its low ``CHALLENGE_BYTES``
  = 16 bytes, i.e. ``int(prev).to_bytes(16, "little")``), then ``L``, then ``R``;
  squeeze the round challenge ξ_j.

Encodings are byte-identical to the NumPy oracle:

* **points** (commitment / L / R) ride in through the in-trace
  ``absorbable.absorb_points_jax`` — the ``[x, y, infinity]`` SW-affine packing —
  so a device-resident commitment threads straight into the sponge with no host
  hop (the fold's L/R are on-device under the scan).
* the **previous challenge** absorb is reproduced with field arithmetic rather
  than a host ``int().to_bytes`` so ``challenge`` traces cleanly. The byte packing
  ``u8::batch_to_sponge_field_elements`` of ``prev.to_bytes(16, "little")`` is a
  single ``fq`` element: an 8-byte ``u64`` length prefix (value ``16``) in the low
  bytes, then the 16-byte challenge, zero-padded to the 32-byte repr — i.e.
  ``fq(CHALLENGE_BYTES + prev·2**64)``. ``prev`` (an ``fr`` value < 2**128) is
  reinterpreted to ``fq`` by its canonical LE bytes (``fr → u8 → fq`` bitcast),
  and ``prev·2**64 < 2**192 < fq_modulus`` stays canonical.
* the **seed** scalars ``point`` / ``value`` are bound once (before the rounds,
  host-side) via the ``_fr32`` 32-byte LE encoding, matching ``ipa_pc._fr32`` and
  the oracle's ``to_bytes![point, value]`` byte-for-byte.

The squeeze is the in-trace truncated-128 nonnative squeeze
(``jsponge.challenges_from_fq`` via ``jsponge.squeeze_challenges``): one ``fq``
element, low 128 bits packed LE into an ``fr`` element (128 ≤ both Pasta scalar
capacities, so no reduction).

Pytree: ``ArkIpaChallenger`` is a ``register_dataclass`` pytree matching
``TranscriptChallenger``'s shape — the sponge field state and the previous
challenge are the data leaves (they ride the fold's ``lax.scan`` carry), the curve
+ Poseidon params + the forked sponge's (static) absorb mode/position are the aux
meta. The base sponge is forked ONCE (`ark_challenger`); each round reconstructs
the working sponge from the carried state, so the domain-fork permute is not
repeated per round.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.tree_util import register_dataclass

from . import absorbable, jsponge, sponge
from .curve import Curve

# ark `ipa_pc` domain (`IpaPCDomain`): every fresh succinct-check sponge is a
# `DomainSeparatedSponge` forked with this label. Mirrors `ipa_pc.IPA_PC_DOMAIN`.
IPA_PC_DOMAIN = b"IPA-PC-2020"

# Each challenge is a `Truncated(CHALLENGE_SIZE=128)` squeeze; both Pasta scalar
# fields are 254-cap > 128, so `min(128, fr_capacity) == 128` is curve-invariant.
_CHALLENGE_SIZE = sponge.CHALLENGE_SIZE

# `to_bytes![round_challenge]` resized to `(CHALLENGE_SIZE + 7) / 8` = 16 bytes —
# the previous (truncated-128) challenge's low 16 bytes, re-absorbed per round.
_CHALLENGE_BYTES = (_CHALLENGE_SIZE + 7) // 8

# `u64::to_le_bytes` is 8 bytes; the `u8` batch length prefix therefore rides in
# the low 8 bytes (64 bits) of the packed field element (see `_absorb_prev`).
_U64_SHIFT = 1 << 64


def _fr32(cv: Curve, value: Array | int) -> bytes:
    """`to_bytes![Fr]` — the 32-byte canonical LE serialization of a scalar. Matches
    `ipa_pc._fr32`. `value` is a host int (the bound opening statement) or a 0-d
    ``fr`` array (read back through its canonical LE bytes)."""
    if isinstance(value, (int, np.integer)):
        v = int(value)
    else:
        v = int.from_bytes(np.asarray(value, dtype=cv.fr).tobytes(), "little")
    return v.to_bytes(32, "little")


@partial(
    register_dataclass,
    data_fields=["state", "prev"],
    meta_fields=["cv", "params", "mode", "pos"],
)
@dataclass(frozen=True)
class ArkIpaChallenger:
    """The arkworks-faithful `IpaChallenger` — and, via :meth:`hiding_challenge`,
    the `ZkIpaChallenger` for the hiding open. A JAX pytree: `state` (the fresh
    `"IPA-PC-2020"`-forked Poseidon sponge's field state) and `prev` (the previous
    round challenge, an ``fr`` scalar) are the data leaves the fold's `lax.scan`
    carries; `cv` / `params` / the forked sponge's absorb `mode`+`pos` are the
    static aux meta.

    Each round rebuilds the working sponge from the carried `state` (no re-fork of
    the domain), absorbs that round's inputs, and squeezes one truncated-128
    challenge. Build with :func:`ark_challenger`."""

    state: Array  # data leaf: the fresh IPA-PC-forked sponge's (width,) fq state
    prev: Array  # data leaf: previous round challenge as an `cv.fr` scalar
    cv: Curve  # aux: curve (the fr/fq dtypes + moduli)
    params: Any  # aux: Poseidon params (value-hashable) — rebuilds the permutation
    mode: str  # aux: the forked sponge's duplex mode ("absorbing")
    pos: int  # aux: the forked sponge's rate position after the domain fork

    def _sponge(self):  # type: ignore[no-untyped-def]
        """Reconstruct the fresh domain-forked sponge from the carried state: a
        zero-state duplex over the Poseidon params, its rate lanes set to the
        forked `state` at the forked `mode`/`pos`."""
        return sponge.new_sponge(self.params)._with(
            state=self.state, mode=self.mode, pos=self.pos
        )

    def _squeeze(self, sp) -> Array:  # type: ignore[no-untyped-def]
        """One truncated-128 challenge as an `cv.fr` scalar (in-trace nonnative
        squeeze), matching `ipa_pc._squeeze_challenge`."""
        _, ch = jsponge.squeeze_challenges(
            sp, 1, min(_CHALLENGE_SIZE, self.cv.fr_capacity), self.cv
        )
        return ch[0]

    def _absorb_prev(self, sp):  # type: ignore[no-untyped-def]
        """Absorb the previous challenge's low 16 bytes in-trace. `u8`-batch packing
        of ``prev.to_bytes(16, "little")`` is one `fq` element:
        ``fq(CHALLENGE_BYTES + prev·2**64)`` (an 8-byte ``u64`` length prefix of
        value 16 in the low bytes, then the 16-byte challenge). `prev` (< 2**128) is
        reinterpreted `fr → fq` via its canonical LE bytes."""
        cv = self.cv
        prev_bytes = lax.bitcast_convert_type(self.prev, jnp.uint8)  # (32,) LE bytes
        prev_fq = lax.bitcast_convert_type(prev_bytes, cv.fq)  # fq(prev)
        length = jnp.asarray(np.array([_CHALLENGE_BYTES], dtype=cv.fq))[0]
        shift = jnp.asarray(np.array([_U64_SHIFT], dtype=cv.fq))[0]
        fe = length + prev_fq * shift
        return sp.absorb(fe[jnp.newaxis])

    def seed(
        self, commitment: Array, point: Array, value: Array
    ) -> tuple[ArkIpaChallenger, Array]:
        """Bind the opening statement and squeeze the seed challenge ξ₀ (the
        inner-product generator scale ``h' = svk.h·ξ₀``): a fresh sponge absorbs
        `commitment`, then ``to_bytes![point, value]``. ξ₀ becomes the first
        `prev`. `point`/`value` are bound host-side (the statement is fixed before
        the rounds)."""
        cv = self.cv
        sp = self._sponge()
        sp = absorbable.absorb_points_jax(cv, sp, jnp.asarray(commitment).reshape(1))
        sp = sp.absorb(
            jnp.asarray(
                absorbable.u8_batch_field_array(cv, _fr32(cv, point) + _fr32(cv, value))
            )
        )
        xi0 = self._squeeze(sp)
        return ArkIpaChallenger(self.state, xi0, cv, self.params, self.mode, self.pos), xi0

    def challenge(self, l: Array, r: Array) -> tuple[ArkIpaChallenger, Array]:
        """One fold round: a fresh sponge absorbs the previous challenge (low 16
        bytes), then `l`, then `r`; squeeze the round challenge ξ_j. Fully in-trace
        (jit/`lax.scan`-safe) — the round inputs and the carried `prev` are all
        device values."""
        cv = self.cv
        sp = self._sponge()
        sp = self._absorb_prev(sp)
        sp = absorbable.absorb_points_jax(cv, sp, jnp.asarray(l).reshape(1))
        sp = absorbable.absorb_points_jax(cv, sp, jnp.asarray(r).reshape(1))
        u = self._squeeze(sp)
        return ArkIpaChallenger(self.state, u, cv, self.params, self.mode, self.pos), u

    def hiding_challenge(
        self, commitment: Array, hiding_comm: Array, point: Array, value: Array
    ) -> tuple[ArkIpaChallenger, Array]:
        """The zk/hiding opening's one pre-fold challenge (`ZkIpaChallenger`),
        byte-exact to `ipa_pc.succinct_check_challenges_zk`'s hiding-challenge
        derivation: a fresh `"IPA-PC-2020"` sponge absorbs `commitment`, then
        `hiding_comm`, then ``to_bytes![point, value]``, and squeezes one
        truncated-128 challenge. It is `seed` with the extra `hiding_comm` point
        absorbed between the commitment and the point/value bytes — the arkworks
        hiding fold's `commitment + hc·hiding_comm − s·rand` challenge. Squeezed once
        (it does not enter the per-round challenge list); the returned challenger's
        `state` is unchanged (the fold's subsequent `seed` starts fresh)."""
        cv = self.cv
        sp = self._sponge()
        sp = absorbable.absorb_points_jax(cv, sp, jnp.asarray(commitment).reshape(1))
        sp = absorbable.absorb_points_jax(cv, sp, jnp.asarray(hiding_comm).reshape(1))
        sp = sp.absorb(
            jnp.asarray(
                absorbable.u8_batch_field_array(cv, _fr32(cv, point) + _fr32(cv, value))
            )
        )
        hc = self._squeeze(sp)
        return ArkIpaChallenger(self.state, hc, cv, self.params, self.mode, self.pos), hc


def ark_challenger(cv: Curve, params: Any) -> ArkIpaChallenger:
    """Build the arkworks-faithful IPA-PC challenger over `cv` and the Poseidon
    `params`. Forks the `"IPA-PC-2020"` domain once; the carried `prev` is a
    placeholder ``fr(0)`` until `seed` binds the statement."""
    base = absorbable.fork(cv, sponge.new_sponge(params), IPA_PC_DOMAIN)
    zero = jnp.asarray(np.array([0], dtype=cv.fr))[0]
    return ArkIpaChallenger(base._state, zero, cv, params, base._mode, base._pos)
