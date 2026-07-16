"""accumulation-zorch's IPA-PC succinct check (verifier side): the arkworks-faithful
Fiat-Shamir challenger `ArkIpaChallenger`, the host `succinct_check_challenges`
drivers that pull a `SuccinctCheckPolynomial`'s round challenges out of it, and the
dense `h(X)` expansion (`compute_coeffs` / `evaluate_fr`, delegated to zorch's
`pcs/ipa/math`). Together these are the port of ark-poly-commit
`ipa_pc::succinct_check` + `SuccinctCheckPolynomial` — the MSM-free field/sponge half
of the IPA accumulation verifier the AS combinator (`ipa_pc_as`) consumes. The IPA
*fold* itself lives in zorch's `zorch.pcs.ipa` (a Bazel dep), driven from `ipa_open`.

`ArkIpaChallenger` is the single Fiat-Shamir implementation: zorch's IPA fold
(`zorch.pcs.ipa.challenger.IpaChallenger`) derives its round challenges through a
challenger it carries as a `lax.scan` carry, and this is the accumulation-consumer's
byte-exact challenger, built entirely on the jit-able `sponge` / `absorbable`
primitives, so the same challenger drives the CPU port, the fused GPU core, and the
host `succinct_check_challenges` below — all yielding byte-identical challenges.

The reproduced Fiat-Shamir (faithful to ark-poly-commit `ipa_pc::succinct_check`):

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
  ``absorbable.absorb_points_frx`` — the ``[x, y, infinity]`` SW-affine packing —
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
* the **seed** scalars ``point`` / ``value`` are absorbed in-trace too
  (``_seed_pv_fq`` → ``absorbable.u8_batch_field_array_frx``): each is bitcast to
  its 32-byte canonical LE repr (as the previous-challenge absorb does), then the
  ``u8`` batch packing (``to_bytes![point, value]``) is reproduced with frx ops so
  ``seed`` traces cleanly (the fused open core seeds from the on-device combined
  ``value``). Byte-identical to the oracle's ``to_bytes![point, value]``, so the
  eager CPU port is unchanged.

The squeeze is the in-trace truncated-128 nonnative squeeze
(``sponge.challenges_from_fq`` via ``sponge.squeeze_challenges_frx``): one ``fq``
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

import frx.numpy as fnp
import numpy as np
from frx import Array, lax
from frx.tree_util import register_dataclass
from zorch.pcs.ipa.math import challenge_vector, eval_challenge_poly

from . import absorbable, curve, sponge
from .curve import Curve

# ark `ipa_pc` domain (`IpaPCDomain`): every fresh succinct-check sponge is a
# `DomainSeparatedSponge` forked with this label before anything is absorbed.
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


def _as_fr(cv: Curve, x: Array | int) -> Array:
    """A host int or `fr`-ish array as a 0-d ``cv.fr`` frx array (traced-safe). The
    `squeeze` enforces the 0-d contract for a `(1,)`-shaped array input (bitcasting a
    rank-1 scalar would give a `(1, 32)` byte array and skew the u8-batch packing)."""
    if isinstance(x, (int, np.integer)):
        return fnp.asarray(np.array([int(x)], dtype=cv.fr))[0]
    return fnp.squeeze(fnp.asarray(x, dtype=cv.fr))


def _seed_pv_fq(cv: Curve, point: Array | int, value: Array | int) -> Array:
    """In-trace ``to_bytes![point, value]`` u8-batch packing (each a 32-byte canonical
    LE ``fr``), for :meth:`ArkIpaChallenger.seed` / :meth:`hiding_challenge`. The
    jit-able twin of the former host ``_fr32(point) + _fr32(value)`` →
    ``u8_batch_field_array``: bitcast each scalar to its 32 LE bytes (canonical, as in
    ``_absorb_prev``), concatenate, and pack in-trace so the seed rides the fused open
    core's ``@jit`` boundary. Byte-identical eagerly, so the CPU port is unchanged."""
    pb = lax.bitcast_convert_type(_as_fr(cv, point), fnp.uint8)  # (32,) LE
    vb = lax.bitcast_convert_type(_as_fr(cv, value), fnp.uint8)  # (32,) LE
    return absorbable.u8_batch_field_array_frx(cv, fnp.concatenate([pb, vb]))


@partial(
    register_dataclass,
    data_fields=["state", "prev"],
    meta_fields=["cv", "params", "mode", "pos"],
)
@dataclass(frozen=True)
class ArkIpaChallenger:
    """The arkworks-faithful `IpaChallenger` — and, via :meth:`hiding_challenge`,
    the `ZkIpaChallenger` for the hiding open. A FRX pytree: `state` (the fresh
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
        squeeze) — the `Truncated(CHALLENGE_SIZE=128)` succinct-check squeeze."""
        _, ch = sponge.squeeze_challenges_frx(
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
        prev_bytes = lax.bitcast_convert_type(self.prev, fnp.uint8)  # (32,) LE bytes
        prev_fq = lax.bitcast_convert_type(prev_bytes, cv.fq)  # fq(prev)
        length = fnp.asarray(np.array([_CHALLENGE_BYTES], dtype=cv.fq))[0]
        shift = fnp.asarray(np.array([_U64_SHIFT], dtype=cv.fq))[0]
        fe = length + prev_fq * shift
        return sp.absorb(fe[fnp.newaxis])

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
        sp = absorbable.absorb_points_frx(cv, sp, fnp.asarray(commitment).reshape(1))
        sp = sp.absorb(_seed_pv_fq(cv, point, value))
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
        sp = absorbable.absorb_points_frx(cv, sp, fnp.asarray(l).reshape(1))
        sp = absorbable.absorb_points_frx(cv, sp, fnp.asarray(r).reshape(1))
        u = self._squeeze(sp)
        return ArkIpaChallenger(self.state, u, cv, self.params, self.mode, self.pos), u

    def hiding_challenge(
        self, commitment: Array, hiding_comm: Array, point: Array, value: Array
    ) -> tuple[ArkIpaChallenger, Array]:
        """The zk/hiding opening's one pre-fold challenge (`ZkIpaChallenger`),
        byte-exact to `succinct_check_challenges_zk`'s hiding-challenge
        derivation: a fresh `"IPA-PC-2020"` sponge absorbs `commitment`, then
        `hiding_comm`, then ``to_bytes![point, value]``, and squeezes one
        truncated-128 challenge. It is `seed` with the extra `hiding_comm` point
        absorbed between the commitment and the point/value bytes — the arkworks
        hiding fold's `commitment + hc·hiding_comm − s·rand` challenge. Squeezed once
        (it does not enter the per-round challenge list); the returned challenger's
        `state` is unchanged (the fold's subsequent `seed` starts fresh)."""
        cv = self.cv
        sp = self._sponge()
        sp = absorbable.absorb_points_frx(cv, sp, fnp.asarray(commitment).reshape(1))
        sp = absorbable.absorb_points_frx(cv, sp, fnp.asarray(hiding_comm).reshape(1))
        sp = sp.absorb(_seed_pv_fq(cv, point, value))
        hc = self._squeeze(sp)
        return ArkIpaChallenger(self.state, hc, cv, self.params, self.mode, self.pos), hc


def ark_challenger(cv: Curve, params: Any) -> ArkIpaChallenger:
    """Build the arkworks-faithful IPA-PC challenger over `cv` and the Poseidon
    `params`. Forks the `"IPA-PC-2020"` domain once; the carried `prev` is a
    placeholder ``fr(0)`` until `seed` binds the statement."""
    base = absorbable.fork(cv, sponge.new_sponge(params), IPA_PC_DOMAIN)
    zero = fnp.asarray(np.array([0], dtype=cv.fr))[0]
    return ArkIpaChallenger(base._state, zero, cv, params, base._mode, base._pos)


# --- host succinct-check drivers ---------------------------------------------
# The `SuccinctCheckPolynomial` round challenges of `ipa_pc::succinct_check`, pulled
# out of the single `ArkIpaChallenger` FS as canonical `fr` ints — the AS combinator
# (`ipa_pc_as`) consumes these host-side (absorbed as bytes into the AS sponge). Each
# round challenge depends only on the absorbed `(L_i, R_i)` and the previous
# challenge, NOT on the running folded commitment, so the whole vector is derivable
# without the L/R fold (that fold — the IPA *open* — is zorch's, driven by `ipa_open`).


def _challenge_int(cv: Curve, u: Array) -> int:
    """A 0-d `fr` challenge as its canonical `fr` int (the `check_poly` element form)."""
    return int.from_bytes(np.asarray(u, dtype=cv.fr).tobytes(), "little")


def succinct_check_challenges(
    cv: Curve, params, commitment: np.ndarray, point: int, value: int,
    l_vec: list[np.ndarray], r_vec: list[np.ndarray],
) -> list[int]:  # type: ignore[no-untyped-def]
    """The `SuccinctCheckPolynomial` round challenges (ξ₁..ξ_log_d) of
    `ipa_pc::succinct_check` (no-zk), as canonical `fr` ints, by driving
    :class:`ArkIpaChallenger`: `seed(commitment, point, value)` gives ξ₀ (consumed by
    the fold seed, not pushed into the check polynomial), then each round's
    `challenge(L_i, R_i)` gives ξ_i.

    `commitment` is the (combined) IPA commitment point; `point` / `value` are the
    opening's scalar point and claimed evaluation; `l_vec` / `r_vec` are the proof's
    per-round fold commitments (host affine point arrays). No-zk ⇒ the round-challenge
    seed is the bare combined commitment (no hiding fold)."""
    fs = ark_challenger(cv, params)
    fs, _xi0 = fs.seed(commitment, point, value)
    challenges: list[int] = []
    for l, r in zip(l_vec, r_vec):
        fs, u = fs.challenge(l, r)
        challenges.append(_challenge_int(cv, u))
    return challenges


def succinct_check_challenges_zk(
    cv: Curve, params, commitment: np.ndarray, point: int, value: int,
    l_vec: list[np.ndarray], r_vec: list[np.ndarray],
    s: np.ndarray, hiding_comm: np.ndarray, rand: int,
) -> list[int]:  # type: ignore[no-untyped-def]
    """The zk/hiding `ipa_pc::succinct_check` round challenges. Before the round
    challenges, `ArkIpaChallenger.hiding_challenge` absorbs `commitment`, `hiding_comm`,
    then `to_bytes![point, value]` and squeezes one `Truncated(128)` `hiding_challenge`;
    the commitment is folded to `commitment + hiding_comm·hiding_challenge − s·rand`,
    and the round challenges are seeded from THAT (`ArkIpaChallenger.seed`). `s` is the
    succinct verifier key's hiding generator; `hiding_comm` / `rand` are the zk proof's
    `hiding_comm` / `rand`."""
    fs = ark_challenger(cv, params)
    fs, hc = fs.hiding_challenge(commitment, hiding_comm, point, value)

    # combined_commitment + hiding_comm·hiding_challenge − s·rand, as one group
    # reduction (the `−s·rand` term rides as `s·(r − rand)`, canonical in `fr`). The
    # hiding challenge leaves `fs.state` at the fresh domain fork, so the same `fs`
    # seeds the rounds from the folded commitment.
    neg_rand = (-int(rand)) % cv.fr_modulus
    seed = curve.pedersen_commit(
        cv, [commitment, hiding_comm, s], [1, _challenge_int(cv, hc), neg_rand])

    fs, _xi0 = fs.seed(seed, point, value)
    challenges: list[int] = []
    for l, r in zip(l_vec, r_vec):
        fs, u = fs.challenge(l, r)
        challenges.append(_challenge_int(cv, u))
    return challenges


# --- the h(X) check polynomial (zorch's `pcs/ipa/math`) -----------------------


def compute_coeffs(cv: Curve, challenges: list[int]) -> np.ndarray:
    """`SuccinctCheckPolynomial::compute_coeffs` — the dense `2^log_d` coefficients of
    `h(X) = ∏_{i=1..log_d} (1 + ξ_i · X^{2^(log_d − i)})`, as a host `fr` array.
    `coeffs[k]` is the product of the ξ_i whose power-of-two block covers index `k`
    (descending: ξ₁ multiplies the `X^{2^(log_d−1)}` block).

    Delegates to zorch's `pcs/ipa/math.challenge_vector` — the identical dense
    check-polynomial coefficients (`s ← concat(s, ξ_i·s)` unrolled last-to-first, same
    descending index order), pinned against this same arkworks oracle in zorch.
    Materialized to a host `cv.fr` array at the boundary (fed straight to
    `pedersen_commit` / the combined check polynomial), never decoded to ints."""
    u = fnp.asarray(np.array(challenges, dtype=cv.fr))
    return np.asarray(challenge_vector(u), dtype=cv.fr)


def evaluate_fr(cv: Curve, challenges: list[int], point: int) -> Array:
    """`SuccinctCheckPolynomial::evaluate(point)` as the `fr` scalar itself (no int
    decode) — `∏ (1 + ξ_i · point^{2^(log_d − i)})` from zorch's
    `pcs/ipa/math.eval_challenge_poly` (the O(log_d), no-inverse read of
    `compute_coeffs`, `point^{2^m}` by repeated squaring). For callers that keep
    working in the field — e.g. the AS combined evaluation's `Σ lc·h` weighted sum — so
    the per-input `h_j` never round-trips through a canonical int."""
    u = fnp.asarray(np.array(challenges, dtype=cv.fr))
    x = fnp.asarray(cv.fr(point))
    return eval_challenge_poly(u, x)
