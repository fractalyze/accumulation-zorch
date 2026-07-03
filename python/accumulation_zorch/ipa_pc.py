"""IPA-PC `succinct_check` challenge derivation + the `h(X)` check polynomial
(port of ark-poly-commit `ipa_pc::succinct_check` and `SuccinctCheckPolynomial`).

Slice 1: the no-zk succinct check's Fiat-Shamir round challenges and the dense
`h(X)` coefficient expansion — the field/sponge half of the IPA accumulation
verifier, MSM-free. Each round challenge depends only on the absorbed
`(L_i, R_i)` and the previous challenge, NOT on the running folded commitment, so
the whole challenge vector is derivable without the size-`2·log d` L/R fold. (The
IPA fold itself now lives in the vendored `zorch.pcs.ipa`, driven from
`ipa_open.py`; this module keeps only the verifier-side succinct-check + the
`h(X)` expansion. The decider's size-`d` MSM is `ipa_pc_as.decide_final_key`.)

Faithful to the arkworks pinned source (`ipa_pc/mod.rs::succinct_check`):

* Every challenge is squeezed from a **fresh** ``DomainSeparatedSponge`` forked
  with the ``"IPA-PC-2020"`` domain (NOT one running sponge) — there is no state
  carried between rounds except the previous challenge value.
* Seed challenge ξ₀ (consumed by the fold seed, NOT pushed into the check
  polynomial): absorb the combined commitment, then ``to_bytes![point, value]``.
* Round i: absorb the previous challenge (low 16 bytes, ``(CHALLENGE_SIZE+7)/8``),
  then ``L_i``, then ``R_i``.
* ``h(X) = ∏_{i=1..log d} (1 + ξ_i · X^{2^(log d − i)})`` (descending exponents, no
  inverses), expanded densely by ``compute_coeffs``.

For the ``ipa_pc_as`` single-input case the opening challenges are the constant
``1``, so the combined commitment / value collapse to the input's own
``commitment`` / ``value`` — what this port takes directly.
"""

from typing import NamedTuple

import numpy as np

from . import absorbable, curve, sponge
from .curve import Curve
from .field import fe_value, fe_values

# ark `ipa_pc` domain (`IpaPCDomain`): every fresh succinct-check sponge is a
# `DomainSeparatedSponge` forked with this label before anything is absorbed.
IPA_PC_DOMAIN = b"IPA-PC-2020"

# Each challenge is squeezed as `Truncated(CHALLENGE_SIZE=128)` into the scalar
# field; both Pasta scalar fields are 254-cap > 128, so this is the
# curve-invariant 128.
_CHALLENGE_BITS = min(sponge.CHALLENGE_SIZE, sponge.FR_CAPACITY)

# `to_bytes![round_challenge]` resized to `(CHALLENGE_SIZE + 7) / 8` bytes before
# the per-round absorb — the previous (truncated-128) challenge's low 16 bytes.
_CHALLENGE_BYTES = (sponge.CHALLENGE_SIZE + 7) // 8


def _new(cv: Curve, params):  # type: ignore[no-untyped-def]
    """A fresh succinct-check sponge: the classic Poseidon duplex forked with the
    IPA-PC domain (`S::new()` for the `DomainSeparatedSponge<_, _, IpaPCDomain>`)."""
    return absorbable.fork(cv, sponge.new_sponge(params), IPA_PC_DOMAIN)


def _squeeze_challenge(cv: Curve, sp) -> int:  # type: ignore[no-untyped-def]
    """Squeeze one truncated-128 challenge as a canonical `fr` int."""
    _, ch = sponge.squeeze_challenges(sp, 1, _CHALLENGE_BITS)
    return ch[0]


def _fr32(value: int) -> bytes:
    """`to_bytes![Fr]` — the 32-byte canonical LE serialization of a scalar."""
    return int(value).to_bytes(32, "little")


def _round_challenges_from_seed(
    cv: Curve, params, seed_commitment: np.ndarray, point: int, value: int,
    l_vec: list[np.ndarray], r_vec: list[np.ndarray],
) -> list[int]:  # type: ignore[no-untyped-def]
    """The round challenges ξ₁..ξ_log_d from a seed commitment: the seed sponge
    (absorb `seed_commitment` then `to_bytes![point, value]`) gives ξ₀, then each
    round's fresh sponge absorbs the previous challenge (low 16 bytes) + `L_i` +
    `R_i`. The seed commitment is the bare input commitment on the no-zk path and
    the hiding-folded commitment on the zk path — the only difference between the
    two succinct checks."""
    sp = _new(cv, params)
    sp = absorbable.absorb_point(cv, sp, seed_commitment)
    sp = absorbable.absorb_bytes(cv, sp, _fr32(point) + _fr32(value))
    round_challenge = _squeeze_challenge(cv, sp)

    challenges: list[int] = []
    for l, r in zip(l_vec, r_vec):
        sp = _new(cv, params)
        sp = absorbable.absorb_bytes(cv, sp, int(round_challenge).to_bytes(_CHALLENGE_BYTES, "little"))
        sp = absorbable.absorb_point(cv, sp, l)
        sp = absorbable.absorb_point(cv, sp, r)
        round_challenge = _squeeze_challenge(cv, sp)
        challenges.append(round_challenge)
    return challenges


def succinct_check_challenges(
    cv: Curve, params, commitment: np.ndarray, point: int, value: int,
    l_vec: list[np.ndarray], r_vec: list[np.ndarray],
) -> list[int]:  # type: ignore[no-untyped-def]
    """The `SuccinctCheckPolynomial` round challenges (ξ₁..ξ_log_d) of
    `ipa_pc::succinct_check` (no-zk), as canonical `fr` ints.

    `commitment` is the (combined) IPA commitment point; `point` / `value` are the
    opening's scalar point and claimed evaluation; `l_vec` / `r_vec` are the
    proof's per-round fold commitments (host affine point arrays). No-zk ⇒ the
    round-challenge seed is the bare combined commitment (no hiding fold)."""
    return _round_challenges_from_seed(cv, params, commitment, point, value, l_vec, r_vec)


def succinct_check_challenges_zk(
    cv: Curve, params, commitment: np.ndarray, point: int, value: int,
    l_vec: list[np.ndarray], r_vec: list[np.ndarray],
    s: np.ndarray, hiding_comm: np.ndarray, rand: int,
) -> list[int]:  # type: ignore[no-untyped-def]
    """The zk/hiding `ipa_pc::succinct_check` round challenges. Before deriving the
    round challenges, a fresh `"IPA-PC-2020"` sponge absorbs `commitment`,
    `hiding_comm`, then `to_bytes![point, value]` and squeezes one `Truncated(128)`
    `hiding_challenge`; the commitment is folded to
    `commitment + hiding_comm·hiding_challenge − s·rand`, and the round challenges
    are seeded from THAT. `s` is the succinct verifier key's hiding generator;
    `hiding_comm` / `rand` are the zk proof's `hiding_comm` / `rand`."""
    sp = _new(cv, params)
    sp = absorbable.absorb_point(cv, sp, commitment)
    sp = absorbable.absorb_point(cv, sp, hiding_comm)
    sp = absorbable.absorb_bytes(cv, sp, _fr32(point) + _fr32(value))
    hiding_challenge = _squeeze_challenge(cv, sp)

    # combined_commitment + hiding_comm·hiding_challenge − s·rand, as one group
    # reduction (the `−s·rand` term rides as `s·(r − rand)`, canonical in `fr`).
    neg_rand = (-int(rand)) % cv.fr_modulus
    seed = curve.pedersen_commit(
        cv, [commitment, hiding_comm, s], [1, int(hiding_challenge), neg_rand])
    return _round_challenges_from_seed(cv, params, seed, point, value, l_vec, r_vec)


def compute_coeffs(cv: Curve, challenges: list[int]) -> list[int]:
    """`SuccinctCheckPolynomial::compute_coeffs` — the dense `2^log_d` coefficients
    of `h(X) = ∏_{i=1..log_d} (1 + ξ_i · X^{2^(log_d − i)})`, as canonical `fr`
    ints. `coeffs[k]` is the product of the ξ_i whose power-of-two block covers
    index `k` (descending: ξ₁ multiplies the `X^{2^(log_d−1)}` block)."""
    log_d = len(challenges)
    n = 1 << log_d
    coeffs = np.ones(n, dtype=cv.fr)
    for i, ch in enumerate(challenges):
        elem_degree = 1 << (log_d - (i + 1))
        c = np.array([int(ch)], dtype=cv.fr)[0]
        for start in range(elem_degree, n, elem_degree * 2):
            coeffs[start:start + elem_degree] = coeffs[start:start + elem_degree] * c
    return fe_values(coeffs)


def evaluate(cv: Curve, challenges: list[int], point: int) -> int:
    """`SuccinctCheckPolynomial::evaluate(point)` — `∏ (1 + ξ_i · point^{2^(log_d −
    i)})` in `fr`, as a canonical int. The succinct form of `compute_coeffs`
    (no `2^log_d`-size expansion); each power `point^{2^k}` is an `fr` exponentiation."""
    log_d = len(challenges)
    one = np.ones(1, dtype=cv.fr)
    product = np.ones(1, dtype=cv.fr)
    p = int(point)
    for i, ch in enumerate(challenges):
        elem_degree = 1 << (log_d - (i + 1))
        elem = np.array([pow(p, elem_degree, cv.fr_modulus)], dtype=cv.fr)
        ch_fr = np.array([int(ch)], dtype=cv.fr)
        product = product * (one + elem * ch_fr)
    return fe_value(product)


# --- IPA opening proof shape ---
#
# The prover's IPA fold (arkworks' `open_individual_opening_challenges`, no-zk
# and zk) used to be ported here, reproducing arkworks' deferred even/odd
# (BCLMS) generator fold by hand. That port is gone: the fold is now provided
# by zorch's `pcs.ipa` prover, driven through the `ipa_open.py` adapter
# (`ipa_open.open_no_zk` / `ipa_open.open_zk`), which `ipa_pc_as` calls
# instead. `IpaProof` is the shared proof shape both that adapter and
# `ipa_pc_as` use.


class IpaProof(NamedTuple):
    """The IPA opening proof: the per-round fold commitments, the fully-folded
    generator and coefficient, and (zk only) the hiding commitment + combined
    blinder. No-zk leaves `hiding_comm`/`rand` as `None`."""
    l_vec: list[np.ndarray]
    r_vec: list[np.ndarray]
    final_comm_key: np.ndarray
    c: int
    hiding_comm: np.ndarray | None = None
    rand: int | None = None
