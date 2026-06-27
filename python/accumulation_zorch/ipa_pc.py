"""IPA-PC `succinct_check` challenge derivation + the `h(X)` check polynomial
(port of ark-poly-commit `ipa_pc::succinct_check` and `SuccinctCheckPolynomial`).

Slice 1: the no-zk succinct check's Fiat-Shamir round challenges and the dense
`h(X)` coefficient expansion — the field/sponge half of the IPA accumulation
verifier, MSM-free. Each round challenge depends only on the absorbed
`(L_i, R_i)` and the previous challenge, NOT on the running folded commitment, so
the whole challenge vector is derivable without the size-`2·log d` L/R fold. The
fold + final equality check (point ops) and the decider's size-`d` MSM are later
slices.

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

import numpy as np

from . import absorbable, sponge
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


def succinct_check_challenges(
    cv: Curve, params, commitment: np.ndarray, point: int, value: int,
    l_vec: list[np.ndarray], r_vec: list[np.ndarray],
) -> list[int]:  # type: ignore[no-untyped-def]
    """The `SuccinctCheckPolynomial` round challenges (ξ₁..ξ_log_d) of
    `ipa_pc::succinct_check`, as canonical `fr` ints.

    `commitment` is the (combined) IPA commitment point; `point` / `value` are the
    opening's scalar point and claimed evaluation; `l_vec` / `r_vec` are the
    proof's per-round fold commitments (host affine point arrays)."""
    # Seed challenge ξ₀ — fresh sponge, absorb the combined commitment then
    # `to_bytes![point, value]`. Not pushed into the check polynomial.
    sp = _new(cv, params)
    sp = absorbable.absorb_point(cv, sp, commitment)
    sp = absorbable.absorb_bytes(cv, sp, _fr32(point) + _fr32(value))
    round_challenge = _squeeze_challenge(cv, sp)

    # Round i — fresh sponge, absorb the previous challenge (low 16 bytes) then
    # L_i, R_i, and squeeze the next challenge.
    challenges: list[int] = []
    for l, r in zip(l_vec, r_vec):
        sp = _new(cv, params)
        sp = absorbable.absorb_bytes(cv, sp, int(round_challenge).to_bytes(_CHALLENGE_BYTES, "little"))
        sp = absorbable.absorb_point(cv, sp, l)
        sp = absorbable.absorb_point(cv, sp, r)
        round_challenge = _squeeze_challenge(cv, sp)
        challenges.append(round_challenge)
    return challenges


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
