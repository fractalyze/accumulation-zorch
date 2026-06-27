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

from typing import NamedTuple

import numpy as np

from . import absorbable, curve, sponge
from .curve import Curve
from .field import fe_value, fe_values, fr_mul

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


# --- IPA open (the prover fold) — `open_individual_opening_challenges`, no-zk ---
#
# The prover produces the proof the verifier's succinct_check consumes. It carries
# `coeffs` (the combined polynomial), `z = [1, x, x², …, x^d]`, and the generators
# `comm_key`, folding them log d times. The per-round Fiat-Shamir is IDENTICAL to
# succinct_check (same fresh "IPA-PC-2020" sponge, same seed + per-round absorbs);
# the round challenges of the produced proof are exactly its succinct check's. The
# only intricacy is arkworks' even/odd deferred generator fold (the BCLMS trick):
# on an even round the `comm_key` fold is deferred (unless `n == 2`), and folded
# two levels deep on the next odd round — the L/R commitment MSMs absorb that
# deferred fold. This must be reproduced exactly to be byte-identical.


def _msm(cv: Curve, bases: list[np.ndarray], scalars: list[int]) -> np.ndarray:
    """`cm_commit` — the variable-base MSM `Σ bases_i · scalars_i` (no hiding), the
    CPU group-reduction oracle (`curve.pedersen_commit`)."""
    return curve.pedersen_commit(cv, bases, scalars)


def _inner_product(cv: Curve, a: list[int], b: list[int]) -> int:
    """`inner_product` — `Σ a_i · b_i` in `fr` (a jax `fr` dot, reduced mod r)."""
    return fe_value(np.dot(np.array(a, dtype=cv.fr), np.array(b, dtype=cv.fr)))


def _affine(cv: Curve, point: np.ndarray) -> np.ndarray:
    """Normalize a (possibly jacobian) group result back to an affine point array."""
    return np.asarray(point, dtype=cv.g1)


def _even_commitment_step(cv, comm_key, h_prime, coeffs_l, coeffs_r, z_l, z_r):  # type: ignore[no-untyped-def]
    """`even_commitment_step`: split the FULL `comm_key`; `L = <coeffs_r, key_l> +
    h'·<coeffs_r, z_l>`, `R = <coeffs_l, key_r> + h'·<coeffs_l, z_r>`."""
    n = len(comm_key)
    key_l, key_r = comm_key[: n // 2], comm_key[n // 2:]
    l = _msm(cv, key_l + [h_prime], list(coeffs_r) + [_inner_product(cv, coeffs_r, z_l)])
    r = _msm(cv, key_r + [h_prime], list(coeffs_l) + [_inner_product(cv, coeffs_l, z_r)])
    return l, r


def _odd_commitment_step(cv, comm_key, h_prime, rc, coeffs_l, coeffs_r, z_l, z_r):  # type: ignore[no-untyped-def]
    """`odd_commitment_step`: the deferred-fold case — a 4-way `comm_key` split,
    with `rc·coeffs` fused into the L/R MSMs alongside the bare `coeffs` and the
    `h'` inner-product cross term."""
    n = len(comm_key)
    key_l, key_r = comm_key[: n // 2], comm_key[n // 2:]
    key_l_1, key_l_2 = key_l[: n // 4], key_l[n // 4:]
    key_r_1, key_r_2 = key_r[: n // 4], key_r[n // 4:]
    rc_coeffs_r = [fr_mul(cv, rc, c) for c in coeffs_r]
    rc_coeffs_l = [fr_mul(cv, rc, c) for c in coeffs_l]
    l = _msm(cv, key_l_1 + key_r_1 + [h_prime],
             list(coeffs_r) + rc_coeffs_r + [_inner_product(cv, coeffs_r, z_l)])
    r = _msm(cv, key_l_2 + key_r_2 + [h_prime],
             list(coeffs_l) + rc_coeffs_l + [_inner_product(cv, coeffs_l, z_r)])
    return l, r


def _even_folding_step(cv, comm_key, rc):  # type: ignore[no-untyped-def]
    """`even_folding_step`: fold `key ← key_lo + rc·key_hi` only at `n == 2`;
    otherwise carry the generators unchanged (the fold is deferred to the next
    odd round)."""
    n = len(comm_key)
    if n != 2:
        return list(comm_key)
    key_l, key_r = comm_key[: n // 2], comm_key[n // 2:]
    return [_affine(cv, key_l[k] + key_r[k] * cv.fr(int(rc))) for k in range(len(key_l))]


def _odd_folding_step(cv, comm_key, prev_rc, rc):  # type: ignore[no-untyped-def]
    """`odd_folding_step`: the deferred two-level fold. At `n == 2` fold with
    `prev_rc`; otherwise `key_l_1 + rc·key_l_2 + prev_rc·key_r_1 + (prev_rc·rc)·key_r_2`."""
    n = len(comm_key)
    key_l, key_r = comm_key[: n // 2], comm_key[n // 2:]
    if n == 2:
        return [_affine(cv, key_l[k] + key_r[k] * cv.fr(int(prev_rc))) for k in range(len(key_l))]
    key_l_1, key_l_2 = key_l[: n // 4], key_l[n // 4:]
    key_r_1, key_r_2 = key_r[: n // 4], key_r[n // 4:]
    prc_rc = fr_mul(cv, prev_rc, rc)
    return [
        _affine(cv, key_l_1[k] + key_l_2[k] * cv.fr(int(rc))
                + key_r_1[k] * cv.fr(int(prev_rc)) + key_r_2[k] * cv.fr(int(prc_rc)))
        for k in range(len(key_l_1))
    ]


class IpaProof(NamedTuple):
    """The IPA opening proof (no-zk): the per-round fold commitments, the
    fully-folded generator and coefficient. (`hiding_comm`/`rand` are `None`.)"""
    l_vec: list[np.ndarray]
    r_vec: list[np.ndarray]
    final_comm_key: np.ndarray
    c: int


def open_no_zk(
    cv: Curve, params, svk_h: np.ndarray, combined_commitment: np.ndarray,
    point: int, coeffs: list[int], generators: list[np.ndarray],
) -> "IpaProof":  # type: ignore[no-untyped-def]
    """`IpaPC::open_individual_opening_challenges` (no-zk, single combined poly):
    the IPA fold producing `(l_vec, r_vec, final_comm_key, c)`.

    `coeffs` is the combined check polynomial (length `d+1 = 2^log_d`);
    `generators` the committer-key `comm_key`; `combined_commitment` seeds the
    Fiat-Shamir (same `to_bytes![point, value]` as succinct_check, with
    `value = Σ coeffs_i·point^i`)."""
    d = len(generators) - 1
    z = [pow(int(point), k, cv.fr_modulus) for k in range(d + 1)]
    combined_v = _inner_product(cv, coeffs, z)

    # Seed challenge ξ₀ + h' = svk.h·ξ₀ (the inner-product cross-term base).
    sp = _new(cv, params)
    sp = absorbable.absorb_point(cv, sp, combined_commitment)
    sp = absorbable.absorb_bytes(cv, sp, _fr32(point) + _fr32(combined_v))
    round_challenge = _squeeze_challenge(cv, sp)
    h_prime = _affine(cv, svk_h * cv.fr(int(round_challenge)))

    comm_key = list(generators)
    coeffs = list(coeffs)
    l_vec: list[np.ndarray] = []
    r_vec: list[np.ndarray] = []
    n = d + 1
    i = 0
    while n > 1:
        half = n // 2
        coeffs_l, coeffs_r = coeffs[:half], coeffs[half:]
        z_l, z_r = z[:half], z[half:]
        if i % 2 == 0:
            l, r = _even_commitment_step(cv, comm_key, h_prime, coeffs_l, coeffs_r, z_l, z_r)
        else:
            l, r = _odd_commitment_step(cv, comm_key, h_prime, round_challenge,
                                        coeffs_l, coeffs_r, z_l, z_r)
        l_vec.append(l)
        r_vec.append(r)

        sp = _new(cv, params)
        sp = absorbable.absorb_bytes(cv, sp, int(round_challenge).to_bytes(_CHALLENGE_BYTES, "little"))
        sp = absorbable.absorb_point(cv, sp, l)
        sp = absorbable.absorb_point(cv, sp, r)
        prev_round_challenge = round_challenge
        round_challenge = _squeeze_challenge(cv, sp)
        rc_inv = pow(int(round_challenge), -1, cv.fr_modulus)

        coeffs = [fe_value(np.array([coeffs_l[k]], dtype=cv.fr)
                           + np.array([fr_mul(cv, rc_inv, coeffs_r[k])], dtype=cv.fr))
                  for k in range(half)]
        z = [fe_value(np.array([z_l[k]], dtype=cv.fr)
                      + np.array([fr_mul(cv, round_challenge, z_r[k])], dtype=cv.fr))
             for k in range(half)]

        if i % 2 == 0:
            comm_key = _even_folding_step(cv, comm_key, round_challenge)
        else:
            comm_key = _odd_folding_step(cv, comm_key, prev_round_challenge, round_challenge)
        i += 1
        n //= 2

    return IpaProof(l_vec, r_vec, comm_key[0], coeffs[0])
