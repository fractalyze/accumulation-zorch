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
    """The IPA opening proof: the per-round fold commitments, the fully-folded
    generator and coefficient, and (zk only) the hiding commitment + combined
    blinder. No-zk leaves `hiding_comm`/`rand` as `None`."""
    l_vec: list[np.ndarray]
    r_vec: list[np.ndarray]
    final_comm_key: np.ndarray
    c: int
    hiding_comm: np.ndarray | None = None
    rand: int | None = None


def _open_fold(
    cv: Curve, params, svk_h: np.ndarray, combined_commitment: np.ndarray,
    point: int, coeffs: list[int], generators: list[np.ndarray],
) -> tuple:  # type: ignore[no-untyped-def]
    """The IPA fold producing `(l_vec, r_vec, final_comm_key, c)` from a combined
    commitment + combined polynomial coefficients — shared by the no-zk and zk
    opens (the zk path differs only in the hiding prelude that produces these
    inputs). `combined_v = Σ coeffs_i·point^i` seeds the Fiat-Shamir (the zk hiding
    polynomial evaluates to 0 at the point, so this equals the pre-hiding value)."""
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

    return l_vec, r_vec, comm_key[0], coeffs[0]


def open_no_zk(
    cv: Curve, params, svk_h: np.ndarray, combined_commitment: np.ndarray,
    point: int, coeffs: list[int], generators: list[np.ndarray],
) -> "IpaProof":  # type: ignore[no-untyped-def]
    """`IpaPC::open_individual_opening_challenges` (no-zk, single combined poly):
    the IPA fold producing `(l_vec, r_vec, final_comm_key, c)`.

    `coeffs` is the combined check polynomial (length `d+1 = 2^log_d`);
    `generators` the committer-key `comm_key`; `combined_commitment` seeds the
    Fiat-Shamir."""
    l_vec, r_vec, final_comm_key, c = _open_fold(
        cv, params, svk_h, combined_commitment, point, coeffs, generators)
    return IpaProof(l_vec, r_vec, final_comm_key, c)


def open_zk(
    cv: Curve, params, svk_h: np.ndarray, s: np.ndarray, generators: list[np.ndarray],
    combined_commitment: np.ndarray, point: int, coeffs: list[int],
    hiding_poly_raw: list[int], hiding_rand: int, commitment_randomness: int,
) -> "IpaProof":  # type: ignore[no-untyped-def]
    """`IpaPC::open_individual_opening_challenges` (zk, single combined poly): the
    hiding prelude then the shared IPA fold. Before folding, the combined
    polynomial is masked by a random degree-`d` `hiding_polynomial` (shifted to
    evaluate to 0 at `point`, so the claimed value is unchanged):

    * `hiding_commitment = Σ generators_i·hiding_poly_i + s·hiding_rand`.
    * `hiding_challenge` is squeezed from a fresh IPA-PC sponge absorbing
      `combined_commitment, hiding_commitment, to_bytes![point, combined_v]`.
    * `combined_poly += hiding_challenge·hiding_poly`,
      `combined_rand = commitment_randomness + hiding_challenge·hiding_rand`,
      `combined_commitment += hiding_commitment·hiding_challenge − s·combined_rand`.

    `hiding_poly_raw` is arkworks' `P::rand(d, rng)` (pre-shift); `hiding_rand` /
    `commitment_randomness` its blinders. Returns the proof with
    `hiding_comm`/`rand` set."""
    d = len(generators) - 1
    z = [pow(int(point), k, cv.fr_modulus) for k in range(d + 1)]
    combined_v = _inner_product(cv, coeffs, z)

    # Shift the raw hiding polynomial to evaluate to 0 at `point` (subtract the
    # constant `raw(point)` — only the degree-0 coefficient changes).
    raw = [int(c) for c in hiding_poly_raw] + [0] * (d + 1 - len(hiding_poly_raw))
    raw_eval = _inner_product(cv, raw, z)
    hiding_poly = list(raw)
    hiding_poly[0] = fe_value(np.array([raw[0]], dtype=cv.fr) - np.array([raw_eval], dtype=cv.fr))

    hiding_commitment = curve.pedersen_commit(
        cv, generators, hiding_poly, hiding=s, randomizer=int(hiding_rand))

    sp = _new(cv, params)
    sp = absorbable.absorb_point(cv, sp, combined_commitment)
    sp = absorbable.absorb_point(cv, sp, hiding_commitment)
    sp = absorbable.absorb_bytes(cv, sp, _fr32(point) + _fr32(combined_v))
    hc = _squeeze_challenge(cv, sp)

    hc_fr = np.array([int(hc)], dtype=cv.fr)
    mod_coeffs = [
        fe_value(np.array([coeffs[k]], dtype=cv.fr) + hc_fr * np.array([hiding_poly[k]], dtype=cv.fr))
        for k in range(d + 1)
    ]
    combined_rand = fe_value(
        np.array([int(commitment_randomness)], dtype=cv.fr) + hc_fr * np.array([int(hiding_rand)], dtype=cv.fr))
    neg_rand = (-int(combined_rand)) % cv.fr_modulus
    mod_commitment = curve.pedersen_commit(
        cv, [combined_commitment, hiding_commitment, s], [1, int(hc), neg_rand])

    l_vec, r_vec, final_comm_key, c = _open_fold(
        cv, params, svk_h, mod_commitment, point, mod_coeffs, generators)
    return IpaProof(l_vec, r_vec, final_comm_key, c, hiding_comm=hiding_commitment, rand=int(combined_rand))
