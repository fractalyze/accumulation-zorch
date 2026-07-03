"""IPA-PC accumulation prover (port of `ipa_pc_as/mod.rs`), no-zk path.

Slice 2a: the AS-level Fiat-Shamir + linear combination that turns the per-input
succinct checks into the new accumulator's *instance* fields — the combined
commitment, the new opening point, and the combined evaluation. The accumulator's
IPA *proof* (the `IpaPC::open` fold over the combined check polynomial) is Slice 2b.

Faithful to the arkworks no-zk path (`combine_succinct_check_polynomials_and_commitments`
+ `compute_new_challenge` + `compute_new_accumulator`):

* The AS uses a **`"AS-FOR-IPA-PC-2020"`** domain-separated sponge — distinct from
  the IPA succinct check's `"IPA-PC-2020"` sponge. `combine` and
  `compute_new_challenge` each start from a **fresh** AS sponge (the prover clones
  the same `S::new()` for both); they do NOT share state.
* No-zk ⇒ no `random_linear_polynomial`: `combine` absorbs nothing before the
  per-input loop, and `compute_new_challenge` absorbs the `None` option flag
  (a single `fq` 0) where the random-poly coefficients would go.
* `combine`: per `(check_poly, final_comm_key)` absorb the check polynomial (the
  round challenges concatenated as 32-byte LE, one byte-vec) then the
  `final_comm_key` (group element); squeeze one `Truncated(128)` linear-combination
  challenge per input. `combined_commitment = Σ final_comm_key_j · lc_challenge_j`.
* `compute_new_challenge`: fresh sponge, absorb `combined_commitment`, the `None`
  flag, then per addend the `lc_challenge` (low 16 bytes) and the check polynomial;
  squeeze one `Truncated(184)` element = the new accumulator's opening point.
* The new accumulator's `evaluation = Σ lc_challenge_j · h_j(point)`, the combined
  check polynomial evaluated at the new point (linear, so it is the per-input
  `h_j(point)` weighted by the lc challenges).

The per-input check polynomials and `final_comm_key`s come from
:mod:`ipa_pc` succinct checks (Slice 1).
"""

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from . import absorbable, curve, ipa_open, ipa_pc, sponge
from .curve import Curve
from .field import fe_value, fe_values

# ark `ipa_pc_as` AS-level domain (`ASForIpaPCDomain`).
AS_DOMAIN = b"AS-FOR-IPA-PC-2020"

# ark `ipa_pc_as` challenge sizes (bits): the per-input linear-combination
# challenges and the new opening point.
_LC_CHALLENGE_BITS = 128
_CHALLENGE_POINT_BITS = 184

# `to_bytes![linear_combination_challenge]` resized to `(128 + 7) / 8` before the
# per-addend absorb in `compute_new_challenge`.
_LC_CHALLENGE_BYTES = (_LC_CHALLENGE_BITS + 7) // 8


class SuccinctCheck(NamedTuple):
    """One input's succinct-check output: the `SuccinctCheckPolynomial` round
    challenges and the IPA proof's `final_comm_key` (the AS combines these)."""
    check_poly: list[int]
    final_comm_key: np.ndarray


def _new(cv: Curve, params):  # type: ignore[no-untyped-def]
    """A fresh AS sponge: the classic Poseidon duplex forked with the AS domain."""
    return absorbable.fork(cv, sponge.new_sponge(params), AS_DOMAIN)


def _absorb_check_poly(cv: Curve, sp, check_poly: list[int]):  # type: ignore[no-untyped-def]
    """`absorb_succinct_check_polynomial_into_sponge`: the round challenges
    concatenated as 32-byte LE serializations, absorbed as one byte-vec."""
    bytes_input = b"".join(int(ch).to_bytes(32, "little") for ch in check_poly)
    return absorbable.absorb_bytes(cv, sp, bytes_input)


def combine(
    cv: Curve, params, succinct_checks: list[SuccinctCheck],
) -> tuple[list[int], np.ndarray, list[tuple[int, list[int]]]]:  # type: ignore[no-untyped-def]
    """`combine_succinct_check_polynomials_and_commitments` (no-zk): the
    linear-combination challenges, the combined commitment
    `Σ final_comm_key_j · lc_challenge_j`, and the `(lc_challenge, check_poly)`
    addends for the combined check polynomial."""
    sp = _new(cv, params)
    for sc in succinct_checks:
        sp = _absorb_check_poly(cv, sp, sc.check_poly)
        sp = absorbable.absorb_point(cv, sp, sc.final_comm_key)
    _, lc_challenges = sponge.squeeze_challenges(sp, len(succinct_checks), _LC_CHALLENGE_BITS)

    combined_commitment = curve.pedersen_commit(
        cv, [sc.final_comm_key for sc in succinct_checks], lc_challenges)
    addends = [(lc, sc.check_poly) for lc, sc in zip(lc_challenges, succinct_checks)]
    return lc_challenges, combined_commitment, addends


def compute_new_challenge(
    cv: Curve, params, combined_commitment: np.ndarray,
    addends: list[tuple[int, list[int]]],
) -> int:  # type: ignore[no-untyped-def]
    """`compute_new_challenge` (no-zk): a fresh AS sponge absorbing the combined
    commitment, the `None` random-poly flag, then per addend the lc challenge
    (16 bytes) and the check polynomial; squeeze the new opening point
    (`Truncated(184)`)."""
    sp = _new(cv, params)
    sp = absorbable.absorb_point(cv, sp, combined_commitment)
    sp = absorbable.absorb_none(cv, sp)  # random_linear_polynomial = None
    for lc_challenge, check_poly in addends:
        sp = absorbable.absorb_bytes(cv, sp, int(lc_challenge).to_bytes(_LC_CHALLENGE_BYTES, "little"))
        sp = _absorb_check_poly(cv, sp, check_poly)
    _, point = sponge.squeeze_challenges(sp, 1, _CHALLENGE_POINT_BITS)
    return point[0]


def combined_evaluation(
    cv: Curve, addends: list[tuple[int, list[int]]], point: int,
    rlp_coeffs: list[int] | None = None,
) -> int:
    """`evaluate_combined_succinct_check_polynomials(point, random_polynomial)`:
    `Σ lc_challenge_j · h_j(point)` — the combined check polynomial is linear in the
    per-input check polynomials, so its evaluation is the weighted sum of the
    succinct `h_j(point)`. When `rlp_coeffs` is given (the zk path) the degree-1
    random linear polynomial `rlp(point) = c0 + c1·point` is added on top.
    `rlp_coeffs is None` ⇒ the no-zk path (arkworks `random_polynomial = None`)."""
    eval_fr = _combined_evaluation_fr(cv, addends, point)
    if rlp_coeffs is not None:
        c0, c1 = _rlp_pair(rlp_coeffs)
        eval_fr = eval_fr + (cv.fr(c0) + cv.fr(c1) * cv.fr(point))
    return fe_value(eval_fr)


def _combined_evaluation_fr(cv: Curve, addends: list[tuple[int, list[int]]], point: int):  # type: ignore[no-untyped-def]
    """`Σ lc_challenge_j · h_j(point)` as an `fr` scalar — the field-native core of
    :func:`combined_evaluation`. Each `h_j` comes from :func:`ipa_pc.evaluate_fr` as
    an `fr` value (no per-input int decode); the public wrapper crosses the
    dtype→int boundary once at the end. Returns a numpy `fr` scalar so the zk
    path's `+ rlp(point)` stays numpy-native."""
    lc = jnp.asarray(np.array([lc_challenge for lc_challenge, _ in addends], dtype=cv.fr))
    h = jnp.concatenate(
        [ipa_pc.evaluate_fr(cv, check_poly, point).reshape(1) for _, check_poly in addends])
    return np.asarray(jnp.sum(lc * h), dtype=cv.fr)


def combine_check_polynomials(
    cv: Curve, addends: list[tuple[int, list[int]]],
    rlp_coeffs: list[int] | None = None,
) -> list[int]:
    """`combine_succinct_check_polynomials(random_polynomial)`: the dense combined
    check polynomial `Σ lc_challenge_j · h_j(X)` (length `d+1 = 2^log_d`), each `h_j`
    densely expanded via `compute_coeffs`. When `rlp_coeffs` is given (the zk path)
    the degree-1 random linear polynomial `rlp(X) = c0 + c1·X` seeds the two low
    coefficients before the linear combination — arkworks seeds
    `combined = random_polynomial` then adds the weighted check polynomials.
    `rlp_coeffs is None` ⇒ the no-zk path (arkworks `random_polynomial = None`)."""
    n = 1 << len(addends[0][1])  # 2^log_d
    combined = np.zeros(n, dtype=cv.fr)
    if rlp_coeffs is not None:
        c0, c1 = _rlp_pair(rlp_coeffs)
        combined[0] = np.array([c0], dtype=cv.fr)[0]
        combined[1] = np.array([c1], dtype=cv.fr)[0]
    for lc_challenge, check_poly in addends:
        coeffs = np.array(ipa_pc.compute_coeffs(cv, check_poly), dtype=cv.fr)
        combined = combined + np.array([lc_challenge], dtype=cv.fr) * coeffs
    return fe_values(combined)


def _rlp_pair(rlp_coeffs: list[int]) -> tuple[int, int]:
    """The degree-1 `random_linear_polynomial` coefficients `(c0, c1)`, padded with
    zero to length 2 (arkworks resizes `coeffs` to 2 before absorbing / using
    them)."""
    c0 = int(rlp_coeffs[0]) if len(rlp_coeffs) > 0 else 0
    c1 = int(rlp_coeffs[1]) if len(rlp_coeffs) > 1 else 0
    return c0, c1


def combine_zk(
    cv: Curve, params, succinct_checks: list[SuccinctCheck],
    rlp_coeffs: list[int], rlp_commitment: np.ndarray, s: np.ndarray,
    commitment_randomness: int,
) -> tuple[list[int], np.ndarray, np.ndarray, list[tuple[int, list[int]]]]:  # type: ignore[no-untyped-def]
    """`combine_succinct_check_polynomials_and_commitments` (zk): the AS sponge
    additionally absorbs the random linear polynomial — its two coefficients
    (each `to_bytes![c_i]`, separately) then its commitment — before the per-input
    loop, and the combined commitment is seeded from `rlp_commitment` (not the
    identity). Returns the lc challenges, the **non-randomized** combined
    commitment `rlp_commitment + Σ final_comm_key_j·lc_j` (which seeds
    `compute_new_challenge_zk`), the **randomized** combined commitment
    `combined + s·commitment_randomness` (the new accumulator's commitment), and
    the addends."""
    sp = _new(cv, params)
    c0, c1 = _rlp_pair(rlp_coeffs)
    sp = absorbable.absorb_bytes(cv, sp, c0.to_bytes(32, "little"))
    sp = absorbable.absorb_bytes(cv, sp, c1.to_bytes(32, "little"))
    sp = absorbable.absorb_point(cv, sp, rlp_commitment)
    for sc in succinct_checks:
        sp = _absorb_check_poly(cv, sp, sc.check_poly)
        sp = absorbable.absorb_point(cv, sp, sc.final_comm_key)
    _, lc_challenges = sponge.squeeze_challenges(sp, len(succinct_checks), _LC_CHALLENGE_BITS)

    bases = [rlp_commitment] + [sc.final_comm_key for sc in succinct_checks]
    scalars = [1] + list(lc_challenges)
    combined = curve.pedersen_commit(cv, bases, scalars)
    randomized = curve.pedersen_commit(
        cv, bases, scalars, hiding=s, randomizer=int(commitment_randomness))
    addends = [(lc, sc.check_poly) for lc, sc in zip(lc_challenges, succinct_checks)]
    return lc_challenges, combined, randomized, addends


def compute_new_challenge_zk(
    cv: Curve, params, combined_commitment: np.ndarray,
    addends: list[tuple[int, list[int]]], rlp_coeffs: list[int],
) -> int:  # type: ignore[no-untyped-def]
    """`compute_new_challenge` (zk): a fresh AS sponge absorbs the (non-randomized)
    combined commitment, then `Some(to_bytes![rlp_c0, rlp_c1])` (the random linear
    polynomial coefficients as one byte-vec, where no-zk absorbs `None`), then per
    addend the lc challenge (16 bytes) and the check polynomial; squeeze the new
    opening point (`Truncated(184)`)."""
    sp = _new(cv, params)
    sp = absorbable.absorb_point(cv, sp, combined_commitment)
    c0, c1 = _rlp_pair(rlp_coeffs)
    sp = absorbable.absorb_option_bytes(cv, sp, c0.to_bytes(32, "little") + c1.to_bytes(32, "little"))
    for lc_challenge, check_poly in addends:
        sp = absorbable.absorb_bytes(cv, sp, int(lc_challenge).to_bytes(_LC_CHALLENGE_BYTES, "little"))
        sp = _absorb_check_poly(cv, sp, check_poly)
    _, point = sponge.squeeze_challenges(sp, 1, _CHALLENGE_POINT_BITS)
    return point[0]


class AccumulatorInstance(NamedTuple):
    """The new accumulator's *instance* fields (no-zk), minus the IPA proof
    (Slice 2b): the combined commitment, the new opening point, and the combined
    evaluation."""
    commitment: np.ndarray
    point: int
    evaluation: int


class Accumulator(NamedTuple):
    """The full new accumulator instance (no-zk): the instance fields plus the IPA
    opening proof of the combined check polynomial at the new point."""
    commitment: np.ndarray
    point: int
    evaluation: int
    ipa_proof: ipa_pc.IpaProof


def prove_no_zk_instance(
    cv: Curve, params, succinct_checks: list[SuccinctCheck],
) -> AccumulatorInstance:  # type: ignore[no-untyped-def]
    """The AS no-zk prove up to the new accumulator instance: combine the succinct
    checks, derive the new opening point, and evaluate the combined check
    polynomial there."""
    _, combined_commitment, addends = combine(cv, params, succinct_checks)
    point = compute_new_challenge(cv, params, combined_commitment, addends)
    evaluation = combined_evaluation(cv, addends, point)
    return AccumulatorInstance(combined_commitment, point, evaluation)


def prove_zk_instance(
    cv: Curve, params, succinct_checks: list[SuccinctCheck],
    rlp_coeffs: list[int], rlp_commitment: np.ndarray, s: np.ndarray,
    commitment_randomness: int,
) -> AccumulatorInstance:  # type: ignore[no-untyped-def]
    """The AS zk prove up to the new accumulator instance: the random-linear-
    polynomial combine, the new opening point, and the combined evaluation. The
    accumulator's commitment is the **randomized** combined commitment
    (`+ s·commitment_randomness`); the new point is seeded from the non-randomized
    one."""
    _, combined, randomized, addends = combine_zk(
        cv, params, succinct_checks, rlp_coeffs, rlp_commitment, s, commitment_randomness)
    point = compute_new_challenge_zk(cv, params, combined, addends, rlp_coeffs)
    evaluation = combined_evaluation(cv, addends, point, rlp_coeffs)
    return AccumulatorInstance(randomized, point, evaluation)


def prove_zk_accumulator(
    cv: Curve, params, svk_h: np.ndarray, s: np.ndarray, generators: list[np.ndarray],
    succinct_checks: list[SuccinctCheck], rlp_coeffs: list[int], rlp_commitment: np.ndarray,
    commitment_randomness: int, hiding_poly_raw: list[int], hiding_rand: int,
) -> Accumulator:  # type: ignore[no-untyped-def]
    """The full AS zk prove: the instance fields (randomized commitment, new point,
    evaluation) plus `compute_new_accumulator`'s **hiding** IPA open of the combined
    check polynomial `rlp(X) + Σ lc_j·h_j(X)` at the new point — the complete new
    accumulator (with the hiding `ipa_proof`) arkworks `prove` returns. `svk_h` /
    `s` are the verifier key's IPA fold base / hiding generator; `hiding_poly_raw` /
    `hiding_rand` are the IPA open's replayed hiding randomness."""
    _, combined, randomized, addends = combine_zk(
        cv, params, succinct_checks, rlp_coeffs, rlp_commitment, s, commitment_randomness)
    point = compute_new_challenge_zk(cv, params, combined, addends, rlp_coeffs)
    evaluation = combined_evaluation(cv, addends, point, rlp_coeffs)
    coeffs = combine_check_polynomials(cv, addends, rlp_coeffs)
    ipa_proof = ipa_open.open_zk(
        cv, params, svk_h, s, generators, randomized, point, coeffs,
        hiding_poly_raw, hiding_rand, commitment_randomness)
    return Accumulator(randomized, point, evaluation, ipa_proof)


def prove_no_zk_accumulator(
    cv: Curve, params, svk_h: np.ndarray, generators: list[np.ndarray],
    succinct_checks: list[SuccinctCheck],
) -> Accumulator:  # type: ignore[no-untyped-def]
    """The full AS no-zk prove: the instance fields (`combine` + `compute_new_
    challenge` + combined evaluation) plus `compute_new_accumulator`'s IPA open of
    the combined check polynomial at the new point — the complete new accumulator
    arkworks `prove` returns."""
    _, combined_commitment, addends = combine(cv, params, succinct_checks)
    point = compute_new_challenge(cv, params, combined_commitment, addends)
    evaluation = combined_evaluation(cv, addends, point)
    coeffs = combine_check_polynomials(cv, addends)
    ipa_proof = ipa_open.open_no_zk(cv, params, svk_h, combined_commitment, point, coeffs, generators)
    return Accumulator(combined_commitment, point, evaluation, ipa_proof)


def prove_no_zk_fold(
    cv: Curve, params, svk_h: np.ndarray, generators: list[np.ndarray],
    input_insts: list[Any], acc_prev_insts: list[Any],
) -> Accumulator:  # type: ignore[no-untyped-def]
    """The AS no-zk **fold**: accumulate inputs INTO prior accumulators
    (`old_accumulators` non-empty), arkworks' `prove` with a non-empty
    `old_accumulators`.

    The fold reuses the no-fold prove wholesale. arkworks'
    `succinct_check_inputs_and_accumulators` succinct-checks the inputs first, then
    the accumulators, into ONE list; an accumulator is an `InputInstance` of the
    same shape as an input, so each is checked and combined identically. The fold is
    therefore exactly :func:`prove_no_zk_accumulator` fed
    `[inputs..., accumulators...]` — no new combine/challenge logic, just the prior
    accumulators appended (after the inputs) as further addends."""
    succinct_checks = (
        [succinct_check_input(cv, params, i) for i in input_insts]
        + [succinct_check_input(cv, params, a) for a in acc_prev_insts]
    )
    return prove_no_zk_accumulator(cv, params, svk_h, generators, succinct_checks)


def prove_zk_fold(
    cv: Curve, params, svk_h: np.ndarray, s: np.ndarray, generators: list[np.ndarray],
    input_insts: list[Any], acc_prev_insts: list[Any], rlp_coeffs: list[int],
    rlp_commitment: np.ndarray, commitment_randomness: int, hiding_poly_raw: list[int],
    hiding_rand: int,
) -> Accumulator:  # type: ignore[no-untyped-def]
    """The AS zk **fold**: accumulate (no-zk) inputs INTO prior *hiding* accumulators.

    The fold mirrors the no-zk one (:func:`prove_no_zk_fold`) — inputs first, then
    accumulators, into one succinct-check list combined identically — with one
    wrinkle: a prior accumulator from a zk prove carries a hiding IPA opening, so
    its succinct check is the **zk** path (:func:`succinct_check_input` with `s`,
    folding the hiding seed with `s` and the proof's `hiding_comm`/`rand`), while the new
    inputs stay no-zk. The combined list then runs through the same zk prove."""
    succinct_checks = (
        [succinct_check_input(cv, params, i) for i in input_insts]
        + [succinct_check_input(cv, params, a, s) for a in acc_prev_insts]
    )
    return prove_zk_accumulator(
        cv, params, svk_h, s, generators, succinct_checks, rlp_coeffs, rlp_commitment,
        commitment_randomness, hiding_poly_raw, hiding_rand)


def _input_check_poly(cv: Curve, params, inst: Any, s: np.ndarray | None):  # type: ignore[no-untyped-def]
    """The Slice-1 succinct check's round-challenge polynomial for one instance. When
    `s` (the verifier key's hiding generator) is given, the **hiding** succinct check
    runs — folding the instance's `hiding_comm` / `rand` seed with `s` before deriving
    the round challenges (a prior accumulator from a zk prove). `s is None` ⇒ the
    no-zk succinct check."""
    if s is None:
        return ipa_pc.succinct_check_challenges(
            cv, params, inst.commitment, inst.point, inst.value, inst.l_vec, inst.r_vec)
    return ipa_pc.succinct_check_challenges_zk(
        cv, params, inst.commitment, inst.point, inst.value, inst.l_vec, inst.r_vec,
        s, inst.hiding_comm, inst.rand)


def succinct_check_input(
    cv: Curve, params, inst: Any, s: np.ndarray | None = None,
) -> SuccinctCheck:
    """Run the Slice-1 succinct check on one instance (a dict-like with `commitment`,
    `point`, `evaluation`, `l_vec`, `r_vec`, `final_comm_key`) and pair the resulting
    check polynomial with its `final_comm_key`. Pass `s` (the verifier key's hiding
    generator) to run the **hiding** check on a prior zk accumulator (folding its
    `hiding_comm`/`rand`); `s is None` ⇒ the no-zk check on a fresh input."""
    return SuccinctCheck(_input_check_poly(cv, params, inst, s), inst.final_comm_key)


def decide_final_key(cv: Curve, params, generators: list[np.ndarray], inst: Any,
                     s: np.ndarray | None = None) -> np.ndarray:  # type: ignore[no-untyped-def]
    """The AS decider's size-`d` MSM (`IpaPC::check`'s final check): run the
    accumulator's succinct check, densely expand its check polynomial, and recompute
    `final_key = Σ generators_i · compute_coeffs(check_poly)_i`. The decider accepts
    iff this equals the accumulator's `final_comm_key`. This size-`d` MSM is the IPA
    accumulation's fused GPU-core target; here it is the CPU group-reduction oracle.
    Pass `s` (the verifier key's hiding generator) for a hiding accumulator — the
    **zk** succinct check folds its `hiding_comm`/`rand` seed with `s`; `s is None`
    ⇒ the no-zk decider."""
    coeffs = ipa_pc.compute_coeffs(cv, _input_check_poly(cv, params, inst, s))
    return curve.pedersen_commit(cv, generators, coeffs)
