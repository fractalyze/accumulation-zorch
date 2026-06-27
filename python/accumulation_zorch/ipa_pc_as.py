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

import numpy as np

from . import absorbable, curve, ipa_pc, sponge
from .curve import Curve
from .field import fe_values

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


def combined_evaluation(cv: Curve, addends: list[tuple[int, list[int]]], point: int) -> int:
    """`combined_check_polynomial.evaluate(point)` (no-zk) =
    `Σ lc_challenge_j · h_j(point)` — the combined check polynomial is linear in
    the per-input check polynomials, so its evaluation is the weighted sum of the
    succinct `h_j(point)`."""
    acc = np.zeros(1, dtype=cv.fr)
    for lc_challenge, check_poly in addends:
        h_at_point = np.array([ipa_pc.evaluate(cv, check_poly, point)], dtype=cv.fr)
        acc = acc + np.array([int(lc_challenge)], dtype=cv.fr) * h_at_point
    return int(np.asarray(acc[0]).astype(object))


def combine_check_polynomials(cv: Curve, addends: list[tuple[int, list[int]]]) -> list[int]:
    """`combine_succinct_check_polynomials` (no-zk): the dense combined check
    polynomial `Σ lc_challenge_j · h_j(X)` (length `d+1 = 2^log_d`), each `h_j`
    densely expanded via `compute_coeffs`."""
    n = 1 << len(addends[0][1])  # 2^log_d
    combined = np.zeros(n, dtype=cv.fr)
    for lc_challenge, check_poly in addends:
        coeffs = np.array(ipa_pc.compute_coeffs(cv, check_poly), dtype=cv.fr)
        combined = combined + np.array([int(lc_challenge)], dtype=cv.fr) * coeffs
    return fe_values(combined)


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
    ipa_proof = ipa_pc.open_no_zk(cv, params, svk_h, combined_commitment, point, coeffs, generators)
    return Accumulator(combined_commitment, point, evaluation, ipa_proof)


def succinct_check_input(cv: Curve, params, inst: Any) -> SuccinctCheck:
    """Run the Slice-1 succinct check on one input instance (a dict-like with
    `commitment`, `point`, `evaluation`, `l_vec`, `r_vec`, `final_comm_key`) and
    pair the resulting check polynomial with the input's `final_comm_key`."""
    check_poly = ipa_pc.succinct_check_challenges(
        cv, params, inst.commitment, inst.point, inst.value, inst.l_vec, inst.r_vec)
    return SuccinctCheck(check_poly, inst.final_comm_key)
