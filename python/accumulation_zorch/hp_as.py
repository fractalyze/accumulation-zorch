"""HP accumulation scheme prover (port of `hp_as/mod.rs`), no-zk path.

Ports `ASForHadamardProducts::prove` for `make_zk = false`: absorb the inputs,
squeeze the `mu` challenges, build the product-polynomial coefficient vectors
`t_vecs`, commit them (the `Proof`), squeeze `nu`, and fold the inputs +
product-poly commitments into the combined accumulator instance.
The zk path (hiding vectors/commitments) is slice 6.
"""

import struct
from typing import Any, NamedTuple

import frx
import frx.numpy as fnp
import numpy as np
from frx import lax
from zorch.hash.duplex_sponge import DuplexSponge

from . import absorbable, curve, field, sponge
from .curve import Curve, FrScalar

CHALLENGE_SIZE = 128  # bits, matching ark hp_as::CHALLENGE_SIZE
# Both Pasta scalar fields are 254-cap > 128, so this is the curve-invariant 128.
_CHALLENGE_BITS = min(CHALLENGE_SIZE, sponge.FR_CAPACITY)  # squeeze window per challenge

# A point triple `(comm_1, comm_2, comm_3)`; a vector of fr is `list[int]`.
Instance = tuple[np.ndarray, np.ndarray, np.ndarray]


def serialize_proof(cv: Curve, low: list[np.ndarray], high: list[np.ndarray]) -> bytes:
    """`Proof` CanonicalSerialize (no-zk): the `low` and `high` commitment
    `Vec<G>` (each u64 length + 33B points), then the `None` hiding flag."""
    out = struct.pack("<Q", len(low)) + b"".join(curve.point_to_bytes(cv, p) for p in low)
    out += struct.pack("<Q", len(high)) + b"".join(curve.point_to_bytes(cv, p) for p in high)
    return out + b"\x00"


def serialize_instance(cv: Curve, instance: Instance) -> bytes:
    """`InputInstance` CanonicalSerialize: `comm_1 ‖ comm_2 ‖ comm_3` (33B each)."""
    return b"".join(curve.point_to_bytes(cv, p) for p in instance)


class HpNoZkCore(NamedTuple):
    """Single-input make_zk=false HP prove outputs (on-device). The fold is the
    identity: with one input `mu = nu = [1]` and the lone product-poly row is the
    skipped `(n-1)`-th, so there are no product-poly commitments — the combined
    instance is the input's own commitments and the openings are its own vectors.
    The transcript (absorb instance + None hiding, squeeze nu) is run for
    faithfulness but does not affect the output."""
    instance: frx.Array  # (3,) affine
    a_open: frx.Array    # (L,) fr
    b_open: frx.Array    # (L,) fr


def prove_no_zk_core(cv: Curve, real_inst: frx.Array, a_real: frx.Array, b_real: frx.Array,
                     supported_num_elems: int, params: Any,
                     base_sponge: DuplexSponge | None = None) -> HpNoZkCore:
    """make_zk=false HP prove over a single input (no prior accumulators) returning
    on-device frx — the R1CS-NARK-AS no-zk entry point, so the HP step threads on
    without a host hop. `real_inst` the `(3,)` input commitments; `a_real`/`b_real`
    the `(L,)` opening vectors. Plain so it inlines into the AS top-level
    `@frx.jit`."""
    sp = sponge.new_sponge(params) if base_sponge is None else base_sponge
    sp = absorbable.absorb_u64(cv, sp, supported_num_elems)
    sp = absorbable.absorb_points_frx(cv, sp, real_inst)
    sp = absorbable.absorb_none(cv, sp)  # hiding_comms = None
    # num_inputs == 1: mu = [1] (no sponge consumed), the lone
    # t_vec row is the skipped (n-1)-th (no product-poly commitments to absorb), and
    # nu folds the identity — the nu squeeze runs only to mirror the transcript.
    sp, _nu = squeeze_nu_frx(cv, sp, 1)
    return HpNoZkCore(real_inst, a_real, b_real)


def materialize_no_zk(core: HpNoZkCore) -> tuple[
        Instance, tuple[frx.Array, frx.Array], list[np.ndarray], list[np.ndarray]]:
    """Materialize an `HpNoZkCore` to the host serialize shape `(instance, (a_open,
    b_open), low, high)` — `low`/`high` are empty for the single-input fold."""
    inst_np = np.asarray(core.instance)
    instance = (inst_np[0], inst_np[1], inst_np[2])
    return instance, (core.a_open, core.b_open), [], []


# --- zk path (hiding vectors / commitments) --------------------------------


class HpZkCore(NamedTuple):
    """`prove_zk`'s outputs as on-device frx arrays — the un-materialized form the
    R1CS-NARK-AS path threads on so the HP fold stays in the one prove trace.
    `instance` is the combined `(comm_1, comm_2, comm_3)` (3,) affine; `low`/`high`
    the product-poly commitments; `hiding_comms` the `(comm_h1, h2, h3)` (3,);
    `a_open`/`b_open` the `(L,)` openings; `rand` the `(3,)` `(rand_1, rand_2,
    rand_3)`."""
    instance: frx.Array      # (3,) affine
    a_open: frx.Array        # (L,) fr
    b_open: frx.Array        # (L,) fr
    rand: frx.Array          # (3,) fr
    low: frx.Array           # (n_low,) affine
    high: frx.Array          # (n_high,) affine
    hiding_comms: frx.Array  # (3,) affine


def squeeze_mu_frx(cv: Curve, sp: DuplexSponge, num_inputs: int) -> tuple[DuplexSponge, frx.Array]:
    """`squeeze_mu_challenges` (make_zk) as frx: `[1, c_1, …, c_{n-1}, mu_n]` where
    `mu_n = mu[1]·mu[n-1]` (the extra entry that folds the hiding terms). Returns
    the `(n+1,)` fr array."""
    mu = fnp.asarray(np.array([1], dtype=cv.fr))
    if num_inputs > 1:
        sp, rest = sponge.squeeze_challenges_frx(sp, num_inputs - 1, _CHALLENGE_BITS, cv)
        mu = fnp.concatenate([mu, rest])
    mu_n = mu[1] * mu[num_inputs - 1]
    return sp, fnp.concatenate([mu, mu_n.reshape(1)])


def squeeze_nu_frx(cv: Curve, sp: DuplexSponge, num_inputs: int) -> tuple[DuplexSponge, frx.Array]:
    """`squeeze_nu_challenges` as frx: one truncated-128 `nu` expanded to its
    `2n-1` powers `[nu^0, …, nu^{2n-2}]`."""
    sp, nu = sponge.squeeze_challenges_frx(sp, 1, _CHALLENGE_BITS, cv)
    return sp, field.powers(nu, 2 * num_inputs - 1)


def _product_poly_comm_frx(bases: frx.Array, t_vecs: frx.Array,
                           num_inputs: int) -> tuple[list[frx.Array], list[frx.Array]]:
    """`compute_product_poly_comm` (on-device): commit every
    `t_vec` row except the `(n-1)`-th, split into `low` (`i < n-1`) / `high`
    (`i > n-1`). `bases` is the pre-stacked generators (no hiding base). Plain frx
    (no `np.asarray`) so it inlines into the prove trace. Curve-agnostic — the
    dtype rides on `t_vecs`/`bases`."""
    low: list[frx.Array] = []
    high: list[frx.Array] = []
    for i in range(t_vecs.shape[0]):
        if i == num_inputs - 1:
            continue
        comm = lax.msm(t_vecs[i], bases)
        (low if i < num_inputs - 1 else high).append(comm)
    return low, high


def _combine_randomness(cv: Curve, rands: list[int | None], challenges: list[int],
                        hiding: int | None = None) -> FrScalar:
    """`combine_randomness`: `Σ rands[i]·challenges[i]` over the `Some` entries
    (`None` contributes nothing), plus an optional hiding addend — as an `fr`
    scalar (fed to `pedersen_commit`'s randomizer), never decoded to a python int."""
    pairs = [(int(r), int(challenges[i])) for i, r in enumerate(rands) if r is not None]
    if pairs:
        rs, cs = zip(*pairs)
        acc = np.sum(np.array(rs, dtype=cv.fr) * np.array(cs, dtype=cv.fr))
    else:
        acc = cv.fr(0)
    if hiding is not None:
        acc = acc + cv.fr(int(hiding))
    return acc


def _prove_zk_segment(cv: Curve, params: Any, supported_num_elems: int, bases_h: frx.Array,
                      id_pt: frx.Array, hiding_a: int, hiding_b: int, hr1: int, hr2: int,
                      hr3: int, sp: DuplexSponge, real_inst: frx.Array, a_real: frx.Array,
                      b_real: frx.Array, input_rand: frx.Array, old_inst: frx.Array | None = None,
                      old_a: frx.Array | None = None, old_b: frx.Array | None = None,
                      old_rand: frx.Array | None = None,
                      hp_rand: frx.Array | None = None) -> HpZkCore:
    """The make_zk HP prove as on-device compute (plain, so it inlines into both
    `prove_zk`'s `@frx.jit` and the AS top-level trace). Commits the hiding +
    product-poly terms, squeezes mu/nu off `sp` (the caller-supplied base sponge),
    and folds the real input together with the **second input** — the IVC fold's
    old accumulator, or the zero placeholder arkworks pads a single input with —
    into the combined instance / openings / randomness. Both keep `num_inputs == 2`
    (`prove_with_backend` only pads when `num_all_inputs == 1`), so the t_vecs /
    low / high shapes are the same for the init and the fold.

    `bases_h` is the pre-stacked generators + hiding base (an affine jit argument;
    the product-poly bases are `bases_h[:L]`); `id_pt` the `(1,)` identity used for
    the placeholder commitments. `real_inst` is the `(3,)` input commitments
    `(comm_1, comm_2, comm_3)`; `a_real` / `b_real` the `(L,)` witness opening
    vectors; `input_rand` the `(3,)` `(rand_1, rand_2, rand_3)`.

    The second input is the order-1 addend (`inputs.chain(old_accumulators)`):
    `old_inst` its `(3,)` HP commitments, `old_a` / `old_b` its `(L,)` opening
    vectors, `old_rand` its `(3,)` randomness. All default to the inert zero
    placeholder (identity commitments, zero vectors / randomness), so the
    single-input path is byte-identical; the IVC fold passes the old accumulator."""
    fr_one = fnp.asarray(np.array([1], dtype=cv.fr))
    num_inputs = 2  # one real input + (the old accumulator | the zero placeholder)
    L = a_real.shape[0]
    bases = bases_h[:L]  # generators without the trailing hiding base
    # Row 1 of the fold: the old accumulator (IVC fold) or the inert zero
    # placeholder (single-input init). Commitments default to the identity.
    row1_a = fnp.zeros_like(a_real) if old_a is None else old_a
    row1_b = fnp.zeros_like(b_real) if old_b is None else old_b
    row1_comms = fnp.concatenate([id_pt, id_pt, id_pt]) if old_inst is None else old_inst
    row1_rand = fnp.zeros_like(input_rand) if old_rand is None else old_rand
    a = fnp.stack([a_real, row1_a])  # (n, L), row 1 = old accumulator / placeholder
    b = fnp.stack([b_real, row1_b])
    # The hiding values + randomizers are host constants (baked half-step / fold)
    # or a runtime `(5,)` fr array `[hiding_a, hiding_b, hr1, hr2, hr3]` (the
    # general prover); the latter lets one lowered core prove any HP randomness.
    if hp_rand is not None:
        hiding_a_vec = fnp.broadcast_to(hp_rand[0], (L,))
        hiding_b_vec = fnp.broadcast_to(hp_rand[1], (L,))
        hr = hp_rand[2:5]
        hr1, hr2, hr3 = hr[0], hr[1], hr[2]
    else:
        hiding_a_vec = fnp.asarray(np.array([hiding_a] * L, dtype=cv.fr))
        hiding_b_vec = fnp.asarray(np.array([hiding_b] * L, dtype=cv.fr))
        hr = fnp.asarray(np.array([hr1, hr2, hr3], dtype=cv.fr))

    # Hiding commitments (the cross term mixes input₀'s b-row with the row-1 a-row).
    comm_h1 = curve.commit_hiding(cv, hiding_a_vec, hr1, bases_h)
    comm_h2 = curve.commit_hiding(cv, hiding_b_vec, hr2, bases_h)
    rand_prods_sum = hiding_a_vec * b[0] + a[num_inputs - 1] * hiding_b_vec
    comm_h3 = curve.commit_hiding(cv, rand_prods_sum, hr3, bases_h)
    hiding_comms = fnp.stack([comm_h1, comm_h2, comm_h3])  # (3,)

    # Transcript: supported size, the input commitments (+ the row-1 commitments),
    # the hiding commitments — all threaded as frx, no host hop.
    sp = absorbable.absorb_u64(cv, sp, supported_num_elems)
    sp = absorbable.absorb_points_frx(cv, sp, fnp.concatenate([real_inst, row1_comms]))
    sp = absorbable.absorb_option_points_frx(cv, sp, hiding_comms)

    sp, mu = squeeze_mu_frx(cv, sp, num_inputs)  # (3,) = [1, c, mu_n]
    t_vecs = field.t_vecs_zk(a, b, mu[:num_inputs], hiding_a_vec, hiding_b_vec,
                              mu[num_inputs].reshape(1), mu[1].reshape(1))
    low, high = _product_poly_comm_frx(bases, t_vecs, num_inputs)

    sp = absorbable.absorb_points_frx(cv, sp, fnp.stack(low + high))
    sp, nu = squeeze_nu_frx(cv, sp, num_inputs)  # (2n-1,) powers of nu

    combined = mu[:num_inputs] * nu[:num_inputs]  # (n,)
    mu_n = mu[num_inputs:num_inputs + 1]          # (1,)

    # Combined instance — each commitment one fold (row 1 = old acc / identity),
    # plus the coeff·hiding addend appended as an extra lax.msm term.
    comm_1s = fnp.concatenate([real_inst[0:1], row1_comms[0:1]])
    comm_2s = fnp.concatenate([real_inst[1:2], row1_comms[1:2]])
    comm_3s = fnp.concatenate([real_inst[2:3], row1_comms[2:3]])
    cc1 = lax.msm(fnp.concatenate([combined, mu_n]),
                     fnp.concatenate([comm_1s, hiding_comms[0:1]]))
    cc2 = lax.msm(fnp.concatenate([nu[:num_inputs], mu[1:2]]),
                     fnp.concatenate([fnp.flip(comm_2s, 0), hiding_comms[1:2]]))
    comm3_combine = lax.msm(fnp.concatenate([mu[:num_inputs], mu_n]),
                               fnp.concatenate([comm_3s, hiding_comms[2:3]]))
    low_addend = lax.msm(nu[: len(low)], fnp.stack(low))
    high_addend = lax.msm(nu[num_inputs:num_inputs + len(high)], fnp.stack(high))
    cc3 = lax.msm(fnp.concatenate([nu[num_inputs - 1:num_inputs], fr_one, fr_one]),
                     fnp.stack([comm3_combine, low_addend, high_addend]))
    instance = fnp.stack([cc1, cc2, cc3])

    # Combined openings + randomness over both rows (placeholder row → inert).
    a_open = field.combine_vectors(a, combined) + mu[num_inputs] * hiding_a_vec
    b_open = field.combine_vectors(fnp.flip(b, 0), nu[:num_inputs]) + mu[1] * hiding_b_vec
    a_rand = (input_rand[0] * combined[0] + row1_rand[0] * combined[1]
              + hr[0] * mu[num_inputs])
    b_rand = row1_rand[1] * nu[0] + input_rand[1] * nu[1] + hr[1] * mu[1]
    prod_rand = (input_rand[2] * mu[0] + row1_rand[2] * mu[1]
                 + hr[2] * mu[num_inputs]) * nu[num_inputs - 1]
    rand = fnp.stack([a_rand, b_rand, prod_rand])

    return HpZkCore(instance, a_open, b_open, rand, fnp.stack(low), fnp.stack(high), hiding_comms)


def prove_zk_core(cv: Curve, bases_h: frx.Array, id_pt: frx.Array, real_inst: frx.Array,
                  a_real: frx.Array, b_real: frx.Array, input_rand: frx.Array,
                  supported_num_elems: int, params: Any, hiding_a: int, hiding_b: int,
                  hiding_rand_1: int, hiding_rand_2: int, hiding_rand_3: int,
                  base_sponge: DuplexSponge | None = None, old_inst: frx.Array | None = None,
                  old_a: frx.Array | None = None, old_b: frx.Array | None = None,
                  old_rand: frx.Array | None = None, hp_rand: frx.Array | None = None) -> HpZkCore:
    """make_zk HP prove returning on-device frx (`HpZkCore`) — the R1CS-NARK-AS
    entry point, so the HP fold threads on without a host hop. `bases_h` the
    pre-stacked generators + hiding base; `id_pt` the `(1,)` identity; `real_inst`
    the `(3,)` input commitments; `a_real` / `b_real` the `(L,)` opening vectors;
    `input_rand` the `(3,)` input randomness — all frx (off `lax.msm` / `M·z`).
    `base_sponge` is the AS `AS-FOR-HP-2020` fork (a fresh sponge if omitted).

    `old_inst` / `old_a` / `old_b` / `old_rand` are the IVC fold's old-accumulator
    HP input (instance `(3,)`, opening vectors `(L,)`, randomness `(3,)`); omitting
    them (the default) is the single-input init — the second input is the inert
    zero placeholder. `hp_rand` (the general prover) lifts the hiding values + randomizers
    to a runtime `(5,)` fr array so one lowered core proves any HP randomness;
    omitting it bakes them as host constants. Plain so it inlines into the AS
    top-level `@frx.jit`."""
    sp = sponge.new_sponge(params) if base_sponge is None else base_sponge
    return _prove_zk_segment(cv, params, supported_num_elems, bases_h, id_pt, hiding_a,
                             hiding_b, hiding_rand_1, hiding_rand_2, hiding_rand_3, sp,
                             real_inst, a_real, b_real, input_rand, old_inst=old_inst,
                             old_a=old_a, old_b=old_b, old_rand=old_rand, hp_rand=hp_rand)


def prove_zk(cv: Curve, generators: list[np.ndarray], hiding: np.ndarray, instances: list[Instance],
             a_vecs: list[list[int]], b_vecs: list[list[int]],
             input_rands: list[tuple[int, int, int] | None], supported_num_elems: int,
             params: Any, hiding_a: int, hiding_b: int, hiding_rand_1: int, hiding_rand_2: int,
             hiding_rand_3: int, base_sponge: DuplexSponge | None = None) -> tuple[
                 Instance, tuple[frx.Array, frx.Array, frx.Array],
                 list[np.ndarray], list[np.ndarray], Instance]:
    """zk HP prove over a single real input (the zero placeholder is added by the
    core, as the make_zk path does), replaying the prover's hiding randomness.
    Materializes the on-device `prove_zk_core` to `(instance, (a_open, b_open,
    (rand_1, rand_2, rand_3)), low, high, hiding_comms)` at the serialize seam."""
    assert len(instances) == 1, "this prover folds a single real input"
    L = len(a_vecs[0])
    bases_h = curve.stack_affine(cv, list(generators[:L]) + [hiding])
    real_inst = curve.stack_affine(cv, list(instances[0]))
    id_pt = curve.stack_affine(cv, [cv.g1((0, 0))])  # (1,) identity affine
    a_real = fnp.asarray(np.array(a_vecs[0], dtype=cv.fr))
    b_real = fnp.asarray(np.array(b_vecs[0], dtype=cv.fr))
    ir = input_rands[0]
    assert ir is not None
    input_rand = fnp.asarray(np.array(list(ir), dtype=cv.fr))

    core = prove_zk_core(cv, bases_h, id_pt, real_inst, a_real, b_real, input_rand,
                         supported_num_elems, params, hiding_a, hiding_b,
                         hiding_rand_1, hiding_rand_2, hiding_rand_3, base_sponge=base_sponge)
    return materialize_zk(core)


def materialize_zk(core: HpZkCore) -> tuple[
        Instance, tuple[frx.Array, frx.Array, frx.Array],
        list[np.ndarray], list[np.ndarray], Instance]:
    """Materialize an `HpZkCore` to the host serialize shape `(instance, (a_open,
    b_open, (rand_1, rand_2, rand_3)), low, high, hiding_comms)` — the serialize
    seam shared by `prove_zk` and the R1CS-NARK-AS path that embeds the HP proof."""
    inst_np = np.asarray(core.instance)
    instance = (inst_np[0], inst_np[1], inst_np[2])
    witness = (core.a_open, core.b_open, core.rand)
    low = [np.asarray(core.low[i]) for i in range(core.low.shape[0])]
    high = [np.asarray(core.high[i]) for i in range(core.high.shape[0])]
    hc_np = np.asarray(core.hiding_comms)
    hiding_comms = (hc_np[0], hc_np[1], hc_np[2])
    return instance, witness, low, high, hiding_comms


def serialize_witness_zk(cv: Curve, witness: tuple[frx.Array, frx.Array, frx.Array]) -> bytes:
    """`InputWitness` CanonicalSerialize (zk): `a_vec`, `b_vec`, then `Some`
    randomness (`rand_1, rand_2, rand_3`)."""
    a_vec, b_vec, rands = witness
    out = curve.serialize_fr_vec(cv, a_vec) + curve.serialize_fr_vec(cv, b_vec) + b"\x01"
    out += np.asarray(rands, dtype=cv.fr).tobytes()
    return out


# Shared linear-combination primitives (ark's pub(crate)
# `ASForHadamardProducts::combine_*`), reused by the R1CS-NARK-AS path.
combine_randomness = _combine_randomness


def serialize_proof_zk(cv: Curve, low: list[np.ndarray], high: list[np.ndarray],
                       hiding_comms: Instance) -> bytes:
    """`Proof` CanonicalSerialize (zk): the product-poly commitments, then `Some`
    hiding commitments (`comm_1, comm_2, comm_3`)."""
    out = struct.pack("<Q", len(low)) + b"".join(curve.point_to_bytes(cv, p) for p in low)
    out += struct.pack("<Q", len(high)) + b"".join(curve.point_to_bytes(cv, p) for p in high)
    out += b"\x01" + b"".join(curve.point_to_bytes(cv, c) for c in hiding_comms)
    return out
