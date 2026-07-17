"""R1CS-NARK-AS accumulation prover (port of `r1cs_nark_as/mod.rs`), no-zk path.

Ports `ASForR1CSNark::prove` for `make_zk = false` over a single input and no
prior accumulators — the case `src/oracle.rs` pins to arkworks. The prover:

1. Recomputes the NARK first-round commitments `comm_{a,b,c} = commit(M·z)` for
   `M ∈ {A,B,C}` and `z = r1cs_input ‖ blinded_witness` (the slice-3 path; the
   AS prover reads these from the input instance, but recomputing is identical
   and reuses proven code).
2. Builds the HP input instance `(comm_a, comm_b, comm_c)` and witness
   `(A·z, B·z)`, then runs `hp_as.prove_no_zk_core` on the `AS-FOR-HP-2020`-forked
   sponge to get the HP accumulator + proof.
3. Folds into the combined accumulator. With a single addend and no zk, the
   `beta` challenges are `[1]` (`compute_beta_challenges` squeezes
   `num_addends - 1 = 0` elements), so the fold is the identity: the combined
   `r1cs_input` / `comm_{a,b,c}` are the input's own.

Deferred to slice 6 (the zk path), because they have **no effect on the no-zk
single-input bytes** — confirmed against `mod.rs`:

* `gamma` (`compute_blinded_commitments`) is gated on first-round randomness,
  which is `None` for no-zk, so the blinded commitments are the raw ones.
* the `as_sponge` `beta` absorb (`as_matrices_hash`, instances, proof
  randomness) is discarded because a single addend forces `beta = [1]`.
* `hash_matrices` (blake2b) feeds only `gamma` and `beta`, both inert above.

These all become live in slice 6 (gamma blinds the witness; beta combines
multiple addends), where they get a real byte-output anchor.
"""

import functools
from typing import Any, NamedTuple

import frx
import frx.numpy as fnp
import numpy as np
from frx import lax

from . import absorbable, curve, field, hp_as, nark, sponge
from .curve import Curve, FrVec

# Challenge squeeze window (ark `CHALLENGE_SIZE`, capped at fr capacity). Both
# Pasta scalar fields are 254-cap > 128, so this is the curve-invariant 128.
_CHALLENGE_BITS = min(sponge.CHALLENGE_SIZE, sponge.FR_CAPACITY)

# ark `r1cs_nark_as` protocol names — the domains the AS path forks the HP /
# AS sponges with (the AS base sponge is a fresh `S::new()`, so fork-from-fresh).
HP_AS_PROTOCOL_NAME = b"AS-FOR-HP-2020"
AS_PROTOCOL_NAME = b"AS-FOR-R1CS-NARK-2020"


# --- the accumulate / fold seam (structured, for callers proving their own) ---


class Randomness(NamedTuple):
    """Every value arkworks' zk prover samples for one accumulate or fold.

    The prove entry points take these explicitly because the byte-match replays
    arkworks' draws rather than re-deriving its RNG; :func:`sample_randomness` is
    the counterpart for a caller proving its own statement, where the values only
    have to be uniform."""

    nark_r: list[int]                    # witness-blinding vector, one per witness wire
    nark_blinders: tuple[int, int, int, int, int, int, int, int]
    as_r1cs_r_input: int
    as_r1cs_r_witness: int
    as_rand: tuple[int, int, int]        # sigma_a, sigma_b, sigma_c
    hp_hiding_a: int
    hp_hiding_b: int
    hp_rand: tuple[int, int, int]        # rand_1, rand_2, rand_3


def sample_randomness(cv: Curve, rng: np.random.Generator, num_witness: int) -> Randomness:
    """Draw a :class:`Randomness` uniformly over `fr`.

    Not arkworks' RNG — a fold's soundness rests on the Fiat-Shamir challenges
    (squeezed from the transcript, never sampled here), so these only need to be
    uniform. A byte-match against arkworks must pass its draws in instead."""
    def fr() -> int:
        # Reduce a 512-bit draw rather than reject: `fr` is 255-bit, so the modulo
        # bias is ~2^-257, and `rng.integers` cannot express a range this wide.
        return int.from_bytes(rng.bytes(64), "little") % cv.fr_modulus

    return Randomness(
        nark_r=[fr() for _ in range(num_witness)],
        nark_blinders=tuple(fr() for _ in range(8)),  # type: ignore[arg-type]
        as_r1cs_r_input=fr(),
        as_r1cs_r_witness=fr(),
        as_rand=(fr(), fr(), fr()),
        hp_hiding_a=fr(),
        hp_hiding_b=fr(),
        hp_rand=(fr(), fr(), fr()),
    )


class FoldedAccumulator(NamedTuple):
    """An accumulator as the *fold* consumes it — what `prove`/`fold` emit and what
    the next fold takes back, so a chain needs no deserializer.

    Distinct from :class:`Accumulator` (the decider's view) by `comms`: the fold
    folds the stored commitments, the decider recomputes them. Serialized bytes stay
    the arkworks-facing form; this is the in-process one.

    `comms` is `comm_{a,b,c}` followed by the embedded HP instance's
    `hp_comm_{1,2,3}` — all six, in that order, because the fold slices the HP
    triple straight back out of it (`acc_comms[3:6]`)."""

    r1cs_input: FrVec
    comms: tuple[np.ndarray, ...]
    blinded_witness: FrVec
    sigma_abc: tuple[int, int, int]
    hp_a_vec: FrVec
    hp_b_vec: FrVec
    hp_rand: tuple[int, int, int]

    def to_decide(self) -> "Accumulator":
        """The decider's view of this accumulator. The vectors are `FrVec` on both
        sides now, so they pass through — no list() round-trip."""
        return Accumulator(
            r1cs_input=self.r1cs_input, blinded_witness=self.blinded_witness,
            hp_a_vec=self.hp_a_vec, hp_b_vec=self.hp_b_vec,
            sigmas=self.sigma_abc, hp_rand=self.hp_rand)


def _fr_triple(arr: Any) -> tuple[int, int, int]:
    v = np.asarray(arr).reshape(-1)
    return (int(v[0]), int(v[1]), int(v[2]))


def _materialize_zk(cv: Curve, core_out: tuple, r1cs_r_input: list[int]
                    ) -> tuple[FoldedAccumulator, bytes, bytes, bytes]:
    """The serialize seam shared by the zk prove and the zk fold — their cores emit
    the same ten leaves. Returns the structured accumulator alongside the bytes so a
    caller can fold or decide the result without parsing what was just written."""
    (combined_input, cca, ccb, ccc, cbw, csig, comm_r_a, comm_r_b, comm_r_c, hp_core) = core_out
    hp_instance, hp_witness, low, high, hiding_comms = hp_as.materialize_zk(hp_core)
    r1cs_comms = (np.asarray(cca), np.asarray(ccb), np.asarray(ccc))
    acc_instance = _serialize_acc_instance(cv, combined_input, *r1cs_comms, hp_instance)
    acc_witness = _serialize_acc_witness_zk(cv, cbw, hp_witness, csig)
    proof = _serialize_proof_zk(cv, low, high, hiding_comms, r1cs_r_input,
                                (np.asarray(comm_r_a), np.asarray(comm_r_b),
                                 np.asarray(comm_r_c)))
    acc = FoldedAccumulator(
        r1cs_input=np.asarray(combined_input, dtype=cv.fr),
        comms=r1cs_comms + tuple(np.asarray(p, dtype=cv.g1) for p in hp_instance),
        blinded_witness=np.asarray(cbw, dtype=cv.fr), sigma_abc=_fr_triple(csig),
        hp_a_vec=np.asarray(hp_witness[0], dtype=cv.fr),
        hp_b_vec=np.asarray(hp_witness[1], dtype=cv.fr),
        hp_rand=_fr_triple(hp_witness[2]))
    return acc, acc_instance, acc_witness, proof


def _serialize_acc_instance(cv: Curve, r1cs_input: FrVec, comm_a: np.ndarray,
                            comm_b: np.ndarray, comm_c: np.ndarray,
                            hp_instance: hp_as.Instance) -> bytes:
    """`AccumulatorInstance` CanonicalSerialize: `r1cs_input` (`Vec<Fr>`), the
    three commitments (33B compressed), then the embedded HP instance."""
    out = curve.serialize_fr_vec(cv, r1cs_input)
    out += (curve.point_to_bytes(cv, comm_a) + curve.point_to_bytes(cv, comm_b)
            + curve.point_to_bytes(cv, comm_c))
    out += hp_as.serialize_instance(cv, hp_instance)
    return out


def _serialize_acc_witness(cv: Curve, blinded_witness: FrVec, hp_a_vec: frx.Array,
                           hp_b_vec: frx.Array) -> bytes:
    """`AccumulatorWitness` CanonicalSerialize: `r1cs_blinded_witness`
    (`Vec<Fr>`), the HP witness (`a_vec`, `b_vec`, then `None` hiding flag), then
    the `None` accumulator-witness-randomness flag (both `None` for no-zk)."""
    out = curve.serialize_fr_vec(cv, blinded_witness)
    out += curve.serialize_fr_vec(cv, hp_a_vec) + curve.serialize_fr_vec(cv, hp_b_vec) + b"\x00"  # hp randomness None
    out += b"\x00"  # AccumulatorWitness.randomness = None
    return out


def _serialize_proof(cv: Curve, low: list[np.ndarray], high: list[np.ndarray]) -> bytes:
    """AS `Proof` CanonicalSerialize: the HP proof (product-poly commitments +
    `None` hiding flag), then the `None` AS proof-randomness flag."""
    return hp_as.serialize_proof(cv, low, high) + b"\x00"  # Proof.randomness = None


def _build_no_zk_core(cv: Curve, a: nark.Matrix, b: nark.Matrix, c: nark.Matrix,
                      r1cs_input: FrVec, blinded_witness: FrVec,
                      generators: list[np.ndarray], supported_num_elems: int,
                      params: Any) -> tuple:
    """Build the fused no-zk prove `@frx.jit` core (the **general**
    prover). The circuit (`a`/`b`/`c`, baked) is fixed, but the **assignment is a
    runtime input**: the core takes `(bases, r1cs_input, blinded_witness)` with the
    latter two device `fr` arrays — not captured constants — so one lowered core
    proves any assignment, not a single baked fixture. The no-zk NARK has no hiding
    base (no `id_pt`) and no randomness. Returns
    `(core_fn, bases, r1cs_input_arr, blinded_witness_arr)`: the example assignment
    arrays carry the runtime shapes for lowering; `prove_no_zk` runs + serializes,
    `export/export_prove.py` lowers ONE core (no per-seed `.mlirbc`). Output leaves:
    `comm_a, comm_b, comm_c, hp.instance(3,), hp.a_open, hp.b_open`."""
    rows = len(a)  # num_constraints; a/b/c share the row count
    n = len(r1cs_input) + len(blinded_witness)  # z length = the circuit's num_vars (static)
    bases = curve.stack_affine(cv, generators[:rows])
    a_dense, b_dense, c_dense = (fnp.asarray(nark.to_dense(cv, m, n)) for m in (a, b, c))
    r1cs_input_arr = fnp.asarray(np.array(list(r1cs_input), dtype=cv.fr))
    blinded_witness_arr = fnp.asarray(np.array(list(blinded_witness), dtype=cv.fr))

    @frx.jit
    def _core(bases: frx.Array, r1cs_input: frx.Array, blinded_witness: frx.Array) -> tuple:
        z = fnp.concatenate([r1cs_input, blinded_witness])
        a_arr = field.matvec(a_dense, z)
        b_arr = field.matvec(b_dense, z)
        c_arr = field.matvec(c_dense, z)
        comm_a, comm_b, comm_c = (lax.msm(a_arr, bases), lax.msm(b_arr, bases),
                                  lax.msm(c_arr, bases))
        # HP input: instance (comm_a, comm_b, comm_prod=comm_c) + opening (A·z, B·z).
        hp_sponge = absorbable.fork(cv, sponge.new_sponge(params), HP_AS_PROTOCOL_NAME)
        hp = hp_as.prove_no_zk_core(cv, fnp.stack([comm_a, comm_b, comm_c]), a_arr, b_arr,
                                    supported_num_elems, params, base_sponge=hp_sponge)
        return comm_a, comm_b, comm_c, hp

    return _core, bases, r1cs_input_arr, blinded_witness_arr


def prove_no_zk(cv: Curve, a: nark.Matrix, b: nark.Matrix, c: nark.Matrix, r1cs_input: FrVec,
                blinded_witness: FrVec, generators: list[np.ndarray],
                supported_num_elems: int, params: Any) -> tuple[bytes, bytes, bytes]:
    """no-zk `ASForR1CSNark::prove` over a single input, no prior accumulators.

    `a`/`b`/`c` are sparse `Matrix<Fr>` (rows of `(coeff, var_index)`);
    `r1cs_input`/`blinded_witness` are the `Vec<Fr>` assignment (`FrVec`: fr ints,
    `cv.fr` elements, or a field array — the prover canonicalizes either way);
    `generators` are the committer key's points. Returns the serialized
    `(acc_instance, acc_witness, proof)`.
    """
    core_fn, bases, r1cs_input_arr, blinded_witness_arr = _build_no_zk_core(
        cv, a, b, c, r1cs_input, blinded_witness, generators, supported_num_elems, params)
    comm_a, comm_b, comm_c, hp = core_fn(bases, r1cs_input_arr, blinded_witness_arr)
    hp_instance, (hp_a_vec, hp_b_vec), low, high = hp_as.materialize_no_zk(hp)
    acc_instance = _serialize_acc_instance(cv, r1cs_input, np.asarray(comm_a), np.asarray(comm_b),
                                           np.asarray(comm_c), hp_instance)
    acc_witness = _serialize_acc_witness(cv, blinded_witness, hp_a_vec, hp_b_vec)
    proof = _serialize_proof(cv, low, high)
    return acc_instance, acc_witness, proof


# --- zk path ---------------------------------------------------------------

def _serialize_acc_witness_zk(cv: Curve, blinded_witness: FrVec,
                              hp_witness: tuple[frx.Array, frx.Array, frx.Array],
                              sigmas: frx.Array) -> bytes:
    """`AccumulatorWitness` CanonicalSerialize (zk): `r1cs_blinded_witness`, the
    HP witness (with `Some` randomness), then `Some` accumulator-witness
    randomness (`sigma_a, sigma_b, sigma_c`). `sigmas` is a length-3 `cv.fr`
    array; `blinded_witness` a `cv.fr` array or int list."""
    out = curve.serialize_fr_vec(cv, blinded_witness)
    out += hp_as.serialize_witness_zk(cv, hp_witness)
    out += b"\x01" + np.asarray(sigmas, dtype=cv.fr).tobytes()
    return out


def _serialize_proof_zk(cv: Curve, low: list[np.ndarray], high: list[np.ndarray],
                        hiding_comms: hp_as.Instance, r1cs_r_input: list[int],
                        comm_r: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bytes:
    """AS `Proof` CanonicalSerialize (zk): the HP proof (with hiding comms), then
    `Some` `ProofRandomness` (`r1cs_r_input`, `comm_r_a/b/c`)."""
    out = hp_as.serialize_proof_zk(cv, low, high, hiding_comms)
    out += b"\x01" + curve.serialize_fr_vec(cv, r1cs_r_input)
    out += b"".join(curve.point_to_bytes(cv, c) for c in comm_r)
    return out


def _acc_instance_fe(cv: Curve, r1cs_input_bytes: bytes, comms: frx.Array) -> frx.Array:
    """`AccumulatorInstance::to_sponge_field_elements`: the `r1cs_input` bytes, then
    the six SW-affine points `comm_a, comm_b, comm_c` and the `hp_instance`'s
    `comm_1, comm_2, comm_3` — and **no option flag** (unlike the input instance's
    `Some`-flagged 5 randomness commitments). `comms` is a pre-stacked `(6,)` affine
    array `[comm_a, comm_b, comm_c, hp_1, hp_2, hp_3]`."""
    return fnp.concatenate([
        fnp.asarray(absorbable.u8_batch_field_array(cv, r1cs_input_bytes)),
        absorbable.point_to_field_array_frx(cv, comms),
    ])


def _beta_challenges_frx(cv: Curve, params: Any, as_matrices_hash: bytes, input_inst_fe: frx.Array,
                         proof_rand_fe: frx.Array, acc_inst_fe: frx.Array | None = None,
                         num_challenges: int = 1) -> frx.Array:
    """`compute_beta_challenges` as frx: fork `AS-FOR-R1CS-NARK-2020`, absorb the
    matrices hash, the prior **accumulator** instances, the input instances, and the
    proof randomness (the `compute_beta_challenges` absorb order), then squeeze
    `num_challenges = num_addends − 1` truncated-128 challenges → `beta =
    [1, c₁, …]`. The sponge is built in-trace (closing over `params` + the host
    `as_matrices_hash`), so the whole beta derivation threads into the prove trace.

    `acc_inst_fe` is the accumulator-instance field-element encoding
    (`_acc_instance_fe`), absorbed **before** the input — `None` (the default, with
    `num_challenges=1`) is the single-input init path (num_addends=2); the IVC fold
    passes one accumulator and `num_challenges=2` (num_addends=3)."""
    fr_one = fnp.asarray(np.array([1], dtype=cv.fr))
    sp = absorbable.fork(cv, sponge.new_sponge(params), AS_PROTOCOL_NAME)
    sp = absorbable.absorb_bytes(cv, sp, as_matrices_hash)
    if acc_inst_fe is not None:
        sp = sp.absorb(acc_inst_fe)
    sp = sp.absorb(input_inst_fe)
    sp = sp.absorb(proof_rand_fe)
    sp, ch = sponge.squeeze_challenges_frx(sp, num_challenges, _CHALLENGE_BITS, cv)
    return fnp.concatenate([fr_one, ch])


def _build_zk_core(cv: Curve, a: nark.Matrix, b: nark.Matrix, c: nark.Matrix, r1cs_input: FrVec,
                   witness: FrVec, generators: list[np.ndarray], hiding: np.ndarray, params: Any,
                   nark_matrices_hash: bytes, as_matrices_hash: bytes, supported_num_elems: int,
                   nark_r: list[int], nark_blinders: tuple[int, int, int, int, int, int, int, int],
                   as_r1cs_r_input: int, as_r1cs_r_witness: int, as_rand: tuple[int, int, int],
                   hp_hiding_a: int, hp_hiding_b: int, hp_rand: tuple[int, int, int]) -> tuple:
    """Build the fused zk-prove `@frx.jit` core (the **general** prover).
    The circuit (`a`/`b`/`c`, the matrices hashes, `params` — none are frx pytrees)
    stays baked, but the **assignment + all replayed randomness are runtime inputs**:
    the core takes `(bases_h, id_pt)` plus the assignment / NARK / AS / HP randomness
    as device arrays, so one lowered core proves any prove (not a single baked
    fixture). The two `u8_batch` Fiat-Shamir absorbs (`r1cs_input` for gamma + the
    AS instance, `r1cs_r_input` for the AS proof randomness) are fed **pre-encoded**
    as fq runtime arrays — the in-trace `fr→u8` rechunk the xla GPU plugin mis-lowers
    is done consumer-side (see `absorbable.point_to_field_array_frx`).

    The host `r1cs_input` / `witness` / `nark_*` / `as_*` / `hp_*` args supply the
    **example** runtime arrays for lowering (the lowered core is seed-independent);
    `prove_zk` runs + materializes, `export/export_prove.py` lowers ONE
    `prove_zk_general.mlirbc`. Returns `(core_fn, bases_h, id_pt, <12 runtime
    arrays>)`: the affine inputs then the runtime fr arrays (`in, wit, r, blinders,
    r_in, r_wit, as_rand, hp_rand`) then the two fq `u8_batch` arrays — the order
    the consumer feeds them in (`src/fused.rs`)."""
    # AS proof randomness: r1cs_r_input / r1cs_r_witness are `vec![rand; n]` (one
    # sampled value, cloned). Host prep — the cloned vectors + their canonical-LE
    # bytes (the only host work; everything else is the one fused trace below).
    input_len, witness_len = len(r1cs_input), len(witness)
    r1cs_r_input = [as_r1cs_r_input] * input_len
    r1cs_r_witness = [as_r1cs_r_witness] * witness_len
    r1cs_input_bytes = b"".join(cv.fr(v).tobytes() for v in r1cs_input)
    r1cs_r_input_bytes = b"".join(cv.fr(v).tobytes() for v in r1cs_r_input)

    # The committer key is the prove's affine input — every commitment uses
    # `bases_h = generators[:rows] ‖ hiding` (rows = num_constraints; the
    # product-poly commits use `bases_h[:rows]`), and the HP placeholder uses the
    # identity. Affine-typed jit constants don't lower, so these enter the fused
    # trace as arguments (the export-correct shape: the key is a runtime input).
    rows = len(a)
    n = input_len + witness_len  # z length = the circuit's num_vars (static)
    # Dense matrices for the AS-level `M·v` reduces: the assignment is a runtime
    # input on the general core, and `field.sparse_matvec`'s scatter-free CSR path
    # needs host-constant row bounds — which a runtime vector doesn't give. A dense
    # matvec (constant matrix · runtime vector) sidesteps that (the no-zk general
    # core's approach); densifying is affordable at the general prover's scale.
    a_dense, b_dense, c_dense = (fnp.asarray(nark.to_dense(cv, m, n)) for m in (a, b, c))
    bases_h = curve.stack_affine(cv, list(generators[:rows]) + [hiding])
    id_pt = curve.stack_affine(cv, [cv.g1((0, 0))])

    # The example assignment + randomness arrays carry the runtime shapes for
    # lowering (`in`/`wit`/`r` fr; `blinders` (8,); `r_in`/`r_wit` fr; `as_rand`
    # (3,); `hp_rand` (5,) = [hiding_a, hiding_b, hr1, hr2, hr3]; the two fq
    # `u8_batch` packings). The lowered core is seed-independent.
    ex_in = fnp.asarray(np.array(r1cs_input, dtype=cv.fr))
    ex_wit = fnp.asarray(np.array(witness, dtype=cv.fr))
    ex_r = fnp.asarray(np.array(nark_r, dtype=cv.fr))
    ex_blinders = fnp.asarray(np.array(list(nark_blinders), dtype=cv.fr))
    ex_r_in = fnp.asarray(np.array(r1cs_r_input, dtype=cv.fr))
    ex_r_wit = fnp.asarray(np.array(r1cs_r_witness, dtype=cv.fr))
    ex_as_rand = fnp.asarray(np.array(list(as_rand), dtype=cv.fr))
    ex_hp_rand = fnp.asarray(np.array([hp_hiding_a, hp_hiding_b, *hp_rand], dtype=cv.fr))
    ex_in_u8b = fnp.asarray(absorbable.u8_batch_field_array(cv, r1cs_input_bytes))
    ex_r_in_u8b = fnp.asarray(absorbable.u8_batch_field_array(cv, r1cs_r_input_bytes))

    @frx.jit
    def _core(bases_h: frx.Array, id_pt: frx.Array, in_arr: frx.Array, wit_arr: frx.Array,
              r_arr: frx.Array, blinders_arr: frx.Array, r_in_arr: frx.Array, r_wit_arr: frx.Array,
              as_rand_arr: frx.Array, hp_rand_arr: frx.Array, in_u8b: frx.Array,
              r_in_u8b: frx.Array) -> tuple:
        fr_one = fnp.asarray(np.array([1], dtype=cv.fr))
        # The assignment + NARK randomness ride in as runtime arrays via `rt`; the
        # host `r1cs_input`/`witness`/`nark_*` args (closed over) are the unused
        # example values, overridden here so one lowered core proves any prove.
        nk = nark.prove_zk_core(cv, a, b, c, r1cs_input, witness, bases_h, params,
                                nark_matrices_hash, nark_r, *nark_blinders,
                                rt=nark.NarkZkRuntime(in_arr, wit_arr, r_arr, blinders_arr, in_u8b))
        gamma = nk.gamma  # (1,)

        def _mz(dense_m: frx.Array, vec: frx.Array) -> frx.Array:
            """`M·vec` over fr as a dense matvec (constant matrix · runtime vector →
            no GPU scatter), threaded straight into the commitments / HP opening
            with no host round-trip. See the `a_dense` note above for why the
            general core densifies instead of reducing the sparse CSR."""
            return field.matvec(dense_m, vec)

        # comm_r_M = commit(M·(r1cs_r_input ‖ r1cs_r_witness)).
        zr = fnp.concatenate([r_in_arr, r_wit_arr])
        comm_r_a = curve.commit_hiding(cv, _mz(a_dense, zr), as_rand_arr[0], bases_h)
        comm_r_b = curve.commit_hiding(cv, _mz(b_dense, zr), as_rand_arr[1], bases_h)
        comm_r_c = curve.commit_hiding(cv, _mz(c_dense, zr), as_rand_arr[2], bases_h)

        # Blinded commitments: fold the NARK first-round randomness in, scaled by
        # gamma (each `comm + coeff·comm_r` is one lax.msm fold).
        one_gamma = fnp.concatenate([fr_one, gamma])
        blinded_comm_a = lax.msm(one_gamma, fnp.stack([nk.comm_a, nk.comm_r_a]))
        blinded_comm_b = lax.msm(one_gamma, fnp.stack([nk.comm_b, nk.comm_r_b]))
        blinded_comm_c = lax.msm(one_gamma, fnp.stack([nk.comm_c, nk.comm_r_c]))
        comm_prod = lax.msm(fnp.concatenate([fr_one, gamma, gamma * gamma]),
                               fnp.stack([nk.comm_c, nk.comm_1, nk.comm_2]))

        # HP input from the blinded commitments + the NARK opening; the HP zk core
        # builds its own `AS-FOR-HP-2020` fork in-trace. `hp_rand_arr` lifts the HP
        # hiding values + randomizers to runtime (the host hp_* args are unused).
        zw = fnp.concatenate([in_arr, nk.blinded_witness])
        hp_sponge = absorbable.fork(cv, sponge.new_sponge(params), HP_AS_PROTOCOL_NAME)
        hp_core = hp_as.prove_zk_core(
            cv, bases_h, id_pt, fnp.stack([blinded_comm_a, blinded_comm_b, comm_prod]),
            _mz(a_dense, zw), _mz(b_dense, zw), fnp.stack([nk.sigma_abc[0], nk.sigma_abc[1], nk.sigma_o]),
            supported_num_elems, params, hp_hiding_a, hp_hiding_b, hp_rand[0], hp_rand[1],
            hp_rand[2], base_sponge=hp_sponge, hp_rand=hp_rand_arr)

        # beta challenges (num_addends=2): the as_sponge absorb of the input
        # instance + proof randomness, packed straight from the frx commitments.
        # The two `u8_batch` packings (`r1cs_input` / `r1cs_r_input`) are runtime
        # fq inputs (pre-encoded consumer-side, not rechunked in-trace).
        inst_fe = fnp.concatenate([
            in_u8b,
            absorbable.point_to_field_array_frx(cv, fnp.stack([nk.comm_a, nk.comm_b, nk.comm_c])),
            fnp.asarray(absorbable.option_flag(cv, True)),
            absorbable.point_to_field_array_frx(
                cv, fnp.stack([nk.comm_r_a, nk.comm_r_b, nk.comm_r_c, nk.comm_1, nk.comm_2])),
        ])
        pr_fe = fnp.concatenate([
            fnp.asarray(absorbable.option_flag(cv, True)),
            r_in_u8b,
            absorbable.point_to_field_array_frx(cv, fnp.stack([comm_r_a, comm_r_b, comm_r_c])),
        ])
        beta = _beta_challenges_frx(cv, params, as_matrices_hash, inst_fe, pr_fe)  # (2,) = [1, c]

        # Fold the input + proof randomness under beta, all on-device.
        combined_input = field.combine_vectors(fnp.stack([in_arr, r_in_arr]), beta)
        combined_comm_a = lax.msm(beta, fnp.stack([blinded_comm_a, comm_r_a]))
        combined_comm_b = lax.msm(beta, fnp.stack([blinded_comm_b, comm_r_b]))
        combined_comm_c = lax.msm(beta, fnp.stack([blinded_comm_c, comm_r_c]))
        combined_blinded_witness = field.combine_vectors(
            fnp.stack([nk.blinded_witness, r_wit_arr]), beta)
        # combined sigma_M = sigma_M·beta[0] + as_r_M·beta[1] (both addends Some).
        combined_sigmas = nk.sigma_abc * beta[0] + as_rand_arr * beta[1]

        return (combined_input, combined_comm_a, combined_comm_b, combined_comm_c,
                combined_blinded_witness, combined_sigmas, comm_r_a, comm_r_b, comm_r_c, hp_core)

    return (_core, bases_h, id_pt, ex_in, ex_wit, ex_r, ex_blinders, ex_r_in, ex_r_wit,
            ex_as_rand, ex_hp_rand, ex_in_u8b, ex_r_in_u8b)


def prove_zk(cv: Curve, a: nark.Matrix, b: nark.Matrix, c: nark.Matrix, r1cs_input: FrVec,
             witness: FrVec, generators: list[np.ndarray], hiding: np.ndarray, params: Any,
             nark_matrices_hash: bytes, as_matrices_hash: bytes, supported_num_elems: int,
             nark_r: list[int], nark_blinders: tuple[int, int, int, int, int, int, int, int],
             as_r1cs_r_input: int, as_r1cs_r_witness: int, as_rand: tuple[int, int, int],
             hp_hiding_a: int, hp_hiding_b: int, hp_rand: tuple[int, int, int]
             ) -> tuple[bytes, bytes, bytes]:
    """zk `ASForR1CSNark::prove` over a single input, no prior accumulators — the
    zk acceptance criterion. Replays every sampled randomness value (NARK, AS, HP)
    rather than re-deriving arkworks' RNG. The whole prove is one fused `@frx.jit`
    core (`_build_zk_core`); materialization (`np.asarray` to host `fr` arrays) is
    the serialize seam below."""
    return _accumulate(
        cv, a, b, c, r1cs_input, witness, generators, hiding, params, nark_matrices_hash,
        as_matrices_hash, supported_num_elems,
        Randomness(nark_r, nark_blinders, as_r1cs_r_input, as_r1cs_r_witness, as_rand,
                   hp_hiding_a, hp_hiding_b, hp_rand))[1:]


def _accumulate(cv: Curve, a: nark.Matrix, b: nark.Matrix, c: nark.Matrix,
                r1cs_input: FrVec, witness: FrVec, generators: list[np.ndarray],
                hiding: np.ndarray, params: Any, nark_matrices_hash: bytes,
                as_matrices_hash: bytes, supported_num_elems: int, rnd: Randomness
                ) -> tuple[FoldedAccumulator, bytes, bytes, bytes]:
    """The zk single-input prove, structured + serialized. `prove_zk` is this with
    the randomness spelled out (the arkworks replay) and the accumulator dropped;
    `accumulate` is this with the hashes derived and the randomness sampled."""
    r1cs_r_input = [rnd.as_r1cs_r_input] * len(r1cs_input)
    (core_fn, bases_h, id_pt, ex_in, ex_wit, ex_r, ex_blinders, ex_r_in, ex_r_wit,
     ex_as_rand, ex_hp_rand, ex_in_u8b, ex_r_in_u8b) = _build_zk_core(
        cv, a, b, c, r1cs_input, witness, generators, hiding, params, nark_matrices_hash,
        as_matrices_hash, supported_num_elems, *rnd)
    core_out = core_fn(bases_h, id_pt, ex_in, ex_wit, ex_r, ex_blinders, ex_r_in, ex_r_wit,
                       ex_as_rand, ex_hp_rand, ex_in_u8b, ex_r_in_u8b)
    return _materialize_zk(cv, core_out, r1cs_r_input)


# --- decide (the accumulation decider, no-zk + zk) ---------------------------


class Accumulator(NamedTuple):
    """The decider's view of an accumulator — the `(instance, witness)` fields
    `ASForR1CSNark::decide` + `ASForHadamardProducts::decide` read. `sigmas` /
    `hp_rand` are the zk Pedersen randomizers (`None` on the no-zk path, where the
    commitments are non-hiding)."""
    r1cs_input: FrVec            # instance.r1cs_input
    blinded_witness: FrVec       # witness.r1cs_blinded_witness
    hp_a_vec: FrVec              # witness.hp_witness.a_vec
    hp_b_vec: FrVec              # witness.hp_witness.b_vec
    sigmas: tuple[int, int, int] | None       # witness.randomness.{sigma_a,sigma_b,sigma_c}
    hp_rand: tuple[int, int, int] | None       # witness.hp_witness.randomness.{rand_1,rand_2,rand_3}


def decide(cv: Curve, a: nark.Matrix, b: nark.Matrix, c: nark.Matrix,
           generators: list[np.ndarray], hiding: np.ndarray | None,
           acc: Accumulator) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray, np.ndarray]:
    """`ASForR1CSNark::decide` recomputed (CPU group-reduction oracle, the fused
    GPU core's target — the byte-match counterpart of :func:`ipa_pc_as.decide_final_key`).

    Steps (BCLMS20), over `z = r1cs_input ‖ r1cs_blinded_witness`:

    1. `comm_M = commit(M·z, σ_M)` for `M ∈ {A, B, C}` — the three size-`n`
       Pedersen commitments (`σ_M` the zk randomizer, `None` ⇒ non-hiding).
    2. the `hp_as` decide check: with the HP witness `(a_vec, b_vec)` and
       `product = a_vec ∘ b_vec`, recompute `test_comm_{1,2,3} =
       commit(a_vec, ρ₁), commit(b_vec, ρ₂), commit(product, ρ₃)`.

    The decider accepts iff these six commitments equal the accumulator's stored
    `comm_{a,b,c}` and `hp_instance.comm_{1,2,3}`; they are returned here so the
    byte-match (and the GPU core) can compare against the golden bytes. The MSMs
    reuse the prove core's `commit` machinery (`curve.pedersen_commit`), per #10."""
    z_a = nark.matrix_vec_mul(cv, a, acc.r1cs_input, acc.blinded_witness)
    z_b = nark.matrix_vec_mul(cv, b, acc.r1cs_input, acc.blinded_witness)
    z_c = nark.matrix_vec_mul(cv, c, acc.r1cs_input, acc.blinded_witness)
    sigma_a, sigma_b, sigma_c = acc.sigmas if acc.sigmas is not None else (None, None, None)
    comm_a = curve.pedersen_commit(cv, generators[:len(z_a)], z_a, hiding, sigma_a)
    comm_b = curve.pedersen_commit(cv, generators[:len(z_b)], z_b, hiding, sigma_b)
    comm_c = curve.pedersen_commit(cv, generators[:len(z_c)], z_c, hiding, sigma_c)

    # HP decide check: product is the element-wise (Hadamard) a_vec ∘ b_vec. This
    # is the host oracle, so it uses a vectorized numpy `fr` multiply (the `fr`
    # dtype reduces mod r) — host arithmetic, not frx, and vectorized rather than
    # a per-element scalar multiply (which would dispatch to frx once per element).
    product = np.array(acc.hp_a_vec, dtype=cv.fr) * np.array(acc.hp_b_vec, dtype=cv.fr)
    rand_1, rand_2, rand_3 = acc.hp_rand if acc.hp_rand is not None else (None, None, None)
    hp_comm_1 = curve.pedersen_commit(cv, generators[:len(acc.hp_a_vec)], acc.hp_a_vec, hiding, rand_1)
    hp_comm_2 = curve.pedersen_commit(cv, generators[:len(acc.hp_b_vec)], acc.hp_b_vec, hiding, rand_2)
    hp_comm_3 = curve.pedersen_commit(cv, generators[:len(product)], product, hiding, rand_3)
    return comm_a, comm_b, comm_c, hp_comm_1, hp_comm_2, hp_comm_3


# --- zk fold (one input folded into one prior accumulator, num_addends=3) ----


def _build_zk_fold_core(cv: Curve, a: nark.Matrix, b: nark.Matrix, c: nark.Matrix, r1cs_input: FrVec,
                        witness: FrVec, generators: list[np.ndarray], hiding: np.ndarray, params: Any,
                        nark_matrices_hash: bytes, as_matrices_hash: bytes, supported_num_elems: int,
                        nark_r: list[int], nark_blinders: tuple[int, int, int, int, int, int, int, int],
                        as_r1cs_r_input: int, as_r1cs_r_witness: int, as_rand: tuple[int, int, int],
                        hp_hiding_a: int, hp_hiding_b: int, hp_rand: tuple[int, int, int],
                        acc_r1cs_input: FrVec, acc_comms: list[np.ndarray], acc_blinded_witness: FrVec,
                        acc_sigma_abc: tuple[int, int, int], acc_hp_a_vec: FrVec,
                        acc_hp_b_vec: FrVec, acc_hp_rand: tuple[int, int, int]) -> tuple:
    """Build the fused zk **fold** `@frx.jit` core (closing over the host constants:
    `params`, the matrices hashes, both inputs' fr components, and replayed
    randomness — none are frx pytrees) plus its three affine arguments
    `(bases_h, id_pt, acc_comms)`: the committer key `generators[:rows] ‖ hiding`, the
    HP placeholder identity, and the old accumulator's `(6,)` commitments
    `[comm_a, comm_b, comm_c, hp_1, hp_2, hp_3]`. The full IVC fold step
    (`num_addends = 3`, `beta = [1, c₁, c₂]`): the core re-derives input's NARK + the
    fold's AS/HP commitments, runs the multi-addend AS-level fold over
    `[acc, input, proof_randomness]`, and the HP-level fold of input's HP input INTO
    the old accumulator's HP input. `prove_zk_fold` runs + materializes it;
    `export/export_fold_zk.py` lowers it to one `.mlirbc`. Returns
    `(core_fn, bases_h, id_pt, acc_comms)`."""
    input_len, witness_len = len(r1cs_input), len(witness)
    r1cs_r_input = [as_r1cs_r_input] * input_len
    r1cs_r_witness = [as_r1cs_r_witness] * witness_len
    as_r1, as_r2, as_r3 = as_rand
    input_bytes = b"".join(cv.fr(v).tobytes() for v in r1cs_input)
    r1cs_r_input_bytes = b"".join(cv.fr(v).tobytes() for v in r1cs_r_input)
    acc_input_bytes = b"".join(cv.fr(v).tobytes() for v in acc_r1cs_input)

    rows = len(a)
    bases_h = curve.stack_affine(cv, list(generators[:rows]) + [hiding])
    id_pt = curve.stack_affine(cv, [cv.g1((0, 0))])
    acc_comms_arr = curve.stack_affine(cv, list(acc_comms))  # (6,)

    # The fold's AS-level `M·v` reduces are all over BAKED vectors (the circuit + the
    # replayed randomness are fixed at export time), so they are computed HOST-SIDE
    # and baked as constants. `matrix_vec_mul` is the host sparse reduce (no 15 GB
    # densify). This is orthogonal to the NARK-internal `M·z` / `M·zr` reduces inside
    # `prove_zk_core`, which DO run on-device — `field.sparse_matvec` keeps those
    # scatter-free (a CSR prefix sum) so they lower to a parallel reduce. Baking the
    # AS reduces here is a data-movement choice (the vectors are constants anyway),
    # not a scatter workaround; the on-device sparse matvec is exercised both by the
    # standalone NARK half-step and by those NARK-internal fold reduces.
    def _host_mz(m: nark.Matrix, inp: FrVec, wit: FrVec) -> frx.Array:
        return fnp.asarray(nark.matrix_vec_mul(cv, m, inp, wit))

    # comm_r = commit(M·(r1cs_r_input ‖ r1cs_r_witness)); the AS proof-randomness reduce.
    mz_r = [_host_mz(m, r1cs_r_input, r1cs_r_witness) for m in (a, b, c)]
    # HP openings A·zw / B·zw with zw = z + gamma·zr: the two baked reduces M·z
    # (z = r1cs_input ‖ witness) and M·zr (zr = 0_input ‖ nark_r), gamma-combined
    # in-trace below.
    hp_mz = [_host_mz(m, r1cs_input, witness) for m in (a, b)]
    hp_mzr = [_host_mz(m, [0] * input_len, nark_r) for m in (a, b)]

    # keep_unused: at num_addends=3 the old accumulator replaces the HP placeholder,
    # so `id_pt` is dead in the fold trace; without this frx DCE drops it and the
    # lowered core has 2 args, mismatching the consumer's 3 (bases_h, id_pt, acc_comms).
    @functools.partial(frx.jit, keep_unused=True)
    def _core(bases_h: frx.Array, id_pt: frx.Array, acc_comms: frx.Array) -> tuple:
        fr_one = fnp.asarray(np.array([1], dtype=cv.fr))
        nk = nark.prove_zk_core(cv, a, b, c, r1cs_input, witness, bases_h, params,
                                nark_matrices_hash, nark_r, *nark_blinders)
        gamma = nk.gamma

        # The fold's AS proof-randomness commitments comm_r_M = commit(M·z_r, as_r_M)
        # (M·z_r pre-baked host-side — `mz_r` above).
        comm_r_a = curve.commit_hiding(cv, mz_r[0], as_r1, bases_h)
        comm_r_b = curve.commit_hiding(cv, mz_r[1], as_r2, bases_h)
        comm_r_c = curve.commit_hiding(cv, mz_r[2], as_r3, bases_h)

        # input's gamma-blinded NARK commitments + the HP comm_prod (gamma² term).
        one_gamma = fnp.concatenate([fr_one, gamma])
        blinded_comm_a = lax.msm(one_gamma, fnp.stack([nk.comm_a, nk.comm_r_a]))
        blinded_comm_b = lax.msm(one_gamma, fnp.stack([nk.comm_b, nk.comm_r_b]))
        blinded_comm_c = lax.msm(one_gamma, fnp.stack([nk.comm_c, nk.comm_r_c]))
        comm_prod = lax.msm(fnp.concatenate([fr_one, gamma, gamma * gamma]),
                               fnp.stack([nk.comm_c, nk.comm_1, nk.comm_2]))

        # HP-level fold: input's HP input (blinded comms + M·z openings, NARK
        # randomness) folded INTO the old accumulator's HP input. The HP openings are
        # A·zw, B·zw for zw = r1cs_input ‖ blinded_witness = z + gamma·zr (since
        # blinded_witness = witness + gamma·r, nark `_prove_zk_segment`); by linearity
        # M·zw = M·z + gamma·M·zr, with M·z (`hp_mz`) and M·zr (`hp_mzr`) the two
        # pre-baked reduces and gamma the only runtime term.
        hp_a_open = hp_mz[0] + gamma * hp_mzr[0]
        hp_b_open = hp_mz[1] + gamma * hp_mzr[1]
        new_hp_rand = fnp.stack([nk.sigma_abc[0], nk.sigma_abc[1], nk.sigma_o])
        old_hp_comms = acc_comms[3:6]
        old_hp_rand = fnp.asarray(np.array(list(acc_hp_rand), dtype=cv.fr))
        hp_sponge = absorbable.fork(cv, sponge.new_sponge(params), HP_AS_PROTOCOL_NAME)
        hp_core = hp_as.prove_zk_core(
            cv, bases_h, id_pt, fnp.stack([blinded_comm_a, blinded_comm_b, comm_prod]),
            hp_a_open, hp_b_open, new_hp_rand, supported_num_elems, params,
            hp_hiding_a, hp_hiding_b, hp_rand[0], hp_rand[1], hp_rand[2],
            old_inst=old_hp_comms,
            old_a=fnp.asarray(np.array(acc_hp_a_vec, dtype=cv.fr)),
            old_b=fnp.asarray(np.array(acc_hp_b_vec, dtype=cv.fr)),
            old_rand=old_hp_rand, base_sponge=hp_sponge)

        # beta over num_addends=3: as_sponge absorbs the accumulator instance, then
        # the input instance, then the proof randomness; squeeze 2 challenges.
        acc_inst_fe = _acc_instance_fe(cv, acc_input_bytes, acc_comms)
        inst_fe = fnp.concatenate([
            fnp.asarray(absorbable.u8_batch_field_array(cv, input_bytes)),
            absorbable.point_to_field_array_frx(cv, fnp.stack([nk.comm_a, nk.comm_b, nk.comm_c])),
            fnp.asarray(absorbable.option_flag(cv, True)),
            absorbable.point_to_field_array_frx(
                cv, fnp.stack([nk.comm_r_a, nk.comm_r_b, nk.comm_r_c, nk.comm_1, nk.comm_2])),
        ])
        pr_fe = fnp.concatenate([
            fnp.asarray(absorbable.option_flag(cv, True)),
            fnp.asarray(absorbable.u8_batch_field_array(cv, r1cs_r_input_bytes)),
            absorbable.point_to_field_array_frx(cv, fnp.stack([comm_r_a, comm_r_b, comm_r_c])),
        ])
        beta = _beta_challenges_frx(
            cv, params, as_matrices_hash, inst_fe, pr_fe, acc_inst_fe=acc_inst_fe, num_challenges=2)

        # AS-level fold under beta, order [acc, input, proof_randomness].
        combined_input = field.combine_vectors(
            fnp.asarray(np.array([acc_r1cs_input, r1cs_input, r1cs_r_input], dtype=cv.fr)), beta)
        cca = lax.msm(beta, fnp.stack([acc_comms[0], blinded_comm_a, comm_r_a]))
        ccb = lax.msm(beta, fnp.stack([acc_comms[1], blinded_comm_b, comm_r_b]))
        ccc = lax.msm(beta, fnp.stack([acc_comms[2], blinded_comm_c, comm_r_c]))
        combined_blinded_witness = field.combine_vectors(fnp.stack([
            fnp.asarray(np.array(acc_blinded_witness, dtype=cv.fr)), nk.blinded_witness,
            fnp.asarray(np.array(r1cs_r_witness, dtype=cv.fr))]), beta)
        combined_sigmas = (fnp.asarray(np.array(list(acc_sigma_abc), dtype=cv.fr)) * beta[0]
                           + nk.sigma_abc * beta[1]
                           + fnp.asarray(np.array([as_r1, as_r2, as_r3], dtype=cv.fr)) * beta[2])

        return (combined_input, cca, ccb, ccc, combined_blinded_witness, combined_sigmas,
                comm_r_a, comm_r_b, comm_r_c, hp_core)

    return _core, bases_h, id_pt, acc_comms_arr


def prove_zk_fold(cv: Curve, a: nark.Matrix, b: nark.Matrix, c: nark.Matrix, r1cs_input: FrVec,
                  witness: FrVec, generators: list[np.ndarray], hiding: np.ndarray, params: Any,
                  nark_matrices_hash: bytes, as_matrices_hash: bytes, supported_num_elems: int,
                  nark_r: list[int], nark_blinders: tuple[int, int, int, int, int, int, int, int],
                  as_r1cs_r_input: int, as_r1cs_r_witness: int, as_rand: tuple[int, int, int],
                  hp_hiding_a: int, hp_hiding_b: int, hp_rand: tuple[int, int, int],
                  acc_r1cs_input: FrVec, acc_comms: list[np.ndarray], acc_blinded_witness: FrVec,
                  acc_sigma_abc: tuple[int, int, int], acc_hp_a_vec: FrVec,
                  acc_hp_b_vec: FrVec, acc_hp_rand: tuple[int, int, int]) -> tuple[bytes, bytes, bytes]:
    """zk `ASForR1CSNark::prove` folding one input INTO one prior accumulator — the
    full IVC fold step (`num_addends = 3`, `beta = [1, c₁, c₂]`). Runs the fused fold
    core (`_build_zk_fold_core`) over its three affine inputs, then materializes +
    serializes the folded accumulator. Returns the serialized
    `(acc_instance, acc_witness, proof)`."""
    return _fold(
        cv, a, b, c, r1cs_input, witness, generators, hiding, params, nark_matrices_hash,
        as_matrices_hash, supported_num_elems,
        Randomness(nark_r, nark_blinders, as_r1cs_r_input, as_r1cs_r_witness, as_rand,
                   hp_hiding_a, hp_hiding_b, hp_rand),
        FoldedAccumulator(acc_r1cs_input, tuple(acc_comms), acc_blinded_witness,
                          acc_sigma_abc, acc_hp_a_vec, acc_hp_b_vec, acc_hp_rand))[1:]


def _fold(cv: Curve, a: nark.Matrix, b: nark.Matrix, c: nark.Matrix, r1cs_input: FrVec,
          witness: FrVec, generators: list[np.ndarray], hiding: np.ndarray, params: Any,
          nark_matrices_hash: bytes, as_matrices_hash: bytes, supported_num_elems: int,
          rnd: Randomness, acc: FoldedAccumulator
          ) -> tuple[FoldedAccumulator, bytes, bytes, bytes]:
    """The zk fold, structured + serialized — the `_accumulate` counterpart."""
    r1cs_r_input = [rnd.as_r1cs_r_input] * len(r1cs_input)
    core_fn, bases_h, id_pt, acc_comms_arr = _build_zk_fold_core(
        cv, a, b, c, r1cs_input, witness, generators, hiding, params, nark_matrices_hash,
        as_matrices_hash, supported_num_elems, *rnd, acc.r1cs_input, list(acc.comms),
        acc.blinded_witness, acc.sigma_abc, acc.hp_a_vec, acc.hp_b_vec, acc.hp_rand)
    core_out = core_fn(bases_h, id_pt, acc_comms_arr)
    return _materialize_zk(cv, core_out, r1cs_r_input)


# --- the user-facing pair: accumulate, then fold into what it returned --------


def _matrices_hashes(cv: Curve, a: nark.Matrix, b: nark.Matrix,
                     c: nark.Matrix) -> tuple[bytes, bytes]:
    """The NARK and AS `hash_matrices` digests. Both are a pure function of the
    circuit, so only a replay against a golden file needs to pass them in."""
    return (nark.hash_matrices(cv, nark.PROTOCOL_NAME, a, b, c),
            nark.hash_matrices(cv, AS_PROTOCOL_NAME, a, b, c))


def accumulate(cv: Curve, a: nark.Matrix, b: nark.Matrix, c: nark.Matrix,
               r1cs_input: FrVec, witness: FrVec, generators: list[np.ndarray],
               hiding: np.ndarray, params: Any, supported_num_elems: int,
               rnd: Randomness) -> tuple[FoldedAccumulator, bytes, bytes, bytes]:
    """zk `ASForR1CSNark::prove` over one input with no prior accumulator — the
    first step of a chain, returning an accumulator :func:`fold` takes back.

    :func:`prove_zk` is the same prove for the arkworks replay: it spells the
    matrices hashes out and hands back only the serialized form."""
    nark_h, as_h = _matrices_hashes(cv, a, b, c)
    return _accumulate(cv, a, b, c, r1cs_input, witness, generators, hiding, params,
                       nark_h, as_h, supported_num_elems, rnd)


def fold(cv: Curve, a: nark.Matrix, b: nark.Matrix, c: nark.Matrix, r1cs_input: FrVec,
         witness: FrVec, generators: list[np.ndarray], hiding: np.ndarray, params: Any,
         supported_num_elems: int, acc: FoldedAccumulator, rnd: Randomness
         ) -> tuple[FoldedAccumulator, bytes, bytes, bytes]:
    """zk `ASForR1CSNark::prove` folding one input INTO `acc` — the IVC step
    (`num_addends = 3`), returning an accumulator that can be folded again or
    decided. :func:`prove_zk_fold` is the replay counterpart."""
    nark_h, as_h = _matrices_hashes(cv, a, b, c)
    return _fold(cv, a, b, c, r1cs_input, witness, generators, hiding, params,
                 nark_h, as_h, supported_num_elems, rnd, acc)
