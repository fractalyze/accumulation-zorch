"""R1CS NARK prover challenge derivation (port of `r1cs_nark/mod.rs`).

So far this ports `R1CSNark::compute_challenge` — the gamma challenge — for the
no-zk path. The NARK prove itself (rounds, blinded witness) lands in slice 3.
"""

import hashlib
import struct
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from zorch.hash.duplex_sponge import DuplexSponge

from . import absorbable, curve, jcurve, jfield, jsponge, sponge
from .curve import Curve

# ark `r1cs_nark::PROTOCOL_NAME` — the domain the NARK sponge is forked with.
PROTOCOL_NAME = b"R1CS-NARK-2020"

# Challenge squeeze window (ark `r1cs_nark::CHALLENGE_SIZE`, capped at fr capacity).
# Both Pasta scalar fields are 254-cap > 128, so this is the curve-invariant 128.
_CHALLENGE_BITS = min(sponge.CHALLENGE_SIZE, sponge.FR_CAPACITY)

# A sparse `Matrix<Fr>` row: `(coeff, var_index)` pairs.
Matrix = list[list[tuple[int, int]]]


def _serialize_matrix(cv: Curve, m: Matrix) -> bytes:
    """CanonicalSerialize of a sparse `Matrix<Fr>` = `Vec<Vec<(Fr, usize)>>`:
    u64-LE length prefixes; each `(Fr, usize)` is the fr (32B LE) ‖ the index as
    a u64 LE."""
    out = bytearray(struct.pack("<Q", len(m)))
    for row in m:
        out += struct.pack("<Q", len(row))
        for coeff, idx in row:
            out += cv.fr(coeff).tobytes() + struct.pack("<Q", idx)
    return bytes(out)


def hash_matrices(cv: Curve, domain: bytes, a: Matrix, b: Matrix, c: Matrix) -> bytes:
    """ark `r1cs_nark::hash_matrices`: blake2b-256 over
    `domain ‖ a.serialize() ‖ b.serialize() ‖ c.serialize()` (the crate uses
    `VarBlake2b::new(32)`, i.e. BLAKE2b with a 32-byte digest)."""
    ser = bytearray(domain)
    for m in (a, b, c):
        ser += _serialize_matrix(cv, m)
    return hashlib.blake2b(bytes(ser), digest_size=32).digest()


def to_dense(cv: Curve, matrix: Matrix, num_vars: int) -> np.ndarray:
    """Densify a sparse `Matrix<Fr>` (rows of `(coeff, var_index)`) to a
    `(rows × num_vars)` fr array — the dense form `jcurve.commit_dense` reduces
    on-device (`M·z`). Host-side data prep; the DummyCircuit matrices are tiny,
    and jagged commitment is a Phase-2 perf concern."""
    dense = np.zeros((len(matrix), num_vars), dtype=cv.fr)
    for r, row in enumerate(matrix):
        for coeff, idx in row:
            dense[r, idx] = cv.fr(coeff)
    return dense


def to_coo(cv: Curve, matrix: Matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a sparse `Matrix<Fr>` (rows of `(coeff, var_index)`) to flat COO
    arrays `(row_idx, col_idx, vals)` — the layout `jfield.sparse_matvec` reduces
    on-device (`segment_sum` over the rows). The sparse analog of `to_dense`: it
    never materializes the `rows × vars` grid, so it scales to the recursion
    circuit where densifying would be ~15 GB. `vals` is `fr`; the indices are
    `int32`. Empty / all-zero rows simply emit no entries — `segment_sum`'s
    `num_segments = len(matrix)` still yields them as zero."""
    row_idx: list[int] = []
    col_idx: list[int] = []
    vals: list[int] = []
    for r, row in enumerate(matrix):
        for coeff, idx in row:
            row_idx.append(r)
            col_idx.append(idx)
            vals.append(coeff)
    return (np.array(row_idx, dtype=np.int32), np.array(col_idx, dtype=np.int32),
            np.array(vals, dtype=cv.fr))


def matrix_vec_mul(cv: Curve, matrix: Matrix, input: list[int], witness: list[int]) -> np.ndarray:
    """ark `matrix_vec_mul`: `matrix * (input ‖ witness)` in fr, as an `fr` array.
    `matrix` is a sparse `Vec<Vec<(coeff, var_index)>>`; `input`/`witness` are fr
    ints. The per-row inner product runs over `cv.fr` arrays, so the dtype reduces
    mod r — no manual `% fr_modulus`. Stays an `fr` array end-to-end (feeding
    `pedersen_commit` / the device `z`), never decoded back to a python int."""
    z = np.array(list(input) + list(witness), dtype=cv.fr)
    out = np.zeros(len(matrix), dtype=cv.fr)
    for i, row in enumerate(matrix):
        if not row:
            continue  # empty row → the `fr` zero already in `out`
        coeffs = np.array([coeff for coeff, _ in row], dtype=cv.fr)
        idxs = [idx for _, idx in row]
        out[i] = np.dot(coeffs, z[idxs])
    return out


def _serialize_proof(cv: Curve, comm_a: np.ndarray, comm_b: np.ndarray, comm_c: np.ndarray,
                     blinded_witness: list[int]) -> bytes:
    """ark `CanonicalSerialize` of a no-zk `Proof`: the three first-round
    commitments (compressed, 33B), the `None` first-round randomness flag, the
    blinded-witness `Vec<Fr>` (u64 length + 32B LE each), and the `None`
    second-round randomness flag."""
    out = (curve.point_to_bytes(cv, comm_a) + curve.point_to_bytes(cv, comm_b)
           + curve.point_to_bytes(cv, comm_c))
    out += b"\x00"  # FirstRoundMessage.randomness = None
    out += struct.pack("<Q", len(blinded_witness))
    out += b"".join(cv.fr(w).tobytes() for w in blinded_witness)
    out += b"\x00"  # SecondRoundMessage.randomness = None
    return out


class NoZkNarkCore(NamedTuple):
    """The no-zk NARK's three first-round commitments as on-device affine point
    arrays — the un-materialized form the fused export lowers. The no-zk
    `blinded_witness` is the raw witness (no `gamma·r`), so it is baked into
    serialization host-side and needn't ride on-device."""
    comm_a: jax.Array
    comm_b: jax.Array
    comm_c: jax.Array


def _coo_dev(coo: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A host `(row_idx, col_idx, vals)` COO triple → device `jnp` arrays."""
    row_idx, col_idx, vals = coo
    return jnp.asarray(row_idx), jnp.asarray(col_idx), jnp.asarray(vals)


def prove_no_zk_core(cv: Curve, coo_a: tuple, coo_b: tuple, coo_c: tuple, z: jax.Array,
                     bases: jax.Array, num_rows: int) -> NoZkNarkCore:
    """On-device no-zk NARK prove: commit `M·z` for M in {a,b,c} with one
    `lax.msm` each, the `M·z` reduced **in-trace** from the sparse COO
    (`jfield.sparse_matvec`) rather than host-side. Plain (un-decorated) so it
    inlines into the export's top-level `@jax.jit`. `coo_*` are
    `(row_idx, col_idx, vals)` device arrays; `z = input ‖ witness` the `(vars,)`
    Fr vector; `bases` the `(num_rows,)` generators (an affine jit argument — the
    committer key is a runtime input on the export path, and an affine constant
    doesn't lower). No blinders (no-zk), so a commitment is just `Σ (M·z)ᵢ·basesᵢ`."""
    def commit(coo: tuple) -> jax.Array:
        row_idx, col_idx, vals = coo
        return lax.msm(jfield.sparse_matvec(vals, col_idx, row_idx, z, num_rows), bases)
    return NoZkNarkCore(commit(coo_a), commit(coo_b), commit(coo_c))


def build_no_zk_core(cv: Curve, a: Matrix, b: Matrix, c: Matrix, input: list[int],
                     witness: list[int], generators: list[np.ndarray]) -> tuple:
    """Build the fused no-zk NARK core as `(core_fn, bases)`: a single `@jax.jit`
    closing over the circuit (the sparse COO matrices + `z = input ‖ witness`) with
    the committer key `bases` as its **sole runtime argument** — the export-correct
    shape (`core_fn(bases) -> NoZkNarkCore`). Shared by `prove_no_zk` (which
    runs + serializes it) and `export/export_nark.py` (which lowers it to one
    StableHLO module). Mirrors `r1cs_nark_as._build_zk_core` for the no-zk NARK."""
    rows = len(a)
    z = jnp.asarray(np.array(list(input) + list(witness), dtype=cv.fr))
    bases = jcurve.stack_affine(cv, list(generators))
    coo_a, coo_b, coo_c = (_coo_dev(to_coo(cv, m)) for m in (a, b, c))

    @jax.jit
    def core_fn(bases: jax.Array) -> NoZkNarkCore:
        return prove_no_zk_core(cv, coo_a, coo_b, coo_c, z, bases, rows)

    return core_fn, bases


def prove_no_zk(cv: Curve, a: Matrix, b: Matrix, c: Matrix, input: list[int],
                witness: list[int], generators: list[np.ndarray]) -> bytes:
    """ark `R1CSNark::prove` (no-zk) as a single fused `@jax.jit` trace — the
    standalone no-zk NARK prove. The `M·z` reduce runs **on-device** from the
    sparse COO (`prove_no_zk_core`), so the whole prove is the trace the GPU
    export lowers; only the committer key (`bases`) is a runtime jit argument, the
    circuit / witness baked in as constants. Materialization + serialization
    (`_serialize_proof`) is the host seam outside the trace. Byte-identical to the
    crate's `R1CSNark::prove` no-zk proof."""
    core_fn, bases = build_no_zk_core(cv, a, b, c, input, witness, generators)
    core = core_fn(bases)
    return _serialize_proof(cv, np.asarray(core.comm_a), np.asarray(core.comm_b),
                            np.asarray(core.comm_c), witness)


class NarkZkProof(NamedTuple):
    """The structured zk NARK `Proof`: the blinded first-round commitments + the
    sigma-protocol randomness commitments, the gamma-blinded witness, and the
    response sigmas. `gamma` is retained for the AS path (which re-derives it)."""
    comm_a: np.ndarray
    comm_b: np.ndarray
    comm_c: np.ndarray
    comm_r_a: np.ndarray
    comm_r_b: np.ndarray
    comm_r_c: np.ndarray
    comm_1: np.ndarray
    comm_2: np.ndarray
    blinded_witness: np.ndarray  # (witness_len,) fr
    sigma_a: Any                 # response sigma fr scalars
    sigma_b: Any
    sigma_c: Any
    sigma_o: Any
    gamma: Any                   # retained fr scalar (the AS path re-derives it)


class NarkZkCore(NamedTuple):
    """`prove_zk`'s outputs as on-device jax arrays — the un-materialized form of
    `NarkZkProof`. The R1CS-NARK-AS path threads these straight on (no host hop)
    so the whole zk prove stays one trace; `prove_zk` materializes them for the
    standalone NARK proof. Each commitment is a single affine point array."""
    comm_a: jax.Array
    comm_b: jax.Array
    comm_c: jax.Array
    comm_r_a: jax.Array
    comm_r_b: jax.Array
    comm_r_c: jax.Array
    comm_1: jax.Array
    comm_2: jax.Array
    blinded_witness: jax.Array  # (witness_len,) Fr
    sigma_abc: jax.Array        # (3,) Fr
    sigma_o: jax.Array          # () Fr
    gamma: jax.Array            # (1,) Fr


class NarkZkRuntime(NamedTuple):
    """The zk NARK prover's per-prove inputs lifted to **runtime** device arrays
    (the general prover): the assignment (`r1cs_input` / `witness`), the
    witness-blinder vector `r`, the 8 sigma blinders `[a, b, c, r_a, r_b, r_c, 1,
    2]`, and the public input's `u8_batch` field-element packing (`input_u8b`, an
    fq array). When `prove_zk_core` receives this it bakes nothing — one lowered
    core proves any (witness, randomness). `input_u8b` is pre-encoded by the
    consumer because the in-trace `fr→u8` rechunk the gamma sponge needs is
    mis-lowered by the zkx GPU plugin (see `absorbable.point_to_field_array_jax`).
    Omitting it (the half-step / fold) keeps the host-baked path."""
    r1cs_input: jax.Array  # (input_len,) fr
    witness: jax.Array     # (witness_len,) fr
    r: jax.Array           # (witness_len,) fr — the NARK witness blinders
    blinders: jax.Array    # (8,) fr — [a, b, c, r_a, r_b, r_c, 1, 2]
    input_u8b: jax.Array   # (·,) fq — u8_batch(r1cs_input) for the gamma sponge


def _prove_zk_field_prep(cv: Curve, a: Matrix, b: Matrix, c: Matrix, input: list[int],
                         witness: list[int], r: list[int], a_blinder: int, b_blinder: int,
                         c_blinder: int, r_a_blinder: int, r_b_blinder: int, r_c_blinder: int,
                         blinder_1: int, blinder_2: int, rt: NarkZkRuntime | None = None) -> tuple:
    """Field operand prep for the zk NARK prove (no group ops): the sparse COO of
    each matrix (`row_idx, col_idx, vals`, the layout `jfield.sparse_matvec`
    reduces), `z` / `z_r` (`z_r = (0…0 ‖ r)`, the witness blinders with a zeroed
    instance part), and the blinder / witness / r fr arrays. Sparse — not dense —
    so the prove exports at recursion scale: densifying the recursion R1CS
    (`rows × vars` ≈ 15 GB; ~6 nonzeros/row) is infeasible. The generator+hiding
    base stack is a separate affine argument (`bases_h`) — an affine-typed jit
    constant doesn't lower, and the committer key is a runtime input on export.

    When `rt` is given (the general prover) the assignment + blinders come
    from runtime device arrays rather than being baked from the host lists, so the
    lowered core proves any assignment; otherwise they are host constants (the
    baked half-step / fold path)."""
    coo_a, coo_b, coo_c = (_coo_dev(to_coo(cv, m)) for m in (a, b, c))
    if rt is not None:
        # The assignment is a runtime input, so `M·z` can't be constant-folded; the
        # sparse `segment_sum` would survive as an i256 scatter-add the zkx GPU
        # atomic-RMW path can't lower (crashes codegen — same constraint as
        # `r1cs_nark_as._build_zk_fold_core`). Reduce DENSE instead (constant matrix
        # · runtime vector → no scatter), as the no-zk general core does. Returns
        # `(a_dense, b_dense, c_dense)`; densifying the recursion R1CS is ~15 GB, so
        # the general single-prove is fixture-/moderate-scale (the general prover's target).
        n = rt.r1cs_input.shape[0] + rt.witness.shape[0]
        dense = tuple(jnp.asarray(to_dense(cv, m, n)) for m in (a, b, c))
        z = jnp.concatenate([rt.r1cs_input, rt.witness])
        zr = jnp.concatenate([jnp.zeros_like(rt.r1cs_input), rt.r])
        return (coo_a, coo_b, coo_c, z, zr, rt.blinders, rt.witness, rt.r, dense)
    z = jnp.asarray(np.array(list(input) + list(witness), dtype=cv.fr))
    zr = jnp.asarray(np.array([0] * len(input) + list(r), dtype=cv.fr))
    blinders = jnp.asarray(np.array(
        [a_blinder, b_blinder, c_blinder, r_a_blinder, r_b_blinder, r_c_blinder, blinder_1, blinder_2],
        dtype=cv.fr))
    witness_arr = jnp.asarray(np.array(witness, dtype=cv.fr))
    r_arr = jnp.asarray(np.array(r, dtype=cv.fr))
    return (coo_a, coo_b, coo_c, z, zr, blinders, witness_arr, r_arr, None)


def _prove_zk_segment(cv: Curve, params: Any, matrices_hash: bytes, input: list[int],
                      coo_a: tuple, coo_b: tuple, coo_c: tuple, num_rows: int, z: jax.Array,
                      zr: jax.Array, bases_h: jax.Array, blinders: jax.Array,
                      witness_arr: jax.Array, r_arr: jax.Array, fork: bool = True,
                      input_u8b: jax.Array | None = None,
                      dense: tuple | None = None) -> NarkZkCore:
    """The zk NARK prove as on-device compute. **Plain** (un-decorated) so it
    inlines both into `prove_zk`'s own `@jax.jit` and into the AS top-level trace.

    The six `M·z` / `M·z_r` reduces run **in-trace** from the sparse COO
    (`jfield.sparse_matvec`, a `segment_sum`) rather than densifying — the
    recursion R1CS densified is infeasible, so this is what lets the zk prove
    export at recursion scale. The DuplexSponge / Poseidon params aren't jax
    pytrees, so the gamma sponge can't cross a `@jit` boundary as an argument — the
    segment instead closes over them (and the host `matrices_hash` / `input`, baked
    in as constants) and builds the sponge in-trace. A commitment is
    `Σ scalars·bases + blinder·hiding` (one `lax.msm`, blinder appended last);
    `blinders` is `[a, b, c, r_a, r_b, r_c, 1, 2]`. The eight commitments, the gamma
    challenge (its `FirstRoundMessage` point absorb), and the gamma-blinded
    responses are all one trace."""
    def commit(scalars: jax.Array, rand: jax.Array) -> jax.Array:
        return lax.msm(jnp.concatenate([scalars, rand.reshape(1)]), bases_h)

    # `dense` (the runtime/general path) reduces `M·v` as a dense matvec (constant
    # matrix · runtime vector → no GPU scatter); otherwise the sparse COO path (the
    # baked half-step / fold, where the reduce constant-folds away). See
    # `_prove_zk_field_prep`.
    dm_a, dm_b, dm_c = dense if dense is not None else (None, None, None)

    def reduce(coo: tuple, dense_m: jax.Array | None, vec: jax.Array) -> jax.Array:
        if dense_m is not None:
            return jfield.matvec(dense_m, vec)
        row_idx, col_idx, vals = coo
        return jfield.sparse_matvec(vals, col_idx, row_idx, vec, num_rows)

    z_a, z_b, z_c = reduce(coo_a, dm_a, z), reduce(coo_b, dm_b, z), reduce(coo_c, dm_c, z)
    r_a, r_b, r_c = reduce(coo_a, dm_a, zr), reduce(coo_b, dm_b, zr), reduce(coo_c, dm_c, zr)
    comm_a, comm_b, comm_c = commit(z_a, blinders[0]), commit(z_b, blinders[1]), commit(z_c, blinders[2])
    comm_r_a, comm_r_b, comm_r_c = commit(r_a, blinders[3]), commit(r_b, blinders[4]), commit(r_c, blinders[5])
    comm_1 = commit(z_a * r_b + z_b * r_a, blinders[6])
    comm_2 = commit(r_a * r_b, blinders[7])

    sp = _gamma_pre_sponge(cv, params, matrices_hash, input, fork, input_u8b=input_u8b)
    gamma = _gamma_finish(cv, sp, jnp.stack([comm_a, comm_b, comm_c]),
                          jnp.stack([comm_r_a, comm_r_b, comm_r_c, comm_1, comm_2]))

    blinded_witness = witness_arr + gamma * r_arr
    # sigma_{a,b,c} = blinder_M + gamma·r_blinder_M; sigma_o = c + gamma·1 + gamma²·2.
    sigma_abc = blinders[0:3] + gamma * blinders[3:6]
    sigma_o = blinders[2] + gamma[0] * blinders[6] + gamma[0] * gamma[0] * blinders[7]
    return NarkZkCore(comm_a, comm_b, comm_c, comm_r_a, comm_r_b, comm_r_c, comm_1,
                      comm_2, blinded_witness, sigma_abc, sigma_o, gamma)


def prove_zk_core(cv: Curve, a: Matrix, b: Matrix, c: Matrix, input: list[int], witness: list[int],
                  bases_h: jax.Array, params: Any, matrices_hash: bytes, r: list[int],
                  a_blinder: int, b_blinder: int, c_blinder: int, r_a_blinder: int,
                  r_b_blinder: int, r_c_blinder: int, blinder_1: int, blinder_2: int,
                  fork: bool = True, rt: NarkZkRuntime | None = None) -> NarkZkCore:
    """zk NARK prove returning on-device jax arrays (`NarkZkCore`) — the AS path's
    entry point, so the NARK commitments / gamma / responses thread into the rest
    of the prove without a host hop. `bases_h` is the pre-stacked generators +
    hiding base (an affine jit argument). Plain so it inlines into the AS
    top-level `@jax.jit`; `prove_zk` is the materializing standalone wrapper.

    `rt` (the general prover) lifts the assignment + randomness to runtime device arrays so
    one lowered core proves any prove; omitting it bakes them as host constants
    (the standalone half-step / fold path)."""
    coo_a, coo_b, coo_c, z, zr, blinders, witness_arr, r_arr, dense = _prove_zk_field_prep(
        cv, a, b, c, input, witness, r, a_blinder, b_blinder, c_blinder, r_a_blinder,
        r_b_blinder, r_c_blinder, blinder_1, blinder_2, rt=rt)
    input_u8b = rt.input_u8b if rt is not None else None
    return _prove_zk_segment(cv, params, matrices_hash, input, coo_a, coo_b, coo_c, len(a),
                             z, zr, bases_h, blinders, witness_arr, r_arr, fork,
                             input_u8b=input_u8b, dense=dense)


def build_zk_core(cv: Curve, a: Matrix, b: Matrix, c: Matrix, input: list[int], witness: list[int],
                  generators: list[np.ndarray], hiding: np.ndarray, params: Any,
                  matrices_hash: bytes, r: list[int], a_blinder: int, b_blinder: int,
                  c_blinder: int, r_a_blinder: int, r_b_blinder: int, r_c_blinder: int,
                  blinder_1: int, blinder_2: int, fork: bool = True) -> tuple:
    """Build the fused zk NARK core as `(core_fn, bases_h)`: a single `@jax.jit`
    closing over the circuit + the prover's sampled randomness (the COO matrices,
    `z`/`z_r`, the blinders — baked as constants) with the committer key + hiding
    base (`bases_h`) as its **sole runtime argument** — the export-correct shape
    (`core_fn(bases_h) -> NarkZkCore`). Shared by `prove_zk` (runs + materializes)
    and `export/export_nark_zk.py` (lowers it to one StableHLO module). The zk twin
    of `build_no_zk_core`; `fork=False` selects the standalone half-step's unforked
    gamma sponge."""
    rows = len(a)
    bases_h = jcurve.stack_affine(cv, list(generators[:rows]) + [hiding])

    @jax.jit
    def core_fn(bases_h: jax.Array) -> NarkZkCore:
        return prove_zk_core(cv, a, b, c, input, witness, bases_h, params, matrices_hash, r,
                             a_blinder, b_blinder, c_blinder, r_a_blinder, r_b_blinder,
                             r_c_blinder, blinder_1, blinder_2, fork)

    return core_fn, bases_h


def prove_zk(cv: Curve, a: Matrix, b: Matrix, c: Matrix, input: list[int], witness: list[int],
             generators: list[np.ndarray], hiding: np.ndarray, params: Any,
             matrices_hash: bytes, r: list[int], a_blinder: int, b_blinder: int,
             c_blinder: int, r_a_blinder: int, r_b_blinder: int, r_c_blinder: int,
             blinder_1: int, blinder_2: int, fork: bool = True) -> NarkZkProof:
    """ark `R1CSNark::prove` for the zk path, replaying the prover's sampled
    randomness (`r`, the blinders) rather than re-deriving arkworks' RNG.

    Blinds the first-round commitments (`comm_M = commit(M·z, blinder_M)`),
    commits the sigma-protocol cross terms (`comm_r_M`, `comm_1`, `comm_2`),
    derives `gamma`, and forms the blinded witness `w + gamma·r` and the response
    sigmas. The device compute (`prove_zk_core`) is one fused `@jax.jit` trace
    closing over the host sponge constants, with the committer key (`bases_h`) as
    its affine argument; materialization (`np.asarray` to host `fr` arrays) is the
    serialize seam. See `prove_zk_core` for the un-materialized (AS-threaded)
    form."""
    core_fn, bases_h = build_zk_core(cv, a, b, c, input, witness, generators, hiding, params,
                                     matrices_hash, r, a_blinder, b_blinder, c_blinder,
                                     r_a_blinder, r_b_blinder, r_c_blinder, blinder_1, blinder_2,
                                     fork)
    core = core_fn(bases_h)
    sigma_abc = np.asarray(core.sigma_abc, dtype=cv.fr)
    return NarkZkProof(
        np.asarray(core.comm_a), np.asarray(core.comm_b), np.asarray(core.comm_c),
        np.asarray(core.comm_r_a), np.asarray(core.comm_r_b), np.asarray(core.comm_r_c),
        np.asarray(core.comm_1), np.asarray(core.comm_2),
        np.asarray(core.blinded_witness, dtype=cv.fr), sigma_abc[0], sigma_abc[1], sigma_abc[2],
        np.asarray(core.sigma_o, dtype=cv.fr).reshape(-1)[0],
        np.asarray(core.gamma, dtype=cv.fr).reshape(-1)[0])


def serialize_zk_proof(cv: Curve, p: NarkZkProof) -> bytes:
    """CanonicalSerialize of the zk NARK `Proof`: the first-round message (three
    commitments, a `Some` flag, the five randomness commitments) then the
    second-round message (blinded-witness `Vec<Fr>`, a `Some` flag, the four
    sigmas)."""
    out = bytearray(curve.point_to_bytes(cv, p.comm_a) + curve.point_to_bytes(cv, p.comm_b)
                    + curve.point_to_bytes(cv, p.comm_c))
    out += b"\x01"  # FirstRoundMessage.randomness = Some
    for pt in (p.comm_r_a, p.comm_r_b, p.comm_r_c, p.comm_1, p.comm_2):
        out += curve.point_to_bytes(cv, pt)
    out += struct.pack("<Q", p.blinded_witness.shape[0]) + p.blinded_witness.tobytes()  # Vec<Fr>
    out += b"\x01"  # SecondRoundMessage.randomness = Some
    for s in (p.sigma_a, p.sigma_b, p.sigma_c, p.sigma_o):
        out += s.tobytes()
    return bytes(out)


def _gamma_pre_sponge(cv: Curve, params: Any, matrices_hash: bytes, inputs: list[int],
                      fork: bool = True, input_u8b: jax.Array | None = None) -> DuplexSponge:
    """The gamma sponge through the host byte-absorbs — optionally fork with the
    NARK protocol name, absorb the 32-byte matrices hash, then the scalar `inputs`
    (each 32B canonical LE, as one `Vec<u8>` Absorbable). This is the constant
    prefix before the `FirstRoundMessage` point absorb, shared by the standalone
    challenge and the in-jit prove core (computed eagerly, threaded into `@jit`).

    `fork` = whether the base sponge is the AS's `nark_sponge` (forked with
    `PROTOCOL_NAME`). The AS-embedded NARK forks; a **standalone** NARK prove —
    the recursion half-step, whose subject passes a plain `Sponge::new()` —
    draws an unforked sponge, so pass `fork=False` there.

    `input_u8b` (the general prover) is the `inputs`' `u8_batch`
    field-element packing supplied as a runtime input and absorbed directly — the
    in-trace `fr→u8` rechunk the zkx GPU plugin mis-lowers is done consumer-side.
    Byte-identical to the host pack: `absorb_bytes` is exactly `sp.absorb(
    u8_batch_field_array(input_bytes))`. Omitting it packs the bytes host-side."""
    sp = sponge.new_sponge(params)
    if fork:
        sp = absorbable.fork(cv, sp, PROTOCOL_NAME)
    sp = absorbable.absorb_bytes(cv, sp, matrices_hash)
    if input_u8b is not None:
        return sp.absorb(input_u8b)
    input_bytes = b"".join(int(s).to_bytes(32, "little") for s in inputs)
    return absorbable.absorb_bytes(cv, sp, input_bytes)


def _gamma_finish(cv: Curve, pre_sponge: DuplexSponge, comms: jax.Array,
                  randomness: jax.Array | None) -> jax.Array:
    """Absorb the `FirstRoundMessage` (comm packs ++ `Option` flag ++ randomness
    packs) into the pre-sponge and squeeze gamma. `comms` / `randomness` are
    pre-stacked `(N,)` affine arrays (`stack_affine` for host points, `jnp.stack`
    for the in-jit `lax.msm` outputs); `randomness` is None on the no-zk path. The
    point packing runs in-jit. Plain so it inlines into the `@jit`
    prove. Returns the `(1,)` truncated-128 fr challenge."""
    parts = [absorbable.point_to_field_array_jax(cv, comms),
             jnp.asarray(absorbable.option_flag(cv, randomness is not None))]
    if randomness is not None:
        parts.append(absorbable.point_to_field_array_jax(cv, randomness))
    sp = pre_sponge.absorb(jnp.concatenate(parts))
    _, ch = jsponge.squeeze_challenges(sp, 1, _CHALLENGE_BITS, cv)
    return ch


def compute_challenge(cv: Curve, params: Any, matrices_hash: bytes, inputs: list[int],
                      comms: list[np.ndarray], randomness: list[np.ndarray] | None = None) -> Any:
    """ark `R1CSNark::compute_challenge` (gamma) over host commitment points, as an
    `fr` scalar.

    `inputs` are fr values as ints; `comms` is the three first-round commitment
    points. `randomness`, when present (zk path), is the five first-round
    randomness commitments `[comm_r_a, comm_r_b, comm_r_c, comm_1, comm_2]`."""
    rstack = jcurve.stack_affine(cv, randomness) if randomness is not None else None
    ch = _gamma_finish(cv, _gamma_pre_sponge(cv, params, matrices_hash, inputs),
                       jcurve.stack_affine(cv, comms), rstack)
    return np.asarray(ch, dtype=cv.fr).reshape(-1)[0]
