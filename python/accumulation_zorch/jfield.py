"""Jit-able Fr field/vector/polynomial kernels — the jit/GPU-exportable
replacement for the numpy `zk_dtypes` arithmetic in `hp_as.py` (zorch#314
Phase 1).

Each kernel takes Fr `jnp` arrays and returns Fr `jnp` arrays; the orchestration
(`hp_as.prove_no_zk`) keeps the challenge / opening values as int lists and
rebuilds the arrays at the kernel boundary (data movement, not arithmetic). No
`zk_dtypes` numpy field math runs on the prove path.
"""

import functools

import jax
import jax.numpy as jnp


@jax.jit
def matvec(coeffs: jax.Array, z: jax.Array) -> jax.Array:
    """`M·z` over Fr: a dense `(rows × vars)` matrix times the `(vars,)` vector
    `z = r1cs_input ‖ witness`, as a broadcast multiply-and-sum → `(rows,)` Fr.
    (`@`/`einsum` don't lower for the field dtype.)"""
    return jnp.sum(coeffs * z[jnp.newaxis, :], axis=1)


@functools.partial(jax.jit, static_argnums=(4,))
def sparse_matvec(vals: jax.Array, col_idx: jax.Array, row_idx: jax.Array,
                  z: jax.Array, num_rows: int) -> jax.Array:
    """`M·z` over Fr from a sparse matrix in flat COO form — `segment_sum` of the
    per-nonzero products `vals·z[col_idx]` over their `row_idx` → `(num_rows,)` Fr.

    `vals` is the `(nnz,)` Fr coefficient vector, `col_idx`/`row_idx` the matching
    `(nnz,)` int column/row indices; `num_rows` is static (the segment count).
    This is the **on-device** equivalent of `nark.matrix_vec_mul`, the in-trace
    reduction the fused export needs: the recursion-verifier R1CS is ~22.5K×21K
    but ~6 nonzeros/row, so densifying it (`rows × vars` ≈ 15 GB) is infeasible —
    only the sparse reduce scales. The gather `z[col_idx]` and the scatter-add
    `segment_sum` (→ `stablehlo.scatter`) both lower over the i256 `fr` dtype in
    the zkx fork (zorch#326). Byte-identical to `matvec` on the densified matrix.
    """
    return jax.ops.segment_sum(vals * z[col_idx], row_idx, num_segments=num_rows)


@jax.jit
def combine_vectors(vectors: jax.Array, challenges: jax.Array) -> jax.Array:
    """`combine_vectors`: `output[li] = Σ_ni challenges[ni]·vectors[ni][li]`.
    `vectors` is `(m, L)` Fr, `challenges` `(m,)` Fr → `(L,)` Fr."""
    return jnp.sum(challenges[:, jnp.newaxis] * vectors, axis=0)


@functools.partial(jax.jit, static_argnums=(1,))
def powers(nu: jax.Array, count: int) -> jax.Array:
    """`[nu^0, …, nu^{count-1}]` as a `(count,)` Fr array. `nu` is `(1,)` Fr; the
    powers are built by repeated Fr multiply (no `lax.pow` over the field dtype)."""
    out = [jnp.ones_like(nu)]
    cur = out[0]
    for _ in range(count - 1):
        cur = cur * nu
        out.append(cur)
    return jnp.concatenate(out)


def _conv(a_col: jax.Array, b_rev: jax.Array) -> jax.Array:
    """Per-column polynomial product of `(n, L)` coefficient grids `a_col` and the
    already-reversed `b_rev`: `out[k,li] = Σ_{i+j=k} a_col[i,li]·b_rev[j,li]` →
    `(2n-1, L)` Fr. `n` is static, so the convolution unrolls at trace time."""
    n = a_col.shape[0]
    cols = []
    for k in range(2 * n - 1):
        acc = None
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                term = a_col[i] * b_rev[j]
                acc = term if acc is None else acc + term
        assert acc is not None  # every k in [0, 2n-2] has ≥1 decomposition i+j=k
        cols.append(acc)
    return jnp.stack(cols, axis=0)


@jax.jit
def t_vecs_no_zk(a: jax.Array, b: jax.Array, mu: jax.Array) -> jax.Array:
    """`compute_t_vecs` (no-zk): the per-column product-polynomial coefficients.

    `a`/`b` are `(n, L)` Fr (per-input witness vectors, padded to the common
    length `L`), `mu` is `(n,)` Fr. For each column, form `a(X,mu)` (coeff `ni` =
    `mu[ni]·a[ni,li]`) and the reversed `b(X)`, convolve, and stack the `2n-1`
    product coefficients → `(2n-1, L)` Fr."""
    return _conv(mu[:, jnp.newaxis] * a, jnp.flip(b, axis=0))


@jax.jit
def t_vecs_zk(a: jax.Array, b: jax.Array, mu: jax.Array, hiding_a: jax.Array,
              hiding_b: jax.Array, mu_n: jax.Array, mu_1: jax.Array) -> jax.Array:
    """`compute_t_vecs` (zk): like `t_vecs_no_zk` plus the hiding addends —
    `hiding_a·mu[n]` on the first `a` coefficient (input 0) and `hiding_b·mu[1]`
    on the first (post-reverse) `b` coefficient. `hiding_a`/`hiding_b` are `(L,)`,
    `mu_n`/`mu_1` are `(1,)`.

    The hiding addend is folded onto row 0 by rebuilding-and-concatenating, NOT by
    `.at[0].add`: that is a scatter-add, and a runtime scatter over the i256 fr
    dtype does not lower on the zkx GPU emitter at recursion scale (the scatter →
    atomic-RMW path bitcasts assuming an integer element type). At HP-fold scale `a`
    / `b` are the full constraint length, so this is the load-bearing difference vs
    the toy single-input prove. The value is identical: row 0 gains the hiding
    addend, the remaining rows are untouched."""
    ma = mu[:, jnp.newaxis] * a
    a_col = jnp.concatenate([(ma[0] + hiding_a * mu_n)[jnp.newaxis], ma[1:]], axis=0)
    fb = jnp.flip(b, axis=0)
    b_rev = jnp.concatenate([(fb[0] + hiding_b * mu_1)[jnp.newaxis], fb[1:]], axis=0)
    return _conv(a_col, b_rev)
