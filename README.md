# accumulation-zorch

A GPU accumulation prover over the Pasta curve. The arkworks
[`ark-accumulation`](https://github.com/arkworks-rs/accumulation) native prove path
(`r1cs_nark_as` + `hp_as`), with the whole prover authored in **Python/JAX** and
compiled to a **single fused GPU kernel** — byte-identical to the reference
arkworks prover.

The GPU prove path is **fused** (`src/fused.rs`): the jax port of the prove
(`python/accumulation_zorch/`) — every commitment, the NARK + HP cores, and all
three Fiat-Shamir Poseidon sponges — is exported to one StableHLO `.mlirbc`
(`export/export_prove.py`) and run as a **single PJRT call**, à la
[bellman-zorch](https://github.com/fractalyze/bellman-zorch). Rust is a thin
consumer that feeds the committer key + assignment/randomness and re-serializes the
output. The assignment + all replayed randomness are runtime PJRT inputs, so one
exported core proves any statement (not a fixture replayer).

The byte-match oracle is the **pristine, unmodified arkworks prover** itself (the
`ark-accumulation` dev-dependency at `../accumulation`); the repo never
re-implements it. The fixture generators (`examples/dump_*.rs` and
`tests/recursion_step.rs`) drive arkworks to emit the golden
`(acc.instance ‖ acc.witness ‖ proof)` bytes, and the jax CPU port + the fused GPU
core are each gated byte-for-byte against those golden bytes.

## Setup

Clone this repo and the arkworks oracle **side by side**, then `cd` in. They must be
siblings because the crate's dev-dependency points at `../accumulation`:

```bash
git clone https://github.com/fractalyze/accumulation-zorch
git clone https://github.com/arkworks-rs/accumulation
git -C accumulation checkout 4a680af   # the revision the crate's ark-* 0.2 deps build against
cd accumulation-zorch
```

**Run every command below from inside `accumulation-zorch/`.** The other git deps
(`ark-sponge` / `ark-poly-commit`, on their `accumulation-experimental` branches) are
fetched by Cargo automatically — only `accumulation` is a manual clone.

Reproduction has two layers: a **Rust toolchain** regenerates the golden fixtures
by driving the pristine arkworks prover directly (no GPU, no Python); the **jax /
GPU byte-match** then checks the port against them and needs the Fractalyze zkx
Pasta toolchain.

### Regenerate the golden fixtures (Rust only — no GPU, no Python)

Just a Rust toolchain. The fixture generators (`examples/dump_*.rs`,
`tests/recursion_step.rs`) drive the unmodified `../accumulation` prover, so they
compile `../accumulation` — which needs `RUSTFLAGS="--cap-lints=warn"` (arkworks'
`#![deny(warnings)]` breaks modern rustc). The commands are under
[Reproduce](#reproduce); plain `cargo build` / `cargo test` (the crate's own suite)
need no extra flags.

### GPU + jax tier (the zkx Pasta toolchain)

Needs an NVIDIA GPU (CUDA), `clang`/`libclang` (the vendored `crates/zkx-pjrt` shim
generates its PJRT bindings with `bindgen` at build time), Python 3.11, and
[`uv`](https://docs.astral.sh/uv/). Install the matched zkx Pasta jax fork + GPU
plugin from the public Fractalyze index (the `zorch` Poseidon sponge is **vendored**
in `python/zorch/`, Apache-2.0 — see its `VENDOR.md` — so it is *not* pip-installed):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv --index-strategy unsafe-best-match \
  --index-url https://fractalyze.github.io/pypi/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  jax==0.0.5.dev20260624111151 jaxlib==0.0.5.dev20260624111151 \
  zkx-cuda-pjrt==0.0.5.dev20260624111151 zk-dtypes==0.0.7 numpy
```

Point the env vars at that venv (copy-paste from the repo root):

```bash
export ZKX_VENV_PYTHON=$PWD/.venv/bin/python
export ZKX_PJRT_PLUGIN=$PWD/.venv/lib/python3.11/site-packages/jax_plugins/cuda/pjrt_c_api_gpu_plugin.so
```

> Use the **`0.0.5.dev`** jax line, not `0.10.0.dev` — only the `0.0.5.dev` fork
> registers the Pasta curve dtypes (`pallas_sf` etc.), and `zk-dtypes==0.0.7` carries
> them (`0.0.6` does not). This exact pin set is verified to byte-match arkworks on
> GPU (single prove, zk + no-zk). `PYTHONPATH=python` resolves both
> `accumulation_zorch` and the vendored `zorch`.

## Reproduce

### Regenerate the golden fixtures from arkworks (fully external)

The golden `(acc.instance ‖ acc.witness ‖ proof)` bytes the byte-match tests check
against are produced by driving the **unmodified** arkworks prover — no GPU, no
Python (the generators compile `../accumulation`, hence `--cap-lints=warn`):

```bash
RUSTFLAGS="--cap-lints=warn" cargo run --example dump_as_zk > python/testdata/as_zk_fixtures.json
# regenerates the committed golden; a clean `git diff` confirms it still matches arkworks
```

### Python jax prove byte-match (CPU)

The jax port reproduces the arkworks `(acc.instance ‖ acc.witness ‖ proof)` bytes,
on CPU (the same trace the GPU export lowers):

```bash
JAX_PLATFORMS=cpu PYTHONPATH=python \
  $ZKX_VENV_PYTHON python/accumulation_zorch/testing/as_zk_test.py
# seed 0 / 42: (acc.instance 398B ‖ acc.witness 922B ‖ proof 482B) byte-matches arkworks
```

### Fused GPU byte-match (one core proves every seed)

```bash
# 1. Lower the ONE general fused core (CPU; no GPU needed for lowering).
JAX_PLATFORMS=cpu PYTHONPATH=python \
  $ZKX_VENV_PYTHON export/export_prove.py           # -> artifacts/prove_zk_general.mlirbc
JAX_PLATFORMS=cpu PYTHONPATH=python \
  $ZKX_VENV_PYTHON export/export_prove.py no-zk      # -> artifacts/prove_no_zk_general.mlirbc

# 2. GPU byte-match: the one core, fed each seed's witness/randomness at run time.
#    (`ZKX_PJRT_PLUGIN` is read from the Setup export; --nocapture shows the
#     per-seed "byte-matches arkworks" lines)
cargo test --features gpu --test gpu_fused_prove_byte_match -- --ignored --test-threads=1 --nocapture
cargo test --features gpu --test gpu_fused_no_zk_prove_byte_match -- --ignored --test-threads=1 --nocapture
```

## Benchmark

Two operations, two stories. Numbers are RTX-5090-class GPU; the fused GPU output
**byte-matches arkworks at every size**.

### Single AS prove (one input)

Fused GPU prove (one PJRT call, warm) vs the arkworks AS prove (CPU, `--release`),
as the circuit size `n` (`num_constraints`) grows:

|    `n` | CPU arkworks (release) | GPU fused (1 PJRT call) |   speedup |
| -----: | ---------------------: | ----------------------: | --------: |
|  4 096 |                 301 ms |                  651 ms |     0.46× |
| 16 384 |               1 053 ms |                  659 ms | **1.60×** |
| 32 768 |               1 924 ms |                  664 ms | **2.90×** |

- The GPU prove is a **flat ~659 ms floor** — 8× more MSM work (4 096 → 32 768) moves
  it +2%. It is GPU-compute-bound (the Pippenger bucket-reduction MSM kernel + the
  Poseidon Fiat-Shamir sponge + dispatch), not transfer. The five sequential Pasta
  MSMs and the composite sponge are the optimization target.
- The CPU prove is ~**O(n)** (MSM-bound). **Crossover ≈ n ≈ 8 K**; the GPU win grows
  with size.

Reproduce — [`bench/bench.sh`](bench/bench.sh) runs the whole sweep and prints
this table. Per size it wraps the arkworks CPU prove + off-tree fixture dump, the
CPU core lowering, and the warm GPU bench (which also gates the byte-match at
scale). Needs an idle GPU and the `ZKX_VENV_PYTHON` / `ZKX_PJRT_PLUGIN` env from
[Setup](#setup):

```bash
PROVE_SIZES="4096 16384 32768" bench/bench.sh prove   # or: bench/bench.sh all
```

### Recursion IVC fold (the accumulation step)

The actual PCD step — fold one verifier-circuit NARK proof into a prior accumulator
(`num_addends = 3`), at recursion scale (`n = 77 556`):

| operation   | CPU arkworks (release) | GPU fused (1 PJRT call, warm) |   speedup |
| ----------- | ---------------------: | ----------------------------: | --------: |
| zk IVC fold |               2 447 ms |                      1 712 ms | **1.43×** |

The fold's GPU win is smaller than the single prove's because per-step accumulation
is **light by design** (it defers verification) and the fold bakes its `M·z` reduces
host-side — the zkx GPU emitter cannot lower the i256 `scatter`-add that survives
constant-folding at recursion scale,
so part of the work stays on CPU (Amdahl-capped).

Reproduce — `bench/bench.sh fold` (arkworks CPU fold timing → recursion
fold-fixture dump → fold-core lowering → warm GPU fold bench):

```bash
bench/bench.sh fold
```
