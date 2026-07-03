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

## Accumulation schemes

Two `ark-accumulation` schemes are ported, each a fused GPU core byte-identical to
the unmodified arkworks prover over the Pasta cycle (Pallas + Vesta):

- **`r1cs_nark_as`** (+ its `hp_as` Hadamard-product sub-step) — the R1CS-NARK
  accumulation the Pasta-cycle recursion uses. The whole zk prove — every
  commitment, the NARK + HP cores, all three Fiat-Shamir sponges — is one fused
  PJRT call. The prover is MSM-heavy (big Pedersen witness commitments), so the
  GPU win is the **prove** itself (the "Single AS prove" benchmark; the recursion
  IVC fold is host-bound by design).
- **`ipa_pc_as`** — the IPA-PC (Halo / DL-style) accumulation of BCMS20, **prove +
  decide, no-zk and zk**. Here the prover is *field*-heavy (building the degree-`d`
  check polynomial) with only small MSMs; the heavy size-`d` MSM is the
  **decider** (`final_comm_key == ⟨combined_check_poly_coeffs, generators⟩`). So
  the decider MSM is the GPU-value op — a pure MSM that scales far better than the
  fold (the "Decider size-`d` MSM" benchmark). The IPA-PC prover/verifier primitive
  itself (commit / open fold / reduce) is the vendored `zorch.pcs.ipa`
  (`python/zorch/`), driven by an arkworks-faithful challenger
  (`ipa_challenger.py`); `ipa_pc_as` supplies only the accumulation scheme on top.

The in-circuit verifier gadgets are reused from `ark-accumulation` as-is (they
have no prover MSM); the repo re-derives neither.

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
plugin from the public Fractalyze index (a vendored `zorch` subset — the Poseidon
sponge and the `pcs/ipa` IPA-PC prover/verifier — lives in-tree at `python/zorch/`,
Apache-2.0, so it is *not* pip-installed):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv --index-strategy unsafe-best-match \
  --index-url https://fractalyze.github.io/pypi/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  jax==0.10.0.dev20260702143130 jaxlib==0.10.0.dev20260702143130 \
  jax-cuda12-pjrt==0.10.0.dev20260702143130 jax-cuda12-plugin==0.10.0.dev20260702143130 \
  zk-dtypes==0.0.7 numpy absl-py
```

Point the env vars at that venv (copy-paste from the repo root):

```bash
export XLA_VENV_PYTHON=$PWD/.venv/bin/python
export XLA_PJRT_PLUGIN=$PWD/.venv/lib/python3.11/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
```

> The `0.10.0.dev` jax fork registers the Pasta curve dtypes (`pallas_sf` etc.);
> `zk-dtypes==0.0.7` carries them (`0.0.6` does not). `PYTHONPATH=python` resolves both
> `accumulation_zorch` and the vendored `zorch`. This pin set byte-matches arkworks:
> the CPU suite is 24/24, and the prover byte-matches on GPU via the Python prove
> (`JAX_PLATFORMS=cuda`).

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
  $XLA_VENV_PYTHON python/accumulation_zorch/testing/as_zk_test.py
# seed 0 / 42: (acc.instance 398B ‖ acc.witness 922B ‖ proof 482B) byte-matches arkworks
```

### Fused GPU byte-match (one core proves every seed)

```bash
# 1. Lower the ONE general fused core (CPU; no GPU needed for lowering).
JAX_PLATFORMS=cpu PYTHONPATH=python \
  $XLA_VENV_PYTHON export/export_prove.py           # -> artifacts/prove_zk_general.mlirbc
JAX_PLATFORMS=cpu PYTHONPATH=python \
  $XLA_VENV_PYTHON export/export_prove.py no-zk      # -> artifacts/prove_no_zk_general.mlirbc

# 2. GPU byte-match: the one core, fed each seed's witness/randomness at run time.
#    (`XLA_PJRT_PLUGIN` is read from the Setup export; --nocapture shows the
#     per-seed "byte-matches arkworks" lines)
cargo test --features gpu --test gpu_fused_prove_byte_match -- --ignored --test-threads=1 --nocapture
cargo test --features gpu --test gpu_fused_no_zk_prove_byte_match -- --ignored --test-threads=1 --nocapture
```

## Benchmark

Every accumulation scheme has two operations with opposite GPU stories:
**accumulate** (the per-step prove / IVC fold, which *defers* verification) and
**decide** (the deferred check, run once at the end). The two schemes sit at
opposite ends — R1CS-NARK's prover is MSM-heavy (so **accumulate** already wins on
GPU), while IPA-PC's prover is field-heavy with only small MSMs (so the GPU win is
the **decider**) — but both deciders are MSM-bound, and that is where the GPU
advantage is largest. Numbers are RTX-5090-class GPU; the fused GPU output
**byte-matches arkworks at every size**.

### R1CS-NARK — accumulate

**Single AS prove (one input).** Fused GPU prove (one PJRT call, warm) vs the
arkworks AS prove (CPU, `--release`), as the circuit size `n` (`num_constraints`)
grows:

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
- Reproduce: `PROVE_SIZES="4096 16384 32768" bench/bench.sh prove` (or
  `bench/bench.sh all`). Needs an idle GPU + the `XLA_VENV_PYTHON` /
  `XLA_PJRT_PLUGIN` env from [Setup](#setup).

**Recursion IVC fold.** The actual PCD step — fold one verifier-circuit NARK proof
into a prior accumulator (`num_addends = 3`), at recursion scale (`n = 77 556`):

| operation   | CPU arkworks (release) | GPU fused (1 PJRT call, warm) |   speedup |
| ----------- | ---------------------: | ----------------------------: | --------: |
| zk IVC fold |               2 447 ms |                      1 712 ms | **1.43×** |

The fold's GPU win is smaller than the single prove's because per-step accumulation
is **light by design** (it defers verification) and the fold bakes its `M·z` reduces
host-side — the zkx GPU emitter cannot lower the i256 `scatter`-add that survives
constant-folding at recursion scale, so part of the work stays on CPU
(Amdahl-capped). Reproduce: `bench/bench.sh fold`.

### R1CS-NARK — decide

The deferred verification the accumulate step set up: recompute the six size-`n`
Pedersen commitments — `comm_{a,b,c} = commit(M·z, σ)` and the `hp_as` check's
`test_comm_{1,2,3} = commit(a_vec, ρ₁), commit(b_vec, ρ₂), commit(a_vec∘b_vec, ρ₃)`
— and accept iff they equal the accumulator's stored commitments. The six MSMs run
as **one** fused PJRT call vs the CPU's six sequential variable-base MSMs:

|     `n` | CPU arkworks (6 MSMs) | GPU fused (1 PJRT call, warm) | speedup |
| ------: | --------------------: | ----------------------------: | ------: |
|  16 384 |                530 ms |                        18 ms |  **29×** |
|  65 536 |              1 698 ms |                        26 ms |  **65×** |
| 262 144 |              6 300 ms |                        54 ms | **117×** |

- The GPU is a slow-growing floor (18 → 54 ms over 16× the points) while the CPU is
  **O(6·n)**; the win grows from 29× to 117×. Fusing the six MSMs into one call (vs
  six separate CPU MSMs) compounds the per-MSM GPU advantage — the largest GPU win
  of any operation here.
- One lowered core is **curve-generic and zk-agnostic**: the same
  `as_decider_<curve>.mlirbc` decides both the no-zk and zk accumulators (the
  randomizers `σ` / `ρ` are runtime inputs, 0 on the no-zk path). Each row gates
  GPU == arkworks at scale.
- Reproduce: `bench/bench.sh r1cs-decide` (sizes from `R1CS_DECIDE_SIZES`).

### IPA-PC — decide

The IPA-PC prover is *field*-heavy (building the degree-`d` check polynomial) with
only small MSMs, so — mirror-image to R1CS-NARK — the GPU-value op is the
**decider's** size-`d` MSM (`final_key = Σ generatorsᵢ·coeffsᵢ`), not the
accumulate step:

|     `d` | CPU arkworks (release) | GPU fused (1 PJRT call, warm) |   speedup |
| ------: | ---------------------: | ----------------------------: | --------: |
|  16 384 |                  83 ms |                         16 ms |  **5.3×** |
|  65 536 |                 279 ms |                         18 ms | **15.5×** |
| 262 144 |                 988 ms |                         28 ms | **34.9×** |

- The GPU MSM is a **~16-18 ms floor** at these sizes (the Pippenger
  bucket-reduction kernel), so 4× the points (2¹⁴→2¹⁶) barely moves it while the
  CPU MSM is **O(d)** — the win grows from 5.3× to 34.9× across the sweep.
- The decider MSM core is curve-specific but zk-agnostic: the same lowered
  `lax.msm` decides both no-zk and zk accumulators (the zk-ness is in the
  host-computed coefficients). Each row gates GPU == arkworks at scale.
- Reproduce: `bench/bench.sh decide` (sizes from `DECIDE_SIZES`).

### IPA-PC — accumulate

The accumulate completes the matrix: the AS **prove** (combine several inputs'
succinct checks into one accumulator) and the **fold** (accumulate one input INTO a
prior accumulator — `old_accumulators` non-empty, the IVC step). An accumulator is
an `InputInstance` of the same shape as an input, so arkworks succinct-checks the
inputs then the accumulators into one list and combines them identically — the fold
is the prove fed `[inputs…, accumulators…]`, a prior *hiding* accumulator's succinct
check taking the zk path. Unlike the decider, this corner carries no GPU bench, and
that *is* the result:

- The accumulate's `IpaPC::open` is **sequential** — each round's Fiat-Shamir
  challenge is squeezed from that round's `L`/`R` fold commitments, so the per-round
  MSMs and the Poseidon sponge interleave with a two-way data dependency (the MSMs
  need the prior challenge; the next challenge needs the MSMs). It is **host-bound by
  design** — accumulation *defers* verification — so the GPU-value op for IPA-PC is
  the decider above, mirror-image to R1CS-NARK where it is the accumulate.
- The CPU port is **curve-generic and zk-agnostic in structure**: prove and fold,
  no-zk and zk, Pallas and Vesta all run the one combine + `IpaPC::open`, each
  byte-matched to arkworks.
- Reproduce: `ipa_as_test.py` (prove) and `ipa_as_fold_test.py` /
  `ipa_as_fold_zk_test.py` (fold), the CPU byte-match (run as in
  [Python jax prove byte-match](#python-jax-prove-byte-match-cpu)).
