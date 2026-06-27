# Project context for Claude Code

Overview, setup, the reproduction path, and the benchmark all live in
[`README.md`](README.md) — start there.

## Non-negotiables

The rules every change must respect:

- **Byte-identical to arkworks.** Every prover change is gated `fused GPU core ≡
  unmodified arkworks` over serialized bytes: the golden fixtures are emitted by the
  pristine `ark-accumulation` prover (the `../accumulation` dev-dependency), and the
  jax CPU port + the fused GPU core byte-match them. No behavior change ships without
  its byte-match.
- **Reuse the gadget, never re-implement it.** The in-circuit verifier gadget is
  reused from `ark-accumulation` as-is (it has no prover MSM); the prover itself
  lives only in the jax port (`python/accumulation_zorch/`). The repo re-derives
  neither.
- **Curve-generic, not duplicated.** Pallas and Vesta are two instantiations of
  one generic prover (`PastaCurve` in Rust, the `Curve` record in Python), never
  per-curve copies.

## Building & testing without a GPU

CI and CPU-only dev run the byte-match without any GPU/CUDA/libclang:

- **Rust:** `cargo test -p accumulation-zorch`, NOT bare `cargo test`. The
  `crates/zkx-pjrt` GPU shim is a workspace member whose `bindgen` build needs
  libclang + a vendored header; `-p` scopes the build to the main package. The
  `gpu` feature (and every `#![cfg(feature = "gpu")]` byte-match test) stays off
  by default, so the CPU build is self-contained.
- **Python:** install the jax fork from the public Fractalyze index *without*
  `zkx-cuda-pjrt` (the CPU backend needs no GPU plugin), then run the
  `python/accumulation_zorch/testing/*_test.py` scripts with `JAX_PLATFORMS=cpu`.
- The GPU byte-match tests are hardware-gated (`feature = "gpu"` + `#[ignore]`)
  and run only on a GPU box with `ZKX_PJRT_PLUGIN` set. See
  `.github/workflows/ci.yml` for the exact lanes.
