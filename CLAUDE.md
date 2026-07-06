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
- **Twin tests pin to the golden, not to each other.** Many primitives have a
  CPU/host impl and a jit/device twin (the `*_jax` convention). A device-twin
  test asserts the jit output against the arkworks golden fixture directly —
  never against the CPU twin's output, a weak oracle where both twins can share a
  bug. Look for an existing golden case before writing a jax-vs-CPU differential.
