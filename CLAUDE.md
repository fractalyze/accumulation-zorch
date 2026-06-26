# Project context for Claude Code

Overview, setup, the reproduction path, and the benchmark all live in
[`README.md`](README.md) — start there.

## Non-negotiables

The rules every change must respect:

- **Byte-identical oracle gates.** Every prover change is gated `fused GPU core ≡
  CPU faithful copy ≡ unmodified arkworks` over serialized bytes (`src/oracle.rs`
  pins the CPU copy to arkworks; the fused GPU gates byte-match the arkworks-derived
  golden). No behavior change ships without its byte-match.
- **Copy the prover, reuse the gadget.** Copy an arkworks component only if it
  contains a prover MSM; the in-circuit verifier gadget has none, so it is reused
  as-is, never copied.
- **Curve-generic, not duplicated.** Pallas and Vesta are two instantiations of
  one generic prover (`PastaCurve` in Rust, the `Curve` record in Python), never
  per-curve copies.
