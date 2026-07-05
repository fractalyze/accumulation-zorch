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
- **Field elements stay as `cv.fr` arrays on the prove path — no python-int
  round-trip.** `fr` values carry as `cv.fr` arrays/scalars end-to-end; prove-stack
  seams accept an `fr` array or an int list via `np.asarray(x, dtype=cv.fr)`. Get
  canonical bytes with `np.asarray(x, dtype=cv.fr).tobytes()` / `arr[i].tobytes()`,
  not `cv.fr(<array element>)` (raises `expected number`). Type `fr` arrays
  `np.ndarray` and `fr` scalars `Any` — never `-> cv.fr` (a per-curve dtype
  instance; `NameError` at import, since these modules have no
  `from __future__ import annotations`).
