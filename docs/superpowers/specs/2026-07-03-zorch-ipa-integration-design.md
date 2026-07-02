# zorch-integration: reuse zorch's IPA-PC primitive

**Status:** design (approved direction; pending spec review)
**Date:** 2026-07-03
**Branch:** `zorch-integration`

## Summary

Replace accumulation-zorch's in-tree, arkworks-faithful IPA-PC primitive
(`python/accumulation_zorch/ipa_pc.py`) with zorch's `pcs/ipa` (vendored,
Apache-2.0), driven through an **arkworks-faithful `IpaChallenger`** so the output
stays **byte-identical to arkworks**. The IPA fold moves from host-side NumPy to
JAX (zorch's `lax.scan` `open`). The accumulation scheme
(`ipa_pc_as.py`) and the byte-match oracle fixtures stay; `ipa_pc_as` rewires to
call zorch's IPA instead of the local copy.

Scope is **IPA-PC only** — `r1cs_nark_as` / `hp_as` are untouched.

## Motivation

zorch now owns a full IPA-PC prover/verifier (`zorch/pcs/ipa`, on `main`),
built to be arkworks-byte-exact over the Pasta curves
with an explicit challenger seam for accumulation consumers. accumulation-zorch
maintains a parallel implementation of the same primitive. Converging on zorch's
canonical IPA removes the duplication and inherits zorch's fused-JAX prover
(scan-fold, `valid_count`-bounded MSMs) instead of a host-NumPy fold.

## Non-negotiables

- **Byte-identical to arkworks.** Every phase is gated by the existing golden
  `ipa_as` fixtures (arkworks reference, `../accumulation`). No phase merges
  without its byte-match. This is the project's standing rule (see `CLAUDE.md`).
- **Reuse, don't re-implement.** The accumulation glue is kept; the IPA-PC math
  is delegated to zorch, not re-derived.
- **Curve-generic (Pallas + Vesta), no per-curve copies.**

## Current architecture (what exists today)

- `ipa_pc.py` (NumPy, host-side) — the arkworks IPA-PC **primitive**: succinct-check
  Fiat-Shamir challenges, `compute_coeffs` (the `h(X)` check polynomial), the
  O(log d) L/R **fold** (`_open_fold`, `open_no_zk`, `open_zk`), `IpaProof`.
- `ipa_pc_as.py` (NumPy, host-side) — the **accumulation scheme** on top: `combine`,
  RLP, `Accumulator`/`AccumulatorInstance`, `prove_*`, `decide_final_key`.
- `export/export_ipa.py` — the **only** GPU core: the decider's size-`d` MSM
  `final_key = Σ generatorsᵢ·coeffsᵢ` (one `jcurve.msm`/`lax.msm`), a single fused
  PJRT call. Challenge derivation + `compute_coeffs` stay host-side by design
  (field-heavy, not GPU-value).
- Byte-match oracle: `testing/ipa_as_test.py` (CPU) + `gpu_fused_ipa_decide_*`
  (GPU decider), all against `../accumulation` golden bytes.

## Target architecture

- Vendor `zorch/pcs/ipa` and its runtime deps into `python/zorch/` as an
  Apache-2.0 subset (extends today's `python/zorch/hash` + `fusion` copy). Pin to a
  specific zorch `main` commit.
- Implement accumulation-zorch's **`IpaChallenger`** (the seam) from the existing
  arkworks FS in `ipa_pc.py` — the one piece that carries byte-exactness:
  fresh domain-separated sponge per round (`"IPA-PC-2020"`), the `to_bytes`
  conventions, and `CHALLENGE_SIZE`.
- `ipa_pc_as.py` calls zorch's `IpaProver`/`IpaVerifier`
  (`commit`/`open`/`reduce_opening`/`settle`) through that challenger. The IPA fold
  now runs in JAX (zorch's `open` = `lax.scan` over `k = log₂ d` rounds).
- `ipa_pc.py` shrinks to the challenger + any accumulation-specific glue not
  covered by zorch; the duplicated fold/`compute_coeffs` are deleted.
- `export/export_ipa.py` re-points the decider MSM at zorch's `settle` (or keeps the
  local one-MSM core if byte-identical — decided during implementation; the two are
  the same `lax.msm`).

### Vendored surface (from the Explore survey)

`zorch/pcs/ipa/` (`config`, `setup`, `prover`, `verifier`, `challenger`, `math`) plus
its runtime deps: `zorch/transcript.py` (only for the default challenger — our
custom challenger may bypass it), `zorch/poly/univariate.py::powers`,
`zorch/utils/bits.py::log2_strict_usize`, `zorch/pcs/protocol.py` (type-only seam).
The sole EC primitive is `lax.msm` (jax-fork custom op) — already what
accumulation-zorch uses.

## Sequencing

### Phase 0 — vendored pin (already satisfied)

zorch's IPA is **already on `main`**: the full module — `prover.py`/`verifier.py`
with `open`/`reduce_opening`/`settle`, the `IpaChallenger` seam, and the `lax.scan`
fold. Pin the vendored copy to
`zorch@8a67799e8abb4df4a04f7ec9b4de280bee3b4d53` (main, 2026-07-03). No zorch merge
PR is needed — the `perf/344-ipa-scan-fold` branch is fully superseded by `main`
(`git diff origin/main <branch> -- zorch/pcs/ipa/` is empty). accumulation-zorch
Phase 1 can start immediately.

### Phase 1 — validation spike (de-risk the byte-match)

Vendor `pcs/ipa` + deps. Implement the arkworks `IpaChallenger`. Drive **one**
opening through zorch's IPA + this challenger and assert it byte-matches a golden
`ipa_as` fixture (equivalently, matches `ipa_pc.open_no_zk`/`open_zk`'s current
output). Also confirm zorch's `lax.msm`-based fold runs on `JAX_PLATFORMS=cpu` (via
the xla-fork CPU `kMsm`→Pippenger lowering). **Gate: if the bytes differ, stop and
reconcile the FS/encoding before deleting anything.**

### Phase 2 — rewire the accumulation scheme

Point `ipa_pc_as.py` at zorch's IPA; keep the accumulation glue. Delete the
duplicated fold/`compute_coeffs` in `ipa_pc.py`. `testing/ipa_as_test.py` stays green
for both curves, no-zk and zk.

### Phase 3 — GPU decider

Re-point `export/export_ipa.py` at zorch's `settle`/MSM (or keep the local one-MSM
core if byte-identical). `gpu_fused_ipa_decide_byte_match` + `_bench` green on both
curves.

### Phase 4 — tests + docs

All byte-match tests green (CPU `ipa_as_test.py` + GPU decider). Update `README.md`
(the `ipa_pc_as` description and the vendored-zorch note). Confirm the full suite
still byte-matches arkworks.

## Testing / gates

- Golden fixtures from `../accumulation` (unchanged) gate every phase.
- Spike (Phase 1) has a dedicated byte-match assertion before any deletion.
- CPU: `testing/ipa_as_test.py`. GPU: `gpu_fused_ipa_decide_byte_match`.
- Curve-generic: Pallas and Vesta both covered.

## Risks

- **Byte-match through the seam** (primary). zorch's IPA + our arkworks challenger
  must reproduce our exact bytes. Mitigation: Phase 1 spike gates it.
- **`lax.msm` on CPU** for the byte-match oracle. The xla-fork has a CPU
  `kMsm`→Pippenger lowering; confirm zorch's fold runs on CPU in Phase 1.
- **Vendoring surface grows** (transcript/poly/bits). Managed as Apache-2.0 copies
  pinned to `zorch@8a67799`; re-sync is manual (matches today's model).

## Out of scope

- `r1cs_nark_as` / `hp_as` (the R1CS-NARK prover) — untouched.
- Deduping `jcurve`/`jfield`/`jsponge` against zorch (a possible later, broader
  pass; explicitly deferred).
- Any change to the arkworks oracle or golden fixtures.

## Open questions (resolve during implementation)

- Does `ipa_pc.py` retain anything after the fold/`compute_coeffs` are removed, or
  is it fully subsumed (challenger moves to its own module)?
- Does the GPU decider keep its own one-MSM core or route through zorch's `settle`
  (byte-identical either way; pick the simpler wiring)?
