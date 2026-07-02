# zorch-integration: IPA-PC reuse — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace accumulation-zorch's in-tree arkworks IPA-PC primitive (`ipa_pc.py`) with zorch's vendored `pcs/ipa`, driven by an arkworks-faithful `IpaChallenger`, keeping output byte-identical to arkworks.

**Architecture:** Vendor `zorch/pcs/ipa` (+ deps) as an Apache-2.0 subset pinned to `zorch@8a67799`. accumulation-zorch supplies a JAX, in-trace `IpaChallenger` reproducing its existing arkworks Fiat-Shamir, so zorch's JAX `lax.scan` fold produces the same bytes. The accumulation scheme (`ipa_pc_as.py`) rewires onto zorch's `commit`/`open`/`reduce_opening`/`settle`; the duplicated fold/coeff code in `ipa_pc.py` is deleted.

**Tech Stack:** Python 3.11, JAX 0.10 (Fractalyze Pasta fork), `zk_dtypes`, NumPy; Rust GPU byte-match consumer; zorch (private) vendored subset.

## Global Constraints

- **Byte-identical to arkworks.** Every phase gated by golden `../accumulation` fixtures under `python/testdata/` (`ipa_fixtures.json`, `ipa_as_fixtures.json`, + `_zk`/`_vesta` variants). No task merges without its byte-match.
- **Curves:** Pallas **and** Vesta, one curve-generic path (no per-curve copies).
- **Vendored code:** Apache-2.0 copies pinned to `zorch@8a67799e8abb4df4a04f7ec9b4de280bee3b4d53`; record the pin in `python/zorch/VENDOR`. Re-sync is manual.
- **CPU byte-match:** run with `JAX_PLATFORMS=cpu PYTHONPATH=python`. GPU byte-match: `RUSTFLAGS="--cap-lints=warn"`, `XLA_PJRT_PLUGIN=<0.10.0.dev20260702143130 plugin>`.
- **Scope:** IPA-PC only. `r1cs_nark_as` / `hp_as` untouched. No change to golden fixtures or the arkworks oracle.

---

## File Structure

**Vendor (copy verbatim from `zorch@8a67799`, add SPDX Apache-2.0 header if missing):**
- `python/zorch/pcs/__init__.py`, `python/zorch/pcs/protocol.py`
- `python/zorch/pcs/ipa/{__init__,config,setup,prover,verifier,challenger,math}.py`
- `python/zorch/transcript.py`
- `python/zorch/poly/__init__.py`, `python/zorch/poly/univariate.py` (for `powers`)
- `python/zorch/utils/__init__.py`, `python/zorch/utils/bits.py` (for `log2_strict_usize`)
- `python/zorch/VENDOR` (new): the pin commit + the `zorch/pcs/ipa` file list.

**Create (accumulation-zorch):**
- `python/accumulation_zorch/ipa_challenger.py` — the arkworks `IpaChallenger` (JAX, in-trace). One responsibility: reproduce accumulation-zorch's IPA-PC FS through zorch's challenger seam.
- `python/accumulation_zorch/testing/ipa_zorch_spike_test.py` — Phase-1 byte-match spike (removed or folded into `ipa_as_test.py` at Phase 4).

**Modify:**
- `python/accumulation_zorch/ipa_pc_as.py` — call zorch's IPA instead of local `ipa_pc`.
- `python/accumulation_zorch/ipa_pc.py` — delete the duplicated fold (`_open_fold`, `open_no_zk`, `open_zk`, folding-step helpers) and `compute_coeffs`/`evaluate` if subsumed; keep only what the challenger/accumulation still needs.
- `export/export_ipa.py` — Phase 3, decider MSM wiring.
- `README.md` — Phase 4, `ipa_pc_as` + vendored-zorch note.

---

## Phase 1 — Vendor + arkworks challenger + byte-match spike (the gate)

### Task 1: Vendor `zorch/pcs/ipa` + deps

**Files:**
- Create: the vendored files listed above under `python/zorch/`
- Create: `python/zorch/VENDOR`

**Interfaces:**
- Produces: importable `zorch.pcs.ipa.prover.IpaProver`, `zorch.pcs.ipa.verifier` (`reduce_opening`, `settle`), `zorch.pcs.ipa.challenger.IpaChallenger` (Protocol: `seed(commitment, point, value) -> (Self, xi0)`, `challenge(l, r) -> (Self, u_j)`), `zorch.pcs.ipa.config.IpaProof(l, r, a)`, `zorch.pcs.ipa.setup.IpaKey(basis, u, s=None)`, `zorch.poly.univariate.powers`, `zorch.utils.bits.log2_strict_usize`.

- [ ] **Step 1: Copy the vendored files** from a `zorch@8a67799` checkout (e.g. `git -C <zorch> archive 8a67799 zorch/pcs/ipa zorch/pcs/protocol.py zorch/pcs/__init__.py zorch/transcript.py zorch/poly zorch/utils/bits.py zorch/utils/__init__.py | tar -x -C /tmp/zorch-ipa`), then move the tree under `python/zorch/`. Keep only the `.py` (drop `BUILD.bazel`). Preserve the Apache-2.0 headers.

- [ ] **Step 2: Write `python/zorch/VENDOR`**

```
zorch vendored subset — pin: 8a67799e8abb4df4a04f7ec9b4de280bee3b4d53 (main, 2026-07-03)
Files: pcs/protocol.py, pcs/ipa/{config,setup,prover,verifier,challenger,math}.py,
transcript.py, poly/univariate.py, utils/bits.py
Re-sync: copy from the pin above; keep Apache-2.0 headers.
```

- [ ] **Step 3: Verify imports resolve**

Run: `JAX_PLATFORMS=cpu PYTHONPATH=python python -c "from zorch.pcs.ipa import prover, verifier, challenger, config, setup, math; from zorch.poly.univariate import powers; from zorch.utils.bits import log2_strict_usize; print('ok')"`
Expected: `ok` (fix any missing transitive import by vendoring that module too; re-run until clean).

- [ ] **Step 4: Commit**

```bash
git add python/zorch/pcs python/zorch/transcript.py python/zorch/poly python/zorch/utils python/zorch/VENDOR
git commit -m "chore: vendor zorch pcs/ipa (+transcript/poly/bits) @ 8a67799"
```

### Task 2: Arkworks `IpaChallenger` (JAX, in-trace)

**Files:**
- Create: `python/accumulation_zorch/ipa_challenger.py`
- Test: `python/accumulation_zorch/testing/ipa_challenger_test.py`

**Interfaces:**
- Consumes: `zorch.pcs.ipa.challenger.IpaChallenger` (Protocol), `jsponge.challenges_from_fq`, `absorbable.{fork, absorb_point, absorb_bytes, absorb_points_jax, point_to_field_array_jax}`, `sponge.new_sponge`, `curve.Curve`.
- Produces: `ArkIpaChallenger` (a `jax.tree_util.register_dataclass` pytree, so it rides zorch's `lax.scan` carry) with `seed(self, commitment, point, value) -> (ArkIpaChallenger, xi0)` and `challenge(self, l, r) -> (ArkIpaChallenger, u_j)`; a constructor `ark_challenger(cv, params) -> ArkIpaChallenger`.

The FS to reproduce is exactly `ipa_pc._round_challenges_from_seed` (`ipa_pc.py:68-91`), in JAX:
- `seed(commitment, point, value)`: fresh `"IPA-PC-2020"` sponge (`absorbable.fork(cv, sponge.new_sponge(params), b"IPA-PC-2020")`), `absorb_point(commitment)`, `absorb_bytes(to_bytes32(point) ++ to_bytes32(value))`, squeeze one truncated-128 challenge. Carry the squeezed challenge as `prev`.
- `challenge(l, r)`: fresh sponge, `absorb_bytes(prev.low_16_bytes)`, `absorb_point(l)`, `absorb_point(r)`, squeeze; return `(self', u_j)`.

Use the **JAX** absorb/squeeze primitives (`absorb_points_jax`, `jsponge.challenges_from_fq`) so the challenger is jittable inside the scan. `to_bytes32` / low-16 encodings must match `ipa_pc._fr32` and `int(rc).to_bytes(_CHALLENGE_BYTES, "little")` byte-for-byte.

- [ ] **Step 1: Write the failing test** — the JAX challenger's challenges equal the NumPy oracle's, on a golden fixture's L/R vectors.

```python
# ipa_challenger_test.py
import json
from pathlib import Path
import jax
from accumulation_zorch import curve, sponge, ipa_pc, ipa_challenger
# load one no-zk fixture (commitment, point, value, l_vec, r_vec) from ipa_fixtures.json
def test_ark_challenger_matches_numpy_oracle():
    cv = curve.PALLAS
    params = _load_params()               # sponge_fixtures.json -> sponge.poseidon_params
    comm, point, value, l_vec, r_vec = _load_ipa_fixture(cv)
    want = ipa_pc.succinct_check_challenges(cv, params, comm, point, value, l_vec, r_vec)
    ch = ipa_challenger.ark_challenger(cv, params)
    ch, _xi0 = ch.seed(comm, point, value)
    got = []
    for l, r in zip(l_vec, r_vec):
        ch, u = ch.challenge(l, r)
        got.append(int(jax.numpy.asarray(u).reshape(()) ))
    assert got == want
```

- [ ] **Step 2: Run it, verify it fails** — `JAX_PLATFORMS=cpu PYTHONPATH=python python -m pytest python/accumulation_zorch/testing/ipa_challenger_test.py -q` → FAIL (`ipa_challenger` has no `ark_challenger`).

- [ ] **Step 3: Implement `ipa_challenger.py`** — the `ArkIpaChallenger` dataclass pytree with `seed`/`challenge` per the FS above, reusing `jsponge`/`absorbable` JAX primitives. Match `ipa_pc._round_challenges_from_seed` exactly (fresh sponge per round; `_fr32` / low-16 encodings).

- [ ] **Step 4: Run it, verify it passes** — same pytest command → PASS for Pallas. Add a Vesta case; PASS.

- [ ] **Step 5: Commit** — `git add python/accumulation_zorch/ipa_challenger.py python/accumulation_zorch/testing/ipa_challenger_test.py && git commit -m "feat: arkworks-faithful IPA IpaChallenger (JAX, in-trace)"`

### Task 3: Byte-match spike — zorch `_open_one` + ark challenger vs golden

**Files:**
- Create: `python/accumulation_zorch/testing/ipa_zorch_spike_test.py`

**Interfaces:**
- Consumes: `zorch.pcs.ipa.prover._open_one(key, commitment, coeffs, x, fs) -> (fs, value, IpaProof)`, `zorch.pcs.ipa.setup.IpaKey`, `ipa_challenger.ark_challenger`, the golden `ipa_fixtures.json` (`generators`, `coeffs`/poly, `point`, expected `l_vec`/`r_vec`/`final_comm_key`/`c`).

This is the **de-risking gate**: does zorch's fold + our challenger reproduce our exact bytes?

- [ ] **Step 1: Write the byte-match test** — build `IpaKey(basis=generators, u=..., )` from the fixture, run `_open_one(key, commitment, coeffs, point, ark_challenger(cv, params))`, and assert the returned `IpaProof.l`/`.r`/`.a` (+ the final commitment key) hex-match the golden `l_vec`/`r_vec`/`final_comm_key`/`c` for Pallas and Vesta.

```python
def test_zorch_open_matches_golden_no_zk():
    for cv, fx in [(curve.PALLAS, "ipa_fixtures.json"), (curve.VESTA, "ipa_vesta_fixtures.json")]:
        key, commitment, coeffs, point, want = _load(cv, fx)
        fs = ipa_challenger.ark_challenger(cv, _params(cv))
        _fs, value, proof = _open_one(key, commitment, coeffs, point, fs)
        assert _hexL(proof.l) == want["l_vec"]
        assert _hexL(proof.r) == want["r_vec"]
        assert _hex(proof.a) == want["c"]
```

- [ ] **Step 2: Run on CPU** — `JAX_PLATFORMS=cpu PYTHONPATH=python python -m pytest python/accumulation_zorch/testing/ipa_zorch_spike_test.py -q`.
  - Expected first outcome is uncertain by design. **If PASS:** the byte-match holds through the seam — proceed to Phase 2.
  - **If FAIL:** diff the first diverging round: print zorch's per-round `(L_j, R_j, u_j)` vs the golden and vs `ipa_pc.open_no_zk`'s. Reconcile the mismatch (candidates: `IpaKey.u`/`h_prime` seed convention, coeff ordering / `powers` direction, the low-16 vs 32-byte challenge encoding, or the fold's scaled-half orientation). Fix in `ipa_challenger.py` or the `IpaKey` construction. **Do not proceed until it passes.**

- [ ] **Step 3: Confirm CPU MSM path** — verify the run above did not error on `lax.msm` (the xla-fork CPU `kMsm`→Pippenger lowering handles it). If it errors, capture the error and stop — this is a prerequisite finding.

- [ ] **Step 4: Commit** — `git add python/accumulation_zorch/testing/ipa_zorch_spike_test.py && git commit -m "test: byte-match spike — zorch IPA open + ark challenger vs golden"`

---

## Phase 2 — Rewire the accumulation scheme

### Task 4: Point `ipa_pc_as.py` at zorch's IPA

**Files:**
- Modify: `python/accumulation_zorch/ipa_pc_as.py` (the `prove_*` / open paths that call `ipa_pc.open_no_zk`/`open_zk`)
- Test: `python/accumulation_zorch/testing/ipa_as_test.py` (existing — must stay green)

**Interfaces:**
- Consumes: `zorch.pcs.ipa.prover._open_one` / `_open_one_zk`, `ipa_challenger.ark_challenger` (+ its zk variant if the zk path needs `hiding_challenge`), `zorch.pcs.ipa.verifier.{reduce_opening, settle}`.
- Produces: unchanged public API of `ipa_pc_as` (`prove_no_zk_accumulator`, `prove_zk_accumulator`, `decide_final_key`, `Accumulator`) so `ipa_as_test.py` and `export_ipa.py` need no signature changes.

- [ ] **Step 1: Run the existing suite to capture the green baseline** — `JAX_PLATFORMS=cpu PYTHONPATH=python python python/accumulation_zorch/testing/ipa_as_test.py` → currently PASS (records the target bytes).

- [ ] **Step 2: Add the zk challenger variant** if needed — extend `ipa_challenger.py` with `hiding_challenge(commitment, hiding_comm, point, value)` matching `ipa_pc.succinct_check_challenges_zk` (`ipa_pc.py:108-131`); unit-test it against the NumPy oracle exactly as Task 2.

- [ ] **Step 3: Rewire the open calls** — replace `ipa_pc.open_no_zk(...)`/`open_zk(...)` inside `ipa_pc_as.py` with `_open_one(...)`/`_open_one_zk(...)` driven by the ark challenger, mapping the fixture arrays onto `IpaKey`. Keep `combine`, RLP, `Accumulator`, `decide_final_key` unchanged.

- [ ] **Step 4: Run `ipa_as_test.py` (both curves, no-zk + zk)** → PASS. If any byte differs, diff against the Phase-1 spike output and fix. Do not proceed until green.

- [ ] **Step 5: Commit** — `git commit -am "refactor: drive ipa_pc_as accumulation open via zorch pcs/ipa"`

### Task 5: Delete the duplicated `ipa_pc.py` fold/coeff code

**Files:**
- Modify: `python/accumulation_zorch/ipa_pc.py` (remove `_open_fold`, `open_no_zk`, `open_zk`, `_even/_odd_*_step`, and `compute_coeffs`/`evaluate` if no longer referenced)
- Test: full CPU suite

- [ ] **Step 1: Find remaining references** — `grep -rn "ipa_pc.open_no_zk\|ipa_pc.open_zk\|_open_fold\|compute_coeffs\|ipa_pc.evaluate" python/ export/` — expect only the challenger/accumulation-specific uses to remain.

- [ ] **Step 2: Delete the dead functions** in `ipa_pc.py`; keep the arkworks challenge helpers only if `ipa_challenger.py` does not fully subsume them (prefer moving the FS constants like `IPA_PC_DOMAIN`, `_CHALLENGE_*` into `ipa_challenger.py` and deleting `ipa_pc.py` if empty).

- [ ] **Step 3: Run the full CPU suite** — `JAX_PLATFORMS=cpu PYTHONPATH=python` over `ipa_as_test.py`, `ipa_challenger_test.py`, `ipa_zorch_spike_test.py`, and `as_zk_test.py` (regression: r1cs untouched) → all PASS.

- [ ] **Step 4: Commit** — `git commit -am "refactor: remove ipa_pc fold/compute now provided by zorch"`

---

## Phase 3 — GPU decider

### Task 6: Decider MSM through zorch (or keep the one-MSM core)

**Files:**
- Modify: `export/export_ipa.py` (the `build_decider_core` MSM)
- Test: `tests/gpu_fused_ipa_decide_byte_match.rs`

- [ ] **Step 1: Decide the wiring** — the decider is `final_key = Σ generatorsᵢ·coeffsᵢ` (one `lax.msm`). zorch's `verifier.settle` is the same size-`n` MSM. If `settle`'s signature accepts the raw `(coeffs, generators)` cleanly, route through it; otherwise keep the local one-MSM core (byte-identical either way). Document the choice in the module docstring.

- [ ] **Step 2: Re-export both curves** — `JAX_PLATFORMS=cpu PYTHONPATH=python <venv>/bin/python export/export_ipa.py pallas` and `... vesta` → `artifacts/ipa_decider_msm_{pallas,vesta}.mlirbc`.

- [ ] **Step 3: GPU byte-match** — `XLA_PJRT_PLUGIN=<0.10.0.dev20260702143130 plugin> RUSTFLAGS="--cap-lints=warn" cargo test --features gpu --test gpu_fused_ipa_decide_byte_match -- --ignored --test-threads=1 --nocapture` → PASS (both curves). (Use `XLA_CLIENT_MEM_FRACTION=0.5` if the GPU is contended.)

- [ ] **Step 4: Commit** — `git commit -am "refactor: IPA decider MSM via zorch settle (byte-match preserved)"`

---

## Phase 4 — Tests + docs

### Task 7: Consolidate tests, README, cleanup

**Files:**
- Modify: `README.md` (the `ipa_pc_as` bullet + the vendored-zorch note under the GPU tier)
- Modify/remove: `python/accumulation_zorch/testing/ipa_zorch_spike_test.py` (fold its assertions into `ipa_as_test.py` or keep as a dedicated zorch-seam byte-match test)

- [ ] **Step 1: Fold the spike test** into the permanent suite (keep a `test_zorch_open_matches_golden` in `ipa_as_test.py`), so the byte-match through the seam is a standing gate.

- [ ] **Step 2: Update `README.md`** — note that `ipa_pc_as` drives zorch's vendored `pcs/ipa` (pin `8a67799`); update the Accumulation-schemes `ipa_pc_as` description if it references the old local IPA.

- [ ] **Step 3: Full green run** — CPU suite (`ipa_as_test.py`, `ipa_challenger_test.py`, `as_zk_test.py`) + GPU (`gpu_fused_ipa_decide_byte_match`) → all PASS. Confirm `git diff` on `python/testdata/*` is empty (golden untouched).

- [ ] **Step 4: Commit** — `git commit -am "docs+test: zorch IPA reuse — README + standing byte-match gate"`

---

## Notes for the implementer

- The **byte-match is the spec**. If any hex differs, stop and diff round-by-round; never "adjust the golden."
- zorch's exact signatures live in the vendored `python/zorch/pcs/ipa/{challenger,prover,verifier,config,setup}.py` — read them; the API summary here is from a survey and the parameter names should be confirmed against the vendored source.
- The zk path (`_open_one_zk`, `hiding_challenge`, `IpaZkProof`) mirrors the no-zk path; do Pallas no-zk end-to-end first (Tasks 2-4), then add zk, then Vesta.
- `lax.msm` is GPU-primary; the CPU byte-match relies on the xla-fork CPU `kMsm` lowering — Task 3 Step 3 confirms it early.
