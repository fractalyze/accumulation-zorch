#!/usr/bin/env bash
#
# Reproduce the README "Benchmark" tables in one command: run the fused GPU
# prover and the arkworks CPU prover at matched sizes and print the comparison
# as a Markdown table — exactly the tables in README.md.
#
#   bench/bench.sh prove        # the "Single AS prove" sweep (sizes from PROVE_SIZES)
#   bench/bench.sh r1cs-decide  # the R1CS-NARK decider (6 size-n MSMs; R1CS_DECIDE_SIZES)
#   bench/bench.sh decide       # the IPA-PC "Decider size-d MSM" sweep (DECIDE_SIZES)
#   bench/bench.sh fold         # the "Recursion IVC fold" point (one recursion-scale row)
#   bench/bench.sh all          # all of the above (default)
#
# Live progress streams to stderr; the finished table(s) print to stdout only
# once every point is measured — so `bench/bench.sh prove >table.md` captures a
# clean table and nothing else.
#
# Prereqs are the README "GPU + frx tier": an **idle** NVIDIA GPU, plus the GPU
# plugin the README Setup exports —
#
#   XLA_PJRT_PLUGIN   path to pjrt_c_api_gpu_plugin.so (loaded by the Rust GPU tests)
#
# The frx lowering steps run via `bazel run //export:export_*` (zorch + the frx
# fork come from Bazel, see MODULE.bazel), so no separate venv interpreter is
# needed here.
#
# Optional knobs:
#   PROVE_SIZES        space-separated num_constraints for `prove` (default "4096 16384 32768")
#   R1CS_DECIDE_SIZES  space-separated MSM sizes n for `r1cs-decide` (default "16384 65536 262144")
#   DECIDE_SIZES       space-separated MSM sizes d for `decide` (default "16384 65536 262144")
#   ARTIFACTS_DIR scratch dir for fixtures, .mlirbc, and per-step logs
#                 (default: $SCRATCH_DIR, else a mktemp dir)
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-all}"

: "${XLA_PJRT_PLUGIN:?set XLA_PJRT_PLUGIN to pjrt_c_api_gpu_plugin.so (see README Setup)}"
[ -f "$XLA_PJRT_PLUGIN" ] || { echo "XLA_PJRT_PLUGIN not found: $XLA_PJRT_PLUGIN" >&2; exit 1; }

ART="${ARTIFACTS_DIR:-${SCRATCH_DIR:-$(mktemp -d)}}"
mkdir -p "$ART"

# All progress goes to stderr so stdout stays a clean table.
say()  { printf '%s\n' "$*" >&2; }
step() { printf '%s' "$*" >&2; }

# A Rust `Duration` Debug token ("1.05s" / "651.2ms"), or a bare number already
# in ms, -> milliseconds. Stdlib only, so any python3 works.
to_ms() {
  python3 - "$1" <<'PY'
import re, sys
s = sys.argv[1].strip().replace(" ", "")
m = re.match(r"^([0-9.]+)(ns|µs|us|ms|s|m)?$", s)
if not m:
    print("nan"); raise SystemExit
v, u = float(m.group(1)), (m.group(2) or "ms")
print("%.1f" % (v * {"ns": 1e-6, "µs": 1e-3, "us": 1e-3, "ms": 1.0, "s": 1e3, "m": 6e4}[u]))
PY
}

speedup() { awk -v c="$1" -v g="$2" 'BEGIN { printf "%.2f", c / g }'; }

# label  cpu_ms  gpu_ms  ->  one Markdown row; speedup bold when the GPU wins.
row() {
  local sp cell; sp=$(speedup "$2" "$3"); cell="${sp}×"
  awk -v s="$sp" 'BEGIN { exit !(s + 0 > 1) }' && cell="**${sp}×**"
  printf '| %s | %.0f ms | %.0f ms | %s |\n' "$1" "$2" "$3" "$cell"
}

require() { [ -n "$2" ] || { say "  ✗ could not parse $1 — see logs in $ART"; exit 1; }; }

# Builds the "Single AS prove" table on stdout; progress on stderr.
bench_prove() {
  local sizes; read -r -a sizes <<<"${PROVE_SIZES:-4096 16384 32768}"
  local n fix derr cpu_tok gpu_tok cpu_ms gpu_ms
  say "Single AS prove — sweeping n = ${sizes[*]} (warm GPU; a few minutes)…"
  echo "## Single AS prove"
  echo
  echo '|     `n` | CPU arkworks (release) | GPU fused (1 PJRT call) | speedup |'
  echo '| ------: | ---------------------: | ----------------------: | ------: |'
  for n in "${sizes[@]}"; do
    fix="$ART/fix_$n.json"; derr="$ART/dump_$n.err"
    printf '  n=%-6s ' "$n" >&2
    step "CPU prove… "
    AS_ZK_NUM_CONSTRAINTS="$n" cargo run --quiet --release --example dump_as_zk >"$fix" 2>"$derr"
    cpu_tok=$(sed -n 's/.*seed 0 .*= \(.*\) (CPU arkworks).*/\1/p' "$derr" | head -1)
    require "CPU timing (n=$n)" "$cpu_tok"
    step "lower… "
    AS_ZK_FIXTURE="$fix" ACCUMULATION_ZORCH_ARTIFACTS="$ART" JAX_PLATFORMS=cpu \
      bazel run //export:export_prove >"$ART/export_$n.log" 2>&1
    step "GPU… "
    gpu_tok=$(AS_ZK_FIXTURE="$fix" FUSED_MLIRBC="$ART/prove_zk_general.mlirbc" \
      XLA_PJRT_PLUGIN="$XLA_PJRT_PLUGIN" \
      cargo test --quiet --release --features gpu --test gpu_fused_bench -- --ignored --nocapture \
      2>"$ART/gpu_$n.log" \
      | sed -n 's/.*median=\([^ ]*\) over.*/\1/p' | head -1)
    require "GPU timing (n=$n)" "$gpu_tok"
    cpu_ms=$(to_ms "$cpu_tok"); gpu_ms=$(to_ms "$gpu_tok")
    say "$(printf '→ CPU %.0f ms  GPU %.0f ms  %s×' "$cpu_ms" "$gpu_ms" "$(speedup "$cpu_ms" "$gpu_ms")")"
    row "$(printf '%7s' "$n")" "$cpu_ms" "$gpu_ms"
  done
}

# Builds the "Recursion IVC fold" table on stdout; progress on stderr.
bench_fold() {
  local feats="recursion,gpu" cpu_ms gpu_ms
  say "Recursion IVC fold — one recursion-scale point (warm GPU)…"
  step "  arkworks CPU fold… "
  cpu_ms=$(cargo test --quiet --release --features "$feats" --test recursion_step \
    vesta::arkworks_fold_timing -- --ignored --nocapture 2>"$ART/fold_cpu.log" \
    | sed -n 's/.*median \([0-9.]*\) ms over.*/\1/p' | head -1)
  require "CPU fold timing" "$cpu_ms"
  step "fixture… "
  ACCUMULATION_ZORCH_ARTIFACTS="$ART" cargo test --quiet --release --features "$feats" \
    --test recursion_step vesta::dump::dump_recursion_fold_zk -- --nocapture >"$ART/fold_dump.log" 2>&1
  step "lower… "
  ACCUMULATION_ZORCH_ARTIFACTS="$ART" PROVE_CURVE=vesta JAX_PLATFORMS=cpu \
    bazel run //export:export_fold_zk >"$ART/fold_export.log" 2>&1
  step "GPU fold… "
  gpu_ms=$(XLA_PJRT_PLUGIN="$XLA_PJRT_PLUGIN" ACCUMULATION_ZORCH_ARTIFACTS="$ART" \
    cargo test --quiet --release --features gpu --test gpu_fused_fold_bench -- --ignored --nocapture \
    2>"$ART/fold_gpu.log" \
    | sed -n 's/.*warm run median \([0-9.]*\) ms.*/\1/p' | head -1)
  require "GPU fold timing" "$gpu_ms"
  say "$(printf '→ CPU %.0f ms  GPU %.0f ms  %s×' "$cpu_ms" "$gpu_ms" "$(speedup "$cpu_ms" "$gpu_ms")")"
  echo "## Recursion IVC fold (the accumulation step)"
  echo
  echo '| operation   | CPU arkworks (release) | GPU fused (1 PJRT call, warm) | speedup |'
  echo '| ----------- | ---------------------: | ----------------------------: | ------: |'
  row "zk IVC fold" "$cpu_ms" "$gpu_ms"
}

# Builds the "Decider size-d MSM" table on stdout; progress on stderr. The decider
# is the IPA accumulation's GPU-value op (`final_key = Σ generators·coeffs`, a pure
# size-`d` MSM), so unlike the host-bound fold it shows the GPU's MSM advantage as
# `d` grows. Inputs are synthetic (random committer key + coefficients) — the MSM
# time is size-driven and the byte-match gate keeps it honest.
bench_decide() {
  local sizes; read -r -a sizes <<<"${DECIDE_SIZES:-16384 65536 262144}"
  say "Decider size-d MSM — sweeping d = ${sizes[*]} (warm GPU)…"
  echo "## Decider size-\`d\` MSM"
  echo
  echo '|       `d` | CPU arkworks (release) | GPU fused (1 PJRT call, warm) | speedup |'
  echo '| --------: | ---------------------: | ----------------------------: | ------: |'
  for n in "${sizes[@]}"; do
    step "  d=$n lower… "
    IPA_DECIDER_SIZE="$n" ACCUMULATION_ZORCH_ARTIFACTS="$ART" JAX_PLATFORMS=cpu \
      bazel run //export:export_ipa >"$ART/decide_export_$n.log" 2>&1
    step "GPU MSM… "
    local line cpu_ms gpu_ms
    line=$(IPA_DECIDER_SIZE="$n" IPA_DECIDER_MLIRBC="$ART/ipa_decider_msm_bench.mlirbc" \
      XLA_PJRT_PLUGIN="$XLA_PJRT_PLUGIN" \
      cargo test --quiet --release --features gpu --test gpu_fused_ipa_decide_bench -- --ignored --nocapture \
      2>"$ART/decide_gpu_$n.log" | grep '\[bench\]')
    cpu_ms=$(to_ms "$(printf '%s' "$line" | sed -n 's/.*CPU arkworks=\([^ ]*\).*/\1/p')")
    gpu_ms=$(to_ms "$(printf '%s' "$line" | sed -n 's/.*min=\([^ ]*\).*/\1/p')")
    require "decide d=$n CPU" "$cpu_ms"; require "decide d=$n GPU" "$gpu_ms"
    say "$(printf '→ d=%s  CPU %.0f ms  GPU %.0f ms  %s×' "$n" "$cpu_ms" "$gpu_ms" "$(speedup "$cpu_ms" "$gpu_ms")")"
    row "$n" "$cpu_ms" "$gpu_ms"
  done
}

# Builds the "R1CS-NARK decider" table on stdout; progress on stderr. The R1CS-NARK
# accumulation decider's GPU-value op is six size-`n` MSMs (`comm_{a,b,c}` + the HP
# `test_comm_{1,2,3}`), fused into one PJRT call vs the CPU's six sequential MSMs —
# so, like the IPA decider, it shows the GPU's MSM advantage as `n` grows. Inputs
# are synthetic (random committer key + reduced vectors); the byte-match-at-scale
# gate keeps it honest.
bench_r1cs_decide() {
  local sizes; read -r -a sizes <<<"${R1CS_DECIDE_SIZES:-16384 65536 262144}"
  say "R1CS-NARK decider (6 size-n MSMs) — sweeping n = ${sizes[*]} (warm GPU)…"
  echo "## R1CS-NARK decider (6 size-\`n\` MSMs)"
  echo
  echo '|       `n` | CPU arkworks (6 MSMs) | GPU fused (1 PJRT call, warm) | speedup |'
  echo '| --------: | --------------------: | ----------------------------: | ------: |'
  for n in "${sizes[@]}"; do
    step "  n=$n lower… "
    AS_DECIDE_SIZE="$n" ACCUMULATION_ZORCH_ARTIFACTS="$ART" JAX_PLATFORMS=cpu \
      bazel run //export:export_as_decide >"$ART/r1cs_decide_export_$n.log" 2>&1
    step "GPU MSMs… "
    local line cpu_ms gpu_ms
    line=$(AS_DECIDE_SIZE="$n" AS_DECIDE_MLIRBC="$ART/as_decider_bench.mlirbc" \
      XLA_PJRT_PLUGIN="$XLA_PJRT_PLUGIN" \
      cargo test --quiet --release --features gpu --test gpu_fused_r1cs_decide_bench -- --ignored --nocapture \
      2>"$ART/r1cs_decide_gpu_$n.log" | grep '\[bench\]')
    cpu_ms=$(to_ms "$(printf '%s' "$line" | sed -n 's/.*CPU arkworks=\([^ ]*\).*/\1/p')")
    gpu_ms=$(to_ms "$(printf '%s' "$line" | sed -n 's/.*min=\([^ ]*\).*/\1/p')")
    require "r1cs-decide n=$n CPU" "$cpu_ms"; require "r1cs-decide n=$n GPU" "$gpu_ms"
    say "$(printf '→ n=%s  CPU %.0f ms  GPU %.0f ms  %s×' "$n" "$cpu_ms" "$gpu_ms" "$(speedup "$cpu_ms" "$gpu_ms")")"
    row "$n" "$cpu_ms" "$gpu_ms"
  done
}

# Measure first (progress to stderr, tables captured), then print the finished
# table(s) to stdout in one clean block.
prove_out=""; decide_out=""; r1cs_decide_out=""; fold_out=""
case "$mode" in
  prove)  prove_out=$(bench_prove) ;;
  decide) decide_out=$(bench_decide) ;;
  r1cs-decide) r1cs_decide_out=$(bench_r1cs_decide) ;;
  fold)   fold_out=$(bench_fold) ;;
  all)    prove_out=$(bench_prove); r1cs_decide_out=$(bench_r1cs_decide)
          decide_out=$(bench_decide); fold_out=$(bench_fold) ;;
  *) echo "usage: bench/bench.sh [prove|decide|r1cs-decide|fold|all]" >&2; exit 2 ;;
esac

say ""
first=1
for out in "$prove_out" "$r1cs_decide_out" "$decide_out" "$fold_out"; do
  [ -n "$out" ] || continue
  [ "$first" -eq 1 ] || echo
  printf '%s\n' "$out"
  first=0
done
say "(artifacts + per-step logs in: $ART)"
