#!/bin/bash
#
# Pattern analysis + per-layer and end-to-end dense vs CSR benchmarks for
# the checkpoint of one line of configs/prune_sweep.txt.
# Run after slurm/prune.sbatch (e.g. with --dependency=afterok:<jobid>).
#
#   sbatch --array=1-39 slurm/bench.sbatch
#
# Options (environment):
#   STEPS        subset of "patterns layers e2e profile" (default: all)
#   CSR_IMPLS    CSR implementations for e2e/profile (v0, int32, mlp_t);
#                v0 = the 2026-10-08 sweep, output names without suffix
#   LAYER_MODES  "hot" and/or "cold" (cold-cache per-layer benchmark)
#   E2E_DIR      results subdirectory for e2e (default e2e; e2e_lk1 = runs
#                with logits_to_keep=1, i.e. after commit "lm_head fix")
#   E2E_REPS     independent processes per e2e configuration (default 1;
#                >1 adds -r<i> to the file names). Dense timings differ
#                between processes (~400 vs ~505 ms prefill at 512).
#
# A whole socket is requested so that no other job shares our L3 and
# memory bandwidth; the run itself is bound to one 16-core NUMA node.

#SBATCH -J bench
#SBATCH -p short
#SBATCH --constraint=ilk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o slurm-logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/common.sh

if [ -n "${SMOKE:-}" ]; then
  CKPT_ROOT="${SMOKE_ROOT:-$STORE/sparse-inference/smoke}/checkpoints"
  RES_ROOT="${SMOKE_ROOT:-$STORE/sparse-inference/smoke}/results"
  LAYERS="--layers 0 --m-values 1 128 --runs 5 --min-time 0.02"
  E2E="--prompt-lengths 128 --decode-tokens 4 --prefill-repeats 2 --decode-repeats 1"
else
  LAYERS="--layers 0 6 12 18 23"
  E2E="--prompt-lengths 128 512 --decode-tokens 32"
fi

NAME=$(sweep_name "$SLURM_ARRAY_TASK_ID")
CKPT="$CKPT_ROOT/$NAME"

echo "Task $SLURM_ARRAY_TASK_ID: $NAME on $(hostname)"

if [ "$NAME" = dense ] || [ ! -d "$CKPT" ]; then
  echo "No pruned checkpoint for $NAME, nothing to do"
  exit 0
fi

# Run a step unless its output already exists (FORCE=1 to redo).
step() {
  local out=$1
  shift
  if [ -f "$out" ] && [ -z "${FORCE:-}" ]; then
    echo "skip: $out exists"
  else
    "$@"
  fi
}

STEPS=${STEPS:-"patterns layers e2e profile"}
CSR_IMPLS=${CSR_IMPLS:-v0}
LAYER_MODES=${LAYER_MODES:-hot}

has_step() { [[ " $STEPS " == *" $1 "* ]]; }

# v0 keeps the original file names
impl_suffix() { [ "$1" = v0 ] && echo "" || echo "-$1"; }

if has_step patterns; then
  step "$RES_ROOT/patterns/$NAME.json" \
    $RUN scripts/analyze_sparsity_patterns.py \
      --model-path "$CKPT" --output "$RES_ROOT/patterns/$NAME.json"
fi

if has_step layers; then
  for MODE in $LAYER_MODES; do
    if [ "$MODE" = cold ]; then DIR=layers_cold; COLD=--cold; else DIR=layers; COLD=""; fi
    step "$RES_ROOT/$DIR/$NAME.json" \
      $RUN scripts/bench_layer_formats.py \
        --model-path "$CKPT" $LAYERS $COLD \
        --csv "$RES_ROOT/$DIR/$NAME.csv" --output "$RES_ROOT/$DIR/$NAME.json"
  done
fi

for IMPL in $CSR_IMPLS; do
  SUF=$(impl_suffix "$IMPL")

  if has_step e2e; then
    for REP in $(seq "${E2E_REPS:-1}"); do
      REP_SUF=""; [ "${E2E_REPS:-1}" -gt 1 ] && REP_SUF="-r$REP"
      for TARGETS in mlp all; do
        OUT="$RES_ROOT/${E2E_DIR:-e2e}/$NAME-$TARGETS$SUF$REP_SUF.json"
        step "$OUT" \
          $RUN scripts/bench_qwen_csr_inference.py \
            --model-path "$CKPT" --targets "$TARGETS" --csr-impl "$IMPL" $E2E \
            --output "$OUT"
      done
    done
  fi

  if has_step profile; then
    for MODE in prefill decode; do
      step "$RES_ROOT/profile/$NAME-csr-all-$MODE$SUF.txt" \
        $RUN scripts/profile_qwen.py \
          --model "$CKPT" --csr-targets all --csr-impl "$IMPL" --mode "$MODE" \
          --output "$RES_ROOT/profile/$NAME-csr-all-$MODE$SUF.txt"
    done
  fi
done
