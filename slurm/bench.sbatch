#!/bin/bash
#
# Pattern analysis + per-layer and end-to-end dense vs CSR benchmarks for
# the checkpoint of one line of configs/prune_sweep.txt.
# Run after slurm/prune.sbatch (e.g. with --dependency=afterok:<jobid>).
#
#   sbatch --array=1-39 slurm/bench.sbatch
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
  CKPT_ROOT="$STORE/sparse-inference/smoke/checkpoints"
  RES_ROOT="$STORE/sparse-inference/smoke/results"
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

step "$RES_ROOT/patterns/$NAME.json" \
  $RUN scripts/analyze_sparsity_patterns.py \
    --model-path "$CKPT" --output "$RES_ROOT/patterns/$NAME.json"

step "$RES_ROOT/layers/$NAME.json" \
  $RUN scripts/bench_layer_formats.py \
    --model-path "$CKPT" $LAYERS \
    --csv "$RES_ROOT/layers/$NAME.csv" --output "$RES_ROOT/layers/$NAME.json"

for TARGETS in mlp all; do
  step "$RES_ROOT/e2e/$NAME-$TARGETS.json" \
    $RUN scripts/bench_qwen_csr_inference.py \
      --model-path "$CKPT" --targets "$TARGETS" $E2E \
      --output "$RES_ROOT/e2e/$NAME-$TARGETS.json"
done

for MODE in prefill decode; do
  step "$RES_ROOT/profile/$NAME-csr-all-$MODE.txt" \
    $RUN scripts/profile_qwen.py \
      --model "$CKPT" --csr-targets all --mode "$MODE" \
      --output "$RES_ROOT/profile/$NAME-csr-all-$MODE.txt"
done
