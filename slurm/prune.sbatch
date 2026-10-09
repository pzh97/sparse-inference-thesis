#!/bin/bash
#
# Prune + full WikiText-2 PPL for one line of configs/prune_sweep.txt.
#
#   mkdir -p slurm-logs
#   sbatch --array=0-$(( $(source slurm/common.sh; sweep_size) - 1 )) slurm/prune.sbatch
#
# Quick smoke test (tiny calibration/eval, separate output dirs):
#   SMOKE=1 sbatch --array=0,1,8,14,19,28 --time=00:30:00 slurm/prune.sbatch

#SBATCH -J prune
#SBATCH -p short
#SBATCH --constraint=ilk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH -o slurm-logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/common.sh

if [ -n "${SMOKE:-}" ]; then
  CKPT_ROOT="$STORE/sparse-inference/smoke/checkpoints"
  RES_ROOT="$STORE/sparse-inference/smoke/results"
  EXTRA="--max-samples 2"
  CALIB="--calib-samples 2"
else
  EXTRA=""
  CALIB=""
fi

read -r METHOD PATTERN SPARSITY <<< "$(sweep_line "$SLURM_ARRAY_TASK_ID")"
NAME=$(sweep_name "$SLURM_ARRAY_TASK_ID")
OUT="$RES_ROOT/pruning/$NAME.json"

echo "Task $SLURM_ARRAY_TASK_ID: $METHOD $PATTERN $SPARSITY -> $NAME on $(hostname)"

if [ -f "$OUT" ] && [ -z "${FORCE:-}" ]; then
  echo "$OUT exists, skipping (FORCE=1 to redo)"
  exit 0
fi

case "$METHOD" in
  dense)
    $RUN scripts/eval_wikitext2_ppl.py \
      --seq-len 2048 $EXTRA --output "$OUT"
    ;;
  magnitude)
    SP=""; [ "$SPARSITY" != "-" ] && SP="--sparsity $SPARSITY"
    $RUN scripts/eval_magnitude_pruning.py \
      --pattern "$PATTERN" $SP --seq-len 2048 $EXTRA \
      --save-path "$CKPT_ROOT/$NAME" --output "$OUT"
    ;;
  wanda|sparsegpt)
    SP=""; [ "$SPARSITY" != "-" ] && SP="--sparsity $SPARSITY"
    $RUN "scripts/eval_${METHOD}_pruning.py" \
      --pattern "$PATTERN" $SP --seq-len 2048 $CALIB $EXTRA \
      --save-path "$CKPT_ROOT/$NAME" --output "$OUT"
    ;;
  *)
    echo "Unknown method $METHOD" >&2
    exit 1
    ;;
esac
