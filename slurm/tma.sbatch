#!/bin/bash
#
# Top-down Microarchitecture Analysis (perf, user-mode) of dense vs CSR
# kernels on real pruned matrices and on random masks.
#
#   sbatch slurm/tma.sbatch
#
# Override the selection with CHECKPOINTS="..." LAYER=... M_VALUES="..."
# THREADS="..." FORMATS="..." (existing runs are skipped).

#SBATCH -J tma
#SBATCH -p short
#SBATCH --constraint=ilk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o slurm-logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/common.sh

if [ -n "${SMOKE:-}" ]; then
  CKPT_ROOT="${SMOKE_ROOT:-$STORE/sparse-inference/smoke}/checkpoints"
  RES_ROOT="${SMOKE_ROOT:-$STORE/sparse-inference/smoke}/results"
fi

CHECKPOINTS=${CHECKPOINTS:-"sparsegpt-u50 sparsegpt-u70 sparsegpt-u90 magnitude-u90 sparsegpt-nm2-4 wanda-block16x16-70"}
LAYER=${LAYER:-model.layers.12.mlp.down_proj}
M_VALUES=${M_VALUES:-"1 128"}
THREADS=${THREADS:-"1 16"}
FORMATS=${FORMATS:-"dense csr csr_rowmajor csr_i32 csr_i32_rowmajor"}
RANDOM_FORMATS=${RANDOM_FORMATS:-"dense csr csr_i32"}
export PERF_SECONDS=${PERF_SECONDS:-10}

OUT_ROOT="$RES_ROOT/tma"

run() {
  local out=$1
  shift
  if [ -f "$out/tma.json" ] && [ -z "${FORCE:-}" ]; then
    echo "skip: $out"
    return
  fi
  echo "== $out"
  # Keep going if one configuration fails.
  scripts/run_tma.sh "$out" "$@" || echo "FAILED: $out"
}

for T in $THREADS; do
  for M in $M_VALUES; do
    for CK in $CHECKPOINTS; do
      [ -d "$CKPT_ROOT/$CK" ] || { echo "missing $CK"; continue; }
      for FMT in $FORMATS; do
        run "$OUT_ROOT/$CK/${LAYER##model.layers.}/t$T-m$M-$FMT" \
          --model-path "$CKPT_ROOT/$CK" --layer "$LAYER" \
          --format "$FMT" --m "$M" --threads "$T"
      done
    done

    # Random-mask controls with the shape of the down_proj
    for S in 0.5 0.7 0.9; do
      for FMT in $RANDOM_FORMATS; do
        run "$OUT_ROOT/random-$S/896x4864/t$T-m$M-$FMT" \
          --shape 896 4864 --sparsity "$S" \
          --format "$FMT" --m "$M" --threads "$T"
      done
    done
  done
done
