#!/bin/bash
#
# Dense timing stability: several fresh processes with and without
# OpenMP thread pinning, on the same node.
#
#   sbatch slurm/stability.sbatch

#SBATCH -J stability
#SBATCH -p short
#SBATCH --constraint=ilk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH -o slurm-logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

REPEATS=${REPEATS:-5}

for i in $(seq "$REPEATS"); do
  # As in the 2026-10-08 sweep: OMP_NUM_THREADS inherited (1), no pinning
  PIN_THREADS=0 scripts/run_cpu_benchmark.sh \
    python scripts/check_timing_stability.py --label "unpinned-$i" | grep RESULT

  PIN_THREADS=1 scripts/run_cpu_benchmark.sh \
    python scripts/check_timing_stability.py --label "pinned-$i" | grep RESULT
done
