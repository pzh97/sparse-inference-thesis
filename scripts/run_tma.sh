#!/bin/bash
#
# Top-down Microarchitecture Analysis of one dense/CSR kernel.
#
# Usage (from the repository root, inside a compute job):
#   scripts/run_tma.sh <out_dir> <tma_kernel.py args...>
#
# Example:
#   scripts/run_tma.sh results/tma/sgpt50-down12-csr-m128 \
#     --model-path $STORE/sparse-inference/checkpoints/sparsegpt-50 \
#     --layer model.layers.12.mlp.down_proj --format csr --m 128
#
# The kernel runs in a steady loop; perf attaches to it with -p only after
# warmup, so model loading is not part of the counts. Use --threads 1 for
# a clean single-core kernel view: with several threads, OpenMP spin-wait
# between calls is counted too.

set -euo pipefail

OUT=$1
shift

PERF_SECONDS=${PERF_SECONDS:-10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT"
READY="$OUT/ready.pid"
rm -f "$READY"

"$SCRIPT_DIR/run_cpu_benchmark.sh" \
  python "$SCRIPT_DIR/tma_kernel.py" \
  --ready-file "$READY" \
  --seconds $((PERF_SECONDS + 5)) \
  --output "$OUT/kernel.json" \
  "$@" > "$OUT/driver.log" 2>&1 &
DRIVER=$!

for _ in $(seq 600); do
  [ -s "$READY" ] && break
  if ! kill -0 "$DRIVER" 2>/dev/null; then
    echo "Driver exited before becoming ready:" >&2
    cat "$OUT/driver.log" >&2
    exit 1
  fi
  sleep 0.5
done

PID=$(cat "$READY")

EVENTS='cycles:u,instructions:u'
EVENTS+=',cpu/event=0xa3,umask=0x04,cmask=4,name=STALLS_TOTAL/u'
EVENTS+=',cpu/event=0xa3,umask=0x14,cmask=20,name=STALLS_MEM_ANY/u'
EVENTS+=',cpu/event=0xa3,umask=0x0c,cmask=12,name=STALLS_L1D_MISS/u'
EVENTS+=',cpu/event=0xa3,umask=0x05,cmask=5,name=STALLS_L2_MISS/u'
EVENTS+=',cpu/event=0xa3,umask=0x06,cmask=6,name=STALLS_L3_MISS/u'
EVENTS+=',cpu/event=0xa6,umask=0x40,name=BOUND_ON_STORES/u'
EVENTS+=',cpu/event=0xa6,umask=0x02,name=EXE_1_PORTS/u'
EVENTS+=',cpu/event=0xa6,umask=0x04,name=EXE_2_PORTS/u'

perf stat -x, -o "$OUT/perf.csv" -p "$PID" \
  -e '{slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound}:u' \
  -e "$EVENTS" \
  -- sleep "$PERF_SECONDS"

wait "$DRIVER"

python3 "$SCRIPT_DIR/tma_report.py" \
  "$OUT/perf.csv" "$OUT/kernel.json" \
  --output "$OUT/tma.json" | tee "$OUT/tma.txt"
