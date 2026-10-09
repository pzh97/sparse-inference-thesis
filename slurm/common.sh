# Shared settings for the SLURM jobs. Source from the repository root.

CKPT_ROOT="$STORE/sparse-inference/checkpoints"
RES_ROOT="${RES_ROOT:-$STORE/sparse-inference/results}"
SWEEP_CONFIG="${SWEEP_CONFIG:-configs/prune_sweep.txt}"

RUN="scripts/run_cpu_benchmark.sh python"

# Print line $1 (0-based) of the sweep config, skipping comments/blank lines.
sweep_line() {
  grep -v '^\s*#' "$SWEEP_CONFIG" | grep -v '^\s*$' | sed -n "$(($1 + 1))p"
}

sweep_size() {
  grep -v '^\s*#' "$SWEEP_CONFIG" | grep -cv '^\s*$'
}

# Same naming as pruning_patterns.pattern_tag():
#   unstructured 0.5 -> u50   2:4 -> nm2-4   block16 0.7 -> block16x16-70
pattern_tag() {
  local pattern=$1 sparsity=$2 pct

  case "$pattern" in
    *:*) echo "nm${pattern%%:*}-${pattern##*:}"; return ;;
  esac

  pct=$(python3 -c "print(round(float('$sparsity') * 100))")

  case "$pattern" in
    unstructured) echo "u${pct}" ;;
    block*x*)     echo "${pattern}-${pct}" ;;
    block*)       local b=${pattern#block}; echo "block${b}x${b}-${pct}" ;;
    *)            echo "unknown pattern $pattern" >&2; return 1 ;;
  esac
}

# Checkpoint name for sweep line $1: <method>-<tag>
sweep_name() {
  local method pattern sparsity
  read -r method pattern sparsity <<< "$(sweep_line "$1")"
  if [ "$method" = dense ]; then
    echo dense
  else
    echo "${method}-$(pattern_tag "$pattern" "$sparsity")"
  fi
}
