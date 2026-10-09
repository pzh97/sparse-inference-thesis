#!/bin/bash

module load singularity/4.2.2

export SINGULARITY_CACHEDIR="$STORE/sparse-inference/singularity-cache"

export PYTHONPATH="$STORE/sparse-inference/pydeps:${PYTHONPATH:-}"

export HF_HOME="$STORE/sparse-inference/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Use only the cached (pinned) model revision and dataset; set to 0 to
# allow downloads.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}

# PIN_THREADS=1: one OpenMP thread per core, pinned (the login environment
# exports OMP_NUM_THREADS=1, which is overridden here).
if [ "${PIN_THREADS:-0}" = 1 ]; then
  export OMP_NUM_THREADS=${BIND_CPUS:-16}
  export MKL_NUM_THREADS=${BIND_CPUS:-16}
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
fi

# Inside a SLURM job the allowed CPUs depend on the allocation, so pick a
# NUMA node whose first BIND_CPUS (default 16) physical cores are all
# allowed (falls back to the first allowed CPUs). CPU_BIND / MEM_BIND
# still win. Not OMP_NUM_THREADS: the login environment exports it as 1.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${CPU_BIND:-}" ]; then
  read -r CPU_BIND MEM_BIND < <(python3 - "${BIND_CPUS:-16}" <<'EOF'
import glob, os, re, sys

n = int(sys.argv[1])
allowed = os.sched_getaffinity(0)

def expand(text):
    cpus = []
    for part in text.strip().split(","):
        a, _, b = part.partition("-")
        cpus.extend(range(int(a), int(b or a) + 1))
    return cpus

for path in sorted(glob.glob("/sys/devices/system/node/node*/cpulist"),
                   key=lambda p: int(re.search(r"node(\d+)", p).group(1))):
    node = re.search(r"node(\d+)", path).group(1)
    cpus = expand(open(path).read())[:n]
    if len(cpus) == n and set(cpus) <= allowed:
        print(f"{cpus[0]}-{cpus[-1]}", node)
        sys.exit()

cpus = sorted(allowed)[:n]
node = os.path.basename(glob.glob(f"/sys/devices/system/cpu/cpu{cpus[0]}/node*")[0])[4:]
print(",".join(map(str, cpus)), node)
EOF
)
fi

CPU_BIND=${CPU_BIND:-0-15}
MEM_BIND=${MEM_BIND:-0}
export CPU_BIND MEM_BIND

numactl --physcpubind="$CPU_BIND" --membind="$MEM_BIND" \
singularity exec \
  "$STORE/sparse-inference/containers/intel-pytorch-cpu.sif" \
  "$@"
