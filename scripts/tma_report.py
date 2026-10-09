"""
Turn the `perf stat -x,` output of run_tma.sh into TMA metrics.

Level 1 comes directly from the Ice Lake PERF_METRICS topdown events
(exact). The Level 2/3 memory breakdown uses the CYCLE_ACTIVITY /
EXE_ACTIVITY formulas of the Intel TMA model for Ice Lake, simplified
(no L3-hit/DRAM split refinements); cross-check with VTune
(`vtune -collect uarch-exploration`) before drawing fine-grained
conclusions.

Usage:
    python3 tma_report.py <perf.csv> [driver.json] [--output tma.json]
"""

import argparse
import json


def parse_perf_csv(path):
    counts = {}

    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")

            if len(parts) < 3 or not parts[0] or line.startswith("#"):
                continue

            try:
                value = float(parts[0])
            except ValueError:
                # "<not counted>" / "<not supported>"
                continue

            name = parts[2].split(":")[0]
            counts[name] = value

    return counts


def tma(c):
    slots = c["slots"]

    retiring = c["topdown-retiring"] / slots
    bad_spec = c["topdown-bad-spec"] / slots
    frontend = c["topdown-fe-bound"] / slots
    backend = c["topdown-be-bound"] / slots

    cycles = c["cycles"]

    backend_cycles = (
        c["STALLS_TOTAL"]
        + c["EXE_1_PORTS"]
        + retiring * c["EXE_2_PORTS"]
        + c["BOUND_ON_STORES"]
    )

    memory_fraction = (
        (c["STALLS_MEM_ANY"] + c["BOUND_ON_STORES"]) / backend_cycles
        if backend_cycles > 0
        else 0.0
    )

    memory = backend * memory_fraction

    return {
        "ipc": c["instructions"] / cycles,
        "L1": {
            "retiring": retiring,
            "bad_speculation": bad_spec,
            "frontend_bound": frontend,
            "backend_bound": backend,
        },
        "L2": {
            "memory_bound": memory,
            "core_bound": backend - memory,
        },
        # Fractions of cycles stalled at each level of the hierarchy.
        "L3_memory": {
            "l1_bound": max(c["STALLS_MEM_ANY"] - c["STALLS_L1D_MISS"], 0) / cycles,
            "l2_bound": max(c["STALLS_L1D_MISS"] - c["STALLS_L2_MISS"], 0) / cycles,
            "l3_bound": max(c["STALLS_L2_MISS"] - c["STALLS_L3_MISS"], 0) / cycles,
            "dram_bound": c["STALLS_L3_MISS"] / cycles,
            "store_bound": c["BOUND_ON_STORES"] / cycles,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("perf_csv")
    parser.add_argument("driver_json", nargs="?", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    counts = parse_perf_csv(args.perf_csv)
    metrics = tma(counts)

    record = {"counts": counts, "tma": metrics}

    if args.driver_json is not None:
        with open(args.driver_json) as f:
            record["kernel"] = json.load(f)

    print(f"IPC: {metrics['ipc']:.2f}")

    for level in ["L1", "L2", "L3_memory"]:
        print(
            f"{level}: "
            + "  ".join(
                f"{k}={v * 100:.1f}%"
                for k, v in metrics[level].items()
            )
        )

    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
