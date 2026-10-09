"""
Collect the JSON records under the results directory into small CSV
tables (meant to be committed in results-meta/).

Runs with plain python3 (no torch needed):

    python3 scripts/summarize_results.py \
        --results $STORE/sparse-inference/results --out results-meta
"""

import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict


def load(path):
    with open(path) as f:
        return json.load(f)


def pruning_info(pruning):
    if not pruning:
        return {"method": None, "pattern": None, "actual_sparsity": None, "ppl": None}

    return {
        "method": pruning.get("method"),
        "pattern": pruning.get("pattern"),
        "actual_sparsity": pruning.get("actual_sparsity"),
        "ppl": pruning.get("ppl"),
    }


def write_csv(path, rows):
    if not rows:
        print(f"no rows for {path}")
        return

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{path}: {len(rows)} rows")


def name_of(path, suffix=".json"):
    return os.path.basename(path)[: -len(suffix)]


def summarize_pruning(results):
    rows = []

    for path in sorted(glob.glob(f"{results}/pruning/*.json")):
        r = load(path)
        meta = r.get("meta", {})

        rows.append(
            {
                "name": name_of(path),
                "method": r["method"],
                "pattern": r["pattern"],
                "target_sparsity": r["target_sparsity"],
                "actual_sparsity": r["actual_sparsity"],
                "ppl": r["ppl"],
                "mean_nll": r["mean_nll"],
                "calib_samples": r.get("calib_samples"),
                "seq_len": r["seq_len"],
                "eval_tokens": r["eval_tokens"],
                "full_eval": r["full_eval"],
                "model_revision": r.get("model_revision"),
                "git_commit": meta.get("git_commit"),
            }
        )

    return rows


def summarize_e2e(results):
    rows = []

    for path in sorted(glob.glob(f"{results}/e2e/*.json")):
        r = load(path)
        info = pruning_info(r.get("pruning"))

        for dense, csr in zip(r["dense"], r["csr"]):
            rows.append(
                {
                    "name": name_of(path),
                    **info,
                    "csr_targets": r["csr_targets"],
                    "csr_target_sparsity": r["csr_target_sparsity"],
                    "prompt_len": dense["prompt_len"],
                    "dense_prefill_ms": dense["prefill_ms"],
                    "csr_prefill_ms": csr["prefill_ms"],
                    "prefill_speedup": dense["prefill_ms"] / csr["prefill_ms"],
                    "dense_decode_ms": dense["decode_ms_per_token"],
                    "csr_decode_ms": csr["decode_ms_per_token"],
                    "decode_speedup": (
                        dense["decode_ms_per_token"] / csr["decode_ms_per_token"]
                    ),
                    "check_relative_l2": r["check_relative_l2"],
                    "check_same_next_token": r["check_same_next_token"],
                    "hostname": r.get("meta", {}).get("hostname"),
                }
            )

    return rows


def summarize_layers(results):
    """
    Mean over the benchmarked layers of each (checkpoint, type, M, variant).
    """
    rows = []

    for path in sorted(glob.glob(f"{results}/layers/*.json")):
        r = load(path)
        info = pruning_info(r.get("pruning"))

        groups = defaultdict(list)
        for row in r["rows"]:
            groups[(row["type"], row["M"], row["variant"])].append(row)

        for (layer_type, M, variant), items in sorted(groups.items()):
            rows.append(
                {
                    "name": name_of(path),
                    **info,
                    "type": layer_type,
                    "M": M,
                    "variant": variant,
                    "layers": len(items),
                    "sparsity": statistics.mean(x["sparsity"] for x in items),
                    "median_ms": statistics.mean(x["median_ms"] for x in items),
                    "speedup_vs_dense": statistics.mean(
                        x["speedup_vs_dense"] for x in items
                    ),
                }
            )

    return rows


def summarize_patterns(results):
    rows = []

    for path in sorted(glob.glob(f"{results}/patterns/*.json")):
        r = load(path)
        info = pruning_info(r.get("pruning"))

        groups = defaultdict(list)
        for layer, stats in r["per_layer"].items():
            groups[layer.rsplit(".", 1)[-1]].append(stats)

        for layer_type, items in sorted(groups.items()):
            row = {"name": name_of(path), **info, "type": layer_type, "layers": len(items)}

            for key in items[0]:
                values = [x[key] for x in items if x[key] == x[key]]  # drop NaN
                row[key] = statistics.mean(values) if values else None

            rows.append(row)

    return rows


def summarize_tma(results):
    rows = []

    for path in sorted(glob.glob(f"{results}/tma/**/tma.json", recursive=True)):
        r = load(path)
        k = r.get("kernel", {})
        t = r["tma"]

        rows.append(
            {
                "run": os.path.relpath(os.path.dirname(path), f"{results}/tma"),
                "model_path": k.get("model_path"),
                "layer": k.get("layer"),
                "shape": "x".join(map(str, k.get("shape", []))),
                "sparsity": k.get("sparsity"),
                "format": k.get("format"),
                "m": k.get("m"),
                "threads": k.get("threads"),
                "ms_per_call": k.get("ms_per_call"),
                "ipc": t["ipc"],
                **t["L1"],
                **t["L2"],
                **t["L3_memory"],
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=os.path.expandvars("$STORE/sparse-inference/results"),
    )
    parser.add_argument("--out", default="results-meta")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    write_csv(f"{args.out}/ppl_sweep.csv", summarize_pruning(args.results))
    write_csv(f"{args.out}/e2e_dense_vs_csr.csv", summarize_e2e(args.results))
    write_csv(f"{args.out}/layer_formats_summary.csv", summarize_layers(args.results))
    write_csv(f"{args.out}/sparsity_patterns.csv", summarize_patterns(args.results))
    write_csv(f"{args.out}/tma.csv", summarize_tma(args.results))


if __name__ == "__main__":
    main()
