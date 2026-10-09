"""
Per-layer dense vs CSR benchmark on the real matrices of a pruned
checkpoint.

For every Transformer Linear (or a subset) and every activation row
count M it times:

    dense        F.linear(x, W)                  (what nn.Linear runs)
    csr          sparse.mm(W_csr, x.T).T         (what CSRLinear runs)
    csr_rowmajor sparse.mm(W_csr, xT)            (x already transposed and
                                                  contiguous; isolates the
                                                  layout cost of csr)
    csr_random   csr on a matrix with the same shape and nnz but uniformly
                 random positions (control: effect of the pruning pattern)

M = 1 corresponds to decode, larger M to prefill with M prompt tokens.

Output: one CSV row per (layer, M, variant) plus a JSON record with the
same rows and the run metadata.
"""

import argparse
import csv
import os
import statistics
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from analyze_sparsity_patterns import analyze_matrix, layer_type_from_name
from result_io import read_json, write_json


def time_fn(fn, warmup, runs, min_time_s):
    for _ in range(warmup):
        fn()

    times_ms = []
    start = time.perf_counter()

    while (
        len(times_ms) < runs
        or time.perf_counter() - start < min_time_s
    ):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return {
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "min_ms": min(times_ms),
        "runs": len(times_ms),
    }


def random_same_nnz(W, generator):
    """
    Same shape and nnz as W, nonzeros at uniformly random positions.
    """
    values = W[W != 0]

    positions = torch.randperm(W.numel(), generator=generator)[
        : values.numel()
    ]

    R = torch.zeros(W.numel(), dtype=W.dtype)
    R[positions] = values[
        torch.randperm(values.numel(), generator=generator)
    ]

    return R.view_as(W)


def layer_index(name):
    # model.layers.<idx>.<block>.<proj>
    return int(name.split(".")[2])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 512],
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Transformer layer indices (default: all)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="Projection types, e.g. q_proj down_proj (default: all)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["dense", "csr", "csr_rowmajor", "csr_random"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument(
        "--min-time",
        type=float,
        default=0.2,
        help="Minimum seconds of timing per measurement",
    )
    parser.add_argument("--no-pattern-stats", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Loading {args.model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float32,
    )
    model.eval()

    targets = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.startswith("model.layers."):
            continue
        if args.layers is not None and layer_index(name) not in args.layers:
            continue
        if args.types is not None and layer_type_from_name(name) not in args.types:
            continue

        targets.append((name, module))

    print(f"Benchmarking {len(targets)} Linear modules")

    rows = []

    with torch.inference_mode():

        for i, (name, module) in enumerate(targets):

            W = module.weight.detach().contiguous()
            out_features, in_features = W.shape

            nnz = int((W != 0).sum().item())
            sparsity = 1.0 - nnz / W.numel()

            W_csr = W.to_sparse_csr()
            W_rand_csr = (
                random_same_nnz(W, generator).to_sparse_csr()
                if "csr_random" in args.variants
                else None
            )

            stats = (
                {}
                if args.no_pattern_stats
                else analyze_matrix(W)
            )

            for M in args.m_values:

                x = torch.randn(M, in_features, generator=generator)
                xT = x.T.contiguous()

                fns = {
                    "dense": lambda: F.linear(x, W),
                    "csr": lambda: torch.sparse.mm(W_csr, x.T).T,
                    "csr_rowmajor": lambda: torch.sparse.mm(W_csr, xT),
                    "csr_random": lambda: torch.sparse.mm(W_rand_csr, x.T).T,
                }

                # Correctness of the CSR path on the real matrix.
                ref = F.linear(x, W)
                rel_err = (
                    torch.linalg.vector_norm(
                        torch.sparse.mm(W_csr, x.T).T - ref
                    )
                    / torch.linalg.vector_norm(ref)
                ).item()

                results = {
                    variant: time_fn(
                        fns[variant],
                        args.warmup,
                        args.runs,
                        args.min_time,
                    )
                    for variant in args.variants
                }

                dense_ms = results.get("dense", {}).get("median_ms")

                for variant, r in results.items():
                    row = {
                        "layer": name,
                        "layer_idx": layer_index(name),
                        "type": layer_type_from_name(name),
                        "out_features": out_features,
                        "in_features": in_features,
                        "nnz": nnz,
                        "sparsity": sparsity,
                        "M": M,
                        "variant": variant,
                        **r,
                        "speedup_vs_dense": (
                            dense_ms / r["median_ms"]
                            if dense_ms is not None
                            else None
                        ),
                        "csr_rel_err": rel_err,
                    }
                    row.update(stats)
                    rows.append(row)

                summary = " ".join(
                    f"{v}={r['median_ms']:.4f}ms"
                    for v, r in results.items()
                )
                print(
                    f"[{i + 1}/{len(targets)}] {name} "
                    f"s={sparsity:.3f} M={M}: {summary}"
                )

    if args.csv is not None and rows:
        path = os.path.expandvars(args.csv)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"CSV written to {path}")

    pruning_meta_path = os.path.join(args.model_path, "pruning_meta.json")

    write_json(
        args.output,
        {
            "experiment": "layer_formats",
            "args": vars(args),
            "pruning": (
                read_json(pruning_meta_path)
                if os.path.exists(pruning_meta_path)
                else None
            ),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
