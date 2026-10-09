"""
Steady-state kernel driver for Top-down Microarchitecture Analysis.

Runs a single dense or CSR matmul in a loop so that an external
`perf stat -p <pid>` (see run_tma.sh) measures only the kernel and not
model loading, tokenization or warmup.

The weight is either a real Linear of a pruned checkpoint
(--model-path + --layer) or a random matrix (--shape N K --sparsity s).

Protocol:
    1. setup + warmup
    2. write our PID to --ready-file
    3. loop for --seconds
    4. print calls/s so perf counts can be normalized per call
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F


def load_weight(args):
    if args.model_path is not None:
        from safetensors import safe_open

        path = os.path.join(args.model_path, "model.safetensors")

        with safe_open(path, framework="pt") as f:
            return f.get_tensor(f"{args.layer}.weight").float().contiguous()

    n, k = args.shape

    generator = torch.Generator()
    generator.manual_seed(0)

    W = torch.randn(n, k, generator=generator)
    W[torch.rand(n, k, generator=generator) < args.sparsity] = 0.0

    return W


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--layer",
        default="model.layers.12.mlp.down_proj",
        help="Linear name inside the checkpoint",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=[4864, 896],
        metavar=("N", "K"),
        help="Random weight shape when no checkpoint is given",
    )
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument(
        "--format",
        choices=["dense", "csr", "csr_rowmajor"],
        required=True,
    )
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--ready-file", default=None)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    W = load_weight(args)
    n, k = W.shape

    x = torch.randn(args.m, k)
    xT = x.T.contiguous()
    W_csr = W.to_sparse_csr()

    if args.format == "dense":
        fn = lambda: F.linear(x, W)
    elif args.format == "csr":
        fn = lambda: torch.sparse.mm(W_csr, x.T).T
    else:
        fn = lambda: torch.sparse.mm(W_csr, xT)

    sparsity = 1.0 - (W != 0).sum().item() / W.numel()

    with torch.inference_mode():

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.warmup:
            fn()

        if args.ready_file is not None:
            with open(args.ready_file, "w") as f:
                f.write(f"{os.getpid()}\n")

        calls = 0
        t0 = time.perf_counter()

        while time.perf_counter() - t0 < args.seconds:
            fn()
            calls += 1

        elapsed = time.perf_counter() - t0

    result = {
        "format": args.format,
        "layer": args.layer if args.model_path else None,
        "model_path": args.model_path,
        "shape": [n, k],
        "sparsity": sparsity,
        "m": args.m,
        "threads": args.threads,
        "calls": calls,
        "seconds": elapsed,
        "calls_per_s": calls / elapsed,
        "ms_per_call": 1000.0 * elapsed / calls,
    }

    print(json.dumps(result))

    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
