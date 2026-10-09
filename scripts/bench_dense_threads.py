import argparse
import os
import statistics
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=896)
    parser.add_argument("--n", type=int, default=4864)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    X = torch.randn(args.m, args.k)
    W = torch.randn(args.n, args.k)

    print(f"Threads: {torch.get_num_threads()}")
    print(f"CPU affinity: {sorted(os.sched_getaffinity(0))}")
    print(f"M={args.m}, K={args.k}, N={args.n}")

    for _ in range(args.warmup):
        _ = X @ W.T

    times_ms = []

    for _ in range(args.runs):
        t0 = time.perf_counter()
        _ = X @ W.T
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    median_ms = statistics.median(times_ms)
    mean_ms = statistics.mean(times_ms)

    flops = 2 * args.m * args.n * args.k
    gflops = flops / (median_ms / 1000) / 1e9

    print(f"Median ms: {median_ms:.4f}")
    print(f"Mean ms:   {mean_ms:.4f}")
    print(f"Min ms:    {min(times_ms):.4f}")
    print(f"Max ms:    {max(times_ms):.4f}")
    print(f"GFLOP/s:   {gflops:.2f}")


if __name__ == "__main__":
    main()