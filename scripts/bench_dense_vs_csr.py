import argparse
import statistics
import time

import torch


def benchmark(fn, warmup, runs):
    for _ in range(warmup):
        fn()

    times_ms = []

    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    return {
        "median": statistics.median(times_ms),
        "mean": statistics.mean(times_ms),
        "min": min(times_ms),
        "max": max(times_ms),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=896)
    parser.add_argument("--n", type=int, default=4864)
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)

    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    X = torch.randn(args.m, args.k)
    W = torch.randn(args.n, args.k)

    # Random unstructured pruning mask.
    mask = torch.rand_like(W) >= args.sparsity
    W = W * mask

    actual_sparsity = (W == 0).float().mean().item()

    W_csr = W.to_sparse_csr()

    # Dense execution
    dense_result = benchmark(
        lambda: X @ W.T,
        args.warmup,
        args.runs,
    )

    # CSR execution.
    #
    # W has shape [N, K], therefore W @ X.T gives [N, M].
    # Transpose it back to [M, N] so that it matches X @ W.T.
    csr_result = benchmark(
        lambda: torch.sparse.mm(W_csr, X.T).T,
        args.warmup,
        args.runs,
    )

    # Correctness check
    Y_dense = X @ W.T
    Y_csr = torch.sparse.mm(W_csr, X.T).T

    diff = (Y_dense - Y_csr).abs()

    max_abs_error = diff.max().item()

    relative_l2_error = (
        torch.linalg.vector_norm(Y_dense - Y_csr)
        / torch.linalg.vector_norm(Y_dense)
    ).item()

    correct = relative_l2_error < 1e-5

    speedup = dense_result["median"] / csr_result["median"]

    print(f"Sparsity requested: {args.sparsity:.2f}")
    print(f"Sparsity actual:    {actual_sparsity:.4f}")
    print(f"NNZ:                {W_csr.values().numel()}")
    print(f"Correct:            {correct}")
    print(f"Max abs error:      {max_abs_error:.8e}")
    print(f"Relative L2 error:  {relative_l2_error:.8e}")
    print()

    print(f"Dense median ms:    {dense_result['median']:.4f}")
    print(f"Dense mean ms:      {dense_result['mean']:.4f}")
    print(f"CSR median ms:      {csr_result['median']:.4f}")
    print(f"CSR mean ms:        {csr_result['mean']:.4f}")
    print(f"CSR speedup:        {speedup:.3f}x")


if __name__ == "__main__":
    main()