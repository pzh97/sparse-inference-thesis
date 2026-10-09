"""
Timing stability check for the dense baseline.

Measures, in one fresh process:
    - dense F.linear for the MLP shapes at several M
    - full-model prefill
and prints one line per measurement, so that several processes (with
different OpenMP settings) can be compared. See slurm/stability.sbatch.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from bench_qwen_csr_inference import make_prompt_tokens


def median_ms(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--label", default="")
    parser.add_argument("--m-values", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--prompt-len", type=int, default=512)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    result = {
        "label": args.label,
        "omp": {
            k: os.environ.get(k)
            for k in ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "KMP_AFFINITY"]
        },
    }

    with torch.inference_mode():
        for name, (n, k) in {"gate_proj": (4864, 896), "down_proj": (896, 4864)}.items():
            W = torch.randn(n, k)
            for M in args.m_values:
                x = torch.randn(M, k)
                result[f"{name}_m{M}"] = median_ms(lambda: F.linear(x, W), 20, 200)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B", dtype=torch.float32
        ).eval()
        ids = make_prompt_tokens(tokenizer, args.prompt_len)
        result[f"prefill{args.prompt_len}"] = median_ms(
            lambda: model(input_ids=ids, use_cache=False), 3, 10
        )

    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
