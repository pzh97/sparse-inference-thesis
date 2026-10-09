import argparse
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparse_inference.csr import CSR_IMPLS, TARGET_GROUPS, replace_linears_with_csr
from sparse_inference.data import make_prompt_tokens
from sparse_inference.results import read_json, write_json


# Prefill and decode only need the logits of the last position; computing
# lm_head for every prompt token (the HF default) adds a dense
# [prompt_len x 896] x [896 x 151936] matmul that is not part of real
# inference and dilutes the CSR/dense comparison.
LOGITS_TO_KEEP = 1


# ============================================================
# Synchronization
# ============================================================

def synchronize():
    """
    CPU-only benchmark.
    Kept as a helper so timing code stays explicit.
    """
    pass


# ============================================================
# Correctness
# ============================================================

def check_prefill_correctness(
    dense_model,
    csr_model,
    input_ids,
):
    dense_model.eval()
    csr_model.eval()

    print()
    print(
        "Checking dense-vs-CSR numerical consistency..."
    )

    with torch.inference_mode():

        dense_out = dense_model(
            input_ids=input_ids,
            use_cache=False,
        ).logits

        csr_out = csr_model(
            input_ids=input_ids,
            use_cache=False,
        ).logits

    diff = (
        dense_out
        - csr_out
    )

    max_abs = (
        diff
        .abs()
        .max()
        .item()
    )

    relative_l2 = (
        torch.linalg.vector_norm(
            diff
        )
        /
        torch.linalg.vector_norm(
            dense_out
        )
    ).item()

    dense_next = (
        dense_out[:, -1, :]
        .argmax(dim=-1)
        .item()
    )

    csr_next = (
        csr_out[:, -1, :]
        .argmax(dim=-1)
        .item()
    )

    print(
        f"max abs error:      "
        f"{max_abs:.6e}"
    )

    print(
        f"relative L2 error:  "
        f"{relative_l2:.6e}"
    )

    print(
        f"dense next token:   "
        f"{dense_next}"
    )

    print(
        f"CSR next token:     "
        f"{csr_next}"
    )

    print(
        f"same next token:    "
        f"{dense_next == csr_next}"
    )

    return (
        max_abs,
        relative_l2,
        dense_next == csr_next,
    )


# ============================================================
# Prefill benchmark
# ============================================================

def benchmark_prefill(
    model,
    input_ids,
    warmup=3,
    repeats=10,
):
    model.eval()

    with torch.inference_mode():

        for _ in range(warmup):
            model(
                input_ids=input_ids,
                use_cache=True,
                logits_to_keep=LOGITS_TO_KEEP,
            )

        times = []

        for _ in range(repeats):

            synchronize()

            t0 = time.perf_counter()

            model(
                input_ids=input_ids,
                use_cache=True,
                logits_to_keep=LOGITS_TO_KEEP,
            )

            synchronize()

            t1 = time.perf_counter()

            times.append(
                (t1 - t0) * 1000.0
            )

    median_ms = statistics.median(
        times
    )

    prompt_len = (
        input_ids.shape[1]
    )

    throughput = (
        prompt_len
        / (median_ms / 1000.0)
    )

    return (
        median_ms,
        throughput,
        times,
    )


# ============================================================
# Decode benchmark
# ============================================================

def decode_once(
    model,
    input_ids,
    decode_tokens,
):
    """
    Prefill once with cache, then autoregressive decode.

    We time decode steps only.
    """

    model.eval()

    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=LOGITS_TO_KEEP,
        )

        past_key_values = (
            outputs.past_key_values
        )

        next_token = (
            outputs.logits[
                :,
                -1,
                :
            ]
            .argmax(
                dim=-1,
                keepdim=True,
            )
        )

        times = []

        for _ in range(
            decode_tokens
        ):

            synchronize()

            t0 = time.perf_counter()

            outputs = model(
                input_ids=next_token,
                past_key_values=
                    past_key_values,
                use_cache=True,
            )

            synchronize()

            t1 = time.perf_counter()

            times.append(
                (t1 - t0)
                * 1000.0
            )

            past_key_values = (
                outputs.past_key_values
            )

            next_token = (
                outputs.logits[
                    :,
                    -1,
                    :
                ]
                .argmax(
                    dim=-1,
                    keepdim=True,
                )
            )

    return times


def benchmark_decode(
    model,
    input_ids,
    decode_tokens=32,
    warmup_runs=1,
    repeats=3,
):
    # Warmup complete decode sequences
    for _ in range(
        warmup_runs
    ):
        decode_once(
            model,
            input_ids,
            decode_tokens,
        )

    all_step_times = []

    sequence_medians = []

    for _ in range(
        repeats
    ):

        step_times = (
            decode_once(
                model,
                input_ids,
                decode_tokens,
            )
        )

        all_step_times.extend(
            step_times
        )

        sequence_medians.append(
            statistics.median(
                step_times
            )
        )

    median_ms_per_token = (
        statistics.median(
            sequence_medians
        )
    )

    throughput = (
        1000.0
        / median_ms_per_token
    )

    return (
        median_ms_per_token,
        throughput,
        all_step_times,
    )


# ============================================================
# One model benchmark
# ============================================================

def benchmark_model(
    label,
    model,
    tokenizer,
    prompt_lengths,
    decode_tokens,
    prefill_warmup,
    prefill_repeats,
    decode_repeats,
):
    results = []

    print()
    print(
        "=" * 72
    )
    print(
        f"Benchmarking: {label}"
    )
    print(
        "=" * 72
    )

    for prompt_len in (
        prompt_lengths
    ):

        input_ids = (
            make_prompt_tokens(
                tokenizer,
                prompt_len,
            )
        )

        print()
        print(
            f"Prompt length: "
            f"{prompt_len}"
        )

        (
            prefill_ms,
            prefill_tps,
            _,
        ) = benchmark_prefill(
            model,
            input_ids,
            warmup=prefill_warmup,
            repeats=prefill_repeats,
        )

        print(
            f"Prefill median: "
            f"{prefill_ms:.3f} ms"
        )

        print(
            f"Prefill throughput: "
            f"{prefill_tps:.2f} tok/s"
        )

        (
            decode_ms,
            decode_tps,
            _,
        ) = benchmark_decode(
            model,
            input_ids,
            decode_tokens=decode_tokens,
            warmup_runs=1,
            repeats=decode_repeats,
        )

        print(
            f"Decode median: "
            f"{decode_ms:.3f} "
            f"ms/token"
        )

        print(
            f"Decode throughput: "
            f"{decode_tps:.2f} tok/s"
        )

        results.append(
            {
                "variant": label,
                "prompt_len":
                    prompt_len,
                "prefill_ms":
                    prefill_ms,
                "prefill_tps":
                    prefill_tps,
                "decode_ms_per_token":
                    decode_ms,
                "decode_tps":
                    decode_tps,
            }
        )

    return results


# ============================================================
# Summary
# ============================================================

def print_summary(
    dense_results,
    csr_results,
):
    print()
    print(
        "=" * 100
    )
    print(
        "SUMMARY"
    )
    print(
        "=" * 100
    )

    print(
        f"{'prompt':>8} "
        f"{'dense prefill':>15} "
        f"{'CSR prefill':>15} "
        f"{'CSR/dense':>12} "
        f"{'dense decode':>15} "
        f"{'CSR decode':>15} "
        f"{'CSR/dense':>12}"
    )

    print(
        "-" * 100
    )

    for dense, csr in zip(
        dense_results,
        csr_results,
    ):

        prefill_speedup = (
            dense[
                "prefill_ms"
            ]
            /
            csr[
                "prefill_ms"
            ]
        )

        decode_speedup = (
            dense[
                "decode_ms_per_token"
            ]
            /
            csr[
                "decode_ms_per_token"
            ]
        )

        print(
            f"{dense['prompt_len']:8d} "
            f"{dense['prefill_ms']:15.3f} "
            f"{csr['prefill_ms']:15.3f} "
            f"{prefill_speedup:12.3f} "
            f"{dense['decode_ms_per_token']:15.3f} "
            f"{csr['decode_ms_per_token']:15.3f} "
            f"{decode_speedup:12.3f}"
        )

    print()

    print(
        "Speedup > 1 means CSR is faster."
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=[
            128,
            512,
        ],
    )

    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--prefill-warmup",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--prefill-repeats",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--decode-repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--targets",
        choices=sorted(TARGET_GROUPS),
        default="mlp",
        help="Which Linear modules are converted to CSR",
    )

    parser.add_argument(
        "--csr-impl",
        choices=CSR_IMPLS,
        default="mlp_t",
        help="v0 = 2026-10-08 sweep (int64 indices, per-Linear)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON file for the result record",
    )

    args = parser.parse_args()

    torch.set_num_threads(
        args.threads
    )

    torch.manual_seed(0)

    print(
        f"PyTorch threads: "
        f"{torch.get_num_threads()}"
    )

    print(
        f"Loading tokenizer from "
        f"{args.model_path}..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model_path
        )
    )

    print(
        "Loading dense-pruned model..."
    )

    dense_model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.float32,
        )
    )

    dense_model.eval()

    print(
        "Loading CSR model copy..."
    )

    # Load again instead of deepcopy.
    # This is slower to initialize but avoids sharing weird internal
    # state and gives a clean apples-to-apples model.
    csr_model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.float32,
        )
    )

    csr_model.eval()

    suffixes = TARGET_GROUPS[
        args.targets
    ]

    (
        num_replaced,
        csr_sparsity,
    ) = replace_linears_with_csr(
        csr_model,
        suffixes,
        args.csr_impl,
    )

    expected = (
        dense_model.config.num_hidden_layers
        * len(suffixes)
    )

    print(
        f"CSR modules replaced: "
        f"{num_replaced}"
    )

    if num_replaced != expected:
        print(
            f"WARNING: expected "
            f"{expected} replaced "
            f"modules."
        )

    # --------------------------------------------------------
    # Numerical correctness check
    # --------------------------------------------------------

    check_ids = (
        make_prompt_tokens(
            tokenizer,
            prompt_len=32,
        )
    )

    (
        max_abs,
        relative_l2,
        same_token,
    ) = check_prefill_correctness(
        dense_model,
        csr_model,
        check_ids,
    )

    print()

    if relative_l2 > 1e-4:
        print(
            "WARNING: relative L2 error "
            "is larger than expected."
        )

    if not same_token:
        print(
            "WARNING: dense and CSR "
            "predict different next tokens."
        )

    # --------------------------------------------------------
    # Dense benchmark first
    # --------------------------------------------------------

    dense_results = benchmark_model(
        label="dense-pruned",
        model=dense_model,
        tokenizer=tokenizer,
        prompt_lengths=
            args.prompt_lengths,
        decode_tokens=
            args.decode_tokens,
        prefill_warmup=
            args.prefill_warmup,
        prefill_repeats=
            args.prefill_repeats,
        decode_repeats=
            args.decode_repeats,
    )

    # Free the dense model before CSR benchmarking if desired?
    # Keep both loaded so the same process/environment is used.
    # On this 0.5B model this should fit comfortably.

    csr_results = benchmark_model(
        label=f"csr-{args.csr_impl}-{args.targets}",
        model=csr_model,
        tokenizer=tokenizer,
        prompt_lengths=
            args.prompt_lengths,
        decode_tokens=
            args.decode_tokens,
        prefill_warmup=
            args.prefill_warmup,
        prefill_repeats=
            args.prefill_repeats,
        decode_repeats=
            args.decode_repeats,
    )

    print_summary(
        dense_results,
        csr_results,
    )

    pruning_meta_path = os.path.join(
        args.model_path,
        "pruning_meta.json",
    )

    write_json(
        args.output,
        {
            "experiment": "e2e_dense_vs_csr",
            "args": vars(args),
            "pruning": (
                read_json(pruning_meta_path)
                if os.path.exists(pruning_meta_path)
                else None
            ),
            "csr_targets": args.targets,
            "csr_impl": args.csr_impl,
            "logits_to_keep": LOGITS_TO_KEEP,
            "csr_modules": num_replaced,
            "csr_target_sparsity": csr_sparsity,
            "check_max_abs": max_abs,
            "check_relative_l2": relative_l2,
            "check_same_next_token": same_token,
            "dense": dense_results,
            "csr": csr_results,
        },
    )


if __name__ == "__main__":
    main()