import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def median_ms(values):
    return statistics.median(values) * 1000.0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
    )

    model.eval()

    # Build a deterministic prompt with approximately the requested length.
    base_text = (
        "Sparse neural network inference is an important problem "
        "because pruning reduces the number of nonzero weights, "
        "but efficient hardware execution depends on representation "
        "and kernel behavior. "
    )

    text = base_text
    while True:
        ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]

        if ids.shape[1] >= args.prompt_len:
            break

        text += base_text

    input_ids = ids[:, :args.prompt_len]

    print()
    print("===== Configuration =====")
    print(f"Threads:       {args.threads}")
    print(f"Prompt length: {input_ids.shape[1]}")
    print(f"Decode tokens: {args.decode_tokens}")

    # --------------------------------------------------
    # Prefill benchmark
    # --------------------------------------------------

    def prefill():
        with torch.inference_mode():
            return model(
                input_ids=input_ids,
                use_cache=True,
            )

    for _ in range(args.warmup):
        prefill()

    prefill_times = []

    for _ in range(args.runs):
        t0 = time.perf_counter()
        prefill()
        t1 = time.perf_counter()

        prefill_times.append(t1 - t0)

    prefill_median = median_ms(prefill_times)

    print()
    print("===== Prefill =====")
    print(f"Median latency: {prefill_median:.3f} ms")
    print(
        f"Throughput:     "
        f"{input_ids.shape[1] / (prefill_median / 1000.0):.2f} tokens/s"
    )

    # --------------------------------------------------
    # Decode benchmark
    # --------------------------------------------------

    # First run the prompt once to create KV cache.
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
        )

    past_key_values = outputs.past_key_values

    next_token = outputs.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    # Warm up decode path.
    warmup_cache = past_key_values
    warmup_token = next_token

    for _ in range(args.warmup):
        with torch.inference_mode():
            out = model(
                input_ids=warmup_token,
                past_key_values=warmup_cache,
                use_cache=True,
            )

        warmup_cache = out.past_key_values
        warmup_token = out.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

    # Recreate clean KV cache before timing.
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
        )

    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    decode_times = []

    with torch.inference_mode():
        for _ in range(args.decode_tokens):
            t0 = time.perf_counter()

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )

            t1 = time.perf_counter()

            decode_times.append(t1 - t0)

            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

    decode_median_ms = median_ms(decode_times)

    total_decode_time = sum(decode_times)

    print()
    print("===== Decode =====")
    print(f"Tokens generated: {args.decode_tokens}")
    print(f"Median ms/token:  {decode_median_ms:.3f}")
    print(
        f"Tokens/s:         "
        f"{args.decode_tokens / total_decode_time:.2f}"
    )


if __name__ == "__main__":
    main()
