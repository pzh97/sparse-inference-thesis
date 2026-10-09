import argparse
import os

import torch

from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_qwen_csr_inference import CSR_IMPLS, TARGET_GROUPS, replace_linears_with_csr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B",
        help="Hub id or path to a pruned checkpoint",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument(
        "--csr-targets",
        choices=["none", *sorted(TARGET_GROUPS)],
        default="none",
        help="Convert these Linear modules to CSR before profiling",
    )
    parser.add_argument(
        "--csr-impl",
        choices=CSR_IMPLS,
        default="mlp_t",
    )
    parser.add_argument(
        "--mode",
        choices=["prefill", "decode"],
        default="prefill",
        help="decode profiles --decode-tokens single-token steps",
    )
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument(
        "--output",
        default=None,
        help="Write the operator table to this text file",
    )
    parser.add_argument(
        "--trace",
        default=None,
        help="Write a Chrome trace (open in chrome://tracing or Perfetto)",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
    )
    model.eval()

    if args.csr_targets != "none":
        replace_linears_with_csr(
            model,
            TARGET_GROUPS[args.csr_targets],
            args.csr_impl,
        )

    base_text = (
        "Sparse neural network inference is an important problem "
        "because pruning reduces the number of nonzero weights, "
        "but efficient execution depends on representation and kernels. "
    )

    text = base_text
    while True:
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]

        if input_ids.shape[1] >= args.prompt_len:
            break

        text += base_text

    input_ids = input_ids[:, :args.prompt_len]

    def run_prefill():
        return model(
            input_ids=input_ids,
            use_cache=args.mode == "decode",
        )

    def run_decode(outputs):
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        for _ in range(args.decode_tokens):
            outputs = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

    # Warmup
    with torch.inference_mode():
        for _ in range(3):
            outputs = run_prefill()
            if args.mode == "decode":
                run_decode(outputs)

    print(
        f"Profiling {args.mode}, prompt length = {args.prompt_len}, "
        f"CSR targets = {args.csr_targets} ({args.csr_impl})"
    )

    with torch.inference_mode():
        if args.mode == "decode":
            # The prefill is not profiled.
            outputs = run_prefill()

        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            if args.mode == "decode":
                run_decode(outputs)
            else:
                run_prefill()

    table = prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=args.row_limit,
    )

    shapes_table = prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cpu_time_total",
        row_limit=args.row_limit,
    )

    print()
    print("===== Top operators by self CPU time =====")
    print(table)

    print()
    print("===== Top operators by self CPU time, grouped by input shape =====")
    print(shapes_table)

    if args.output is not None:
        path = os.path.expandvars(args.output)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        with open(path, "w") as f:
            f.write(f"{vars(args)}\n\n")
            f.write(table)
            f.write("\n\n")
            f.write(shapes_table)

        print(f"Table written to {path}")

    if args.trace is not None:
        path = os.path.expandvars(args.trace)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        prof.export_chrome_trace(path)
        print(f"Trace written to {path}")


if __name__ == "__main__":
    main()
