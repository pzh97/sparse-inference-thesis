import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparse_inference.data import evaluate_ppl
from sparse_inference.results import model_revision, write_json


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON file for the result record",
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

    mean_nll, ppl, num_chunks, total_target_tokens = evaluate_ppl(
        model,
        tokenizer,
        args.seq_len,
        args.max_samples,
    )

    print()
    print("===== Result =====")
    print(f"Model:         {args.model}")
    print(f"Sequence len:  {args.seq_len}")
    print(f"Chunks:        {num_chunks}")
    print(f"Target tokens: {total_target_tokens:,}")
    print(f"Mean NLL:      {mean_nll:.6f}")
    print(f"Perplexity:    {ppl:.4f}")

    write_json(
        args.output,
        {
            "experiment": "pruning",
            "method": "dense",
            "args": vars(args),
            "model_revision": model_revision(model),
            "pattern": "dense",
            "target_sparsity": 0.0,
            "actual_sparsity": 0.0,
            "seq_len": args.seq_len,
            "eval_chunks": num_chunks,
            "eval_tokens": total_target_tokens,
            "full_eval": args.max_samples is None,
            "mean_nll": mean_nll,
            "ppl": ppl,
        },
    )


if __name__ == "__main__":
    main()
