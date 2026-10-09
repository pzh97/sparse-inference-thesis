import argparse
import math
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from result_io import model_revision, write_json


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

    print("Loading WikiText-2...")
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    text = "\n\n".join(dataset["text"])

    print("Tokenizing...")
    encodings = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encodings.input_ids

    print(f"Total tokens: {input_ids.numel():,}")
    print(f"Sequence length: {args.seq_len}")

    nlls = []
    total_target_tokens = 0

    num_chunks = input_ids.size(1) // args.seq_len

    if args.max_samples is not None:
        num_chunks = min(num_chunks, args.max_samples)

    print(f"Evaluating {num_chunks} chunks...")

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * args.seq_len
            end = start + args.seq_len

            batch = input_ids[:, start:end]

            outputs = model(
                input_ids=batch,
                labels=batch,
                use_cache=False,
            )

            # Hugging Face causal LM loss is mean cross-entropy
            # over predicted tokens in this chunk.
            valid_tokens = batch.numel() - 1

            nlls.append(outputs.loss.item() * valid_tokens)
            total_target_tokens += valid_tokens

            if (i + 1) % 20 == 0:
                current_nll = sum(nlls)
                current_ppl = math.exp(
                    current_nll / total_target_tokens
                )

                print(
                    f"[{i + 1}/{num_chunks}] "
                    f"running PPL = {current_ppl:.4f}"
                )

    mean_nll = sum(nlls) / total_target_tokens
    ppl = math.exp(mean_nll)

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