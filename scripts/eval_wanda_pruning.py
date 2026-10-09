import argparse
import math
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruning_patterns import build_mask, parse_pattern, pattern_sparsity
from result_io import model_revision, write_json


# ---------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------

def get_calibration_samples(
    tokenizer,
    nsamples=32,
    seq_len=2048,
    seed=0,
):
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )

    text = "\n\n".join(dataset["text"])

    tokens = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        verbose=False,
    ).input_ids

    if tokens.shape[1] <= seq_len:
        raise ValueError(
            f"Calibration corpus too short: "
            f"{tokens.shape[1]} tokens"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    max_start = (
        tokens.shape[1]
        - seq_len
        - 1
    )

    samples = []

    for _ in range(nsamples):

        start = torch.randint(
            low=0,
            high=max_start,
            size=(1,),
            generator=generator,
        ).item()

        samples.append(
            tokens[
                :,
                start:start + seq_len
            ]
        )

    return samples


# ---------------------------------------------------------------------
# Target Linear modules
# ---------------------------------------------------------------------

def get_target_linears(model):

    modules = {}

    for name, module in model.named_modules():

        if not isinstance(
            module,
            torch.nn.Linear,
        ):
            continue

        if not name.startswith(
            "model.layers."
        ):
            continue

        modules[name] = module

    return modules


# ---------------------------------------------------------------------
# Wanda activation statistics
# ---------------------------------------------------------------------

def collect_activation_stats(
    model,
    calibration_samples,
):
    linears = get_target_linears(model)

    stats = {}

    for name, module in linears.items():

        stats[name] = torch.zeros(
            module.in_features,
            dtype=torch.float64,
        )

    hooks = []

    for name, module in linears.items():

        def make_hook(layer_name):

            def hook(
                module,
                inputs,
                output,
            ):
                x = inputs[0].detach()

                x = x.reshape(
                    -1,
                    x.shape[-1],
                )

                stats[layer_name] += (
                    x.double()
                    .pow(2)
                    .sum(dim=0)
                    .cpu()
                )

            return hook

        hooks.append(
            module.register_forward_hook(
                make_hook(name)
            )
        )

    print(
        f"Collecting activation statistics "
        f"from {len(calibration_samples)} samples..."
    )

    model.eval()

    with torch.inference_mode():

        for i, sample in enumerate(
            calibration_samples
        ):

            model(
                input_ids=sample,
                use_cache=False,
            )

            print(
                f"Calibration "
                f"{i + 1}/"
                f"{len(calibration_samples)}",
                end="\r",
                flush=True,
            )

    print()

    for hook in hooks:
        hook.remove()

    return stats


# ---------------------------------------------------------------------
# Wanda pruning
# ---------------------------------------------------------------------

def prune_wanda(
    model,
    stats,
    sparsity,
    pattern="unstructured",
):
    total_weights = 0
    total_zeros = 0

    linears = get_target_linears(model)

    print()
    print("Applying Wanda pruning...")

    for i, (
        name,
        module,
    ) in enumerate(
        linears.items()
    ):

        W = module.weight.data

        activation_scale = torch.sqrt(
            stats[name]
        ).to(
            device=W.device,
            dtype=W.dtype,
        )

        # Wanda metric:
        #
        # |W_ij| * sqrt(sum_t x_tj^2)
        metric = (
            W.abs()
            * activation_scale.unsqueeze(0)
        )

        mask = build_mask(
            metric,
            pattern,
            sparsity,
            unstructured_scope="row",
        )

        W[mask] = 0.0

        print(
            f"[{i + 1}/{len(linears)}] "
            f"{name}"
        )

    for module in linears.values():

        W = module.weight.data

        total_weights += W.numel()

        total_zeros += (
            W == 0
        ).sum().item()

    actual_sparsity = (
        total_zeros
        / total_weights
    )

    print()
    print("===== Wanda pruning =====")

    print(
        f"Target sparsity:       "
        f"{sparsity:.4f}"
    )

    print(
        f"Actual sparsity:       "
        f"{actual_sparsity:.4f}"
    )

    print(
        f"Pruned weights:        "
        f"{total_zeros:,}"
    )

    print(
        f"Total target weights:  "
        f"{total_weights:,}"
    )

    return actual_sparsity


# ---------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------

def save_model(
    model,
    tokenizer,
    save_path,
):
    if save_path is None:
        return

    save_path = os.path.expandvars(
        os.path.expanduser(
            save_path
        )
    )

    os.makedirs(
        save_path,
        exist_ok=True,
    )

    print()
    print(
        f"Saving pruned model to "
        f"{save_path}..."
    )

    model.save_pretrained(
        save_path,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        save_path
    )

    print("Save complete.")


# ---------------------------------------------------------------------
# WikiText-2 perplexity
# ---------------------------------------------------------------------

def evaluate_ppl(
    model,
    tokenizer,
    seq_len,
    max_samples=None,
):
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    text = "\n\n".join(
        dataset["text"]
    )

    tokens = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        verbose=False,
    ).input_ids

    num_chunks = (
        tokens.shape[1]
        // seq_len
    )

    if max_samples is not None:
        num_chunks = min(
            num_chunks,
            max_samples,
        )

    print()
    print(
        f"Evaluating "
        f"{num_chunks} chunks..."
    )

    total_nll = 0.0
    total_target_tokens = 0

    model.eval()

    with torch.inference_mode():

        for i in range(num_chunks):

            start = (
                i * seq_len
            )

            batch = tokens[
                :,
                start:start + seq_len
            ]

            outputs = model(
                input_ids=batch,
                labels=batch,
                use_cache=False,
            )

            valid_tokens = (
                batch.numel()
                - 1
            )

            total_nll += (
                outputs.loss.item()
                * valid_tokens
            )

            total_target_tokens += (
                valid_tokens
            )

            if (
                (i + 1) % 20 == 0
                or
                i + 1 == num_chunks
            ):

                running_ppl = math.exp(
                    total_nll
                    / total_target_tokens
                )

                print(
                    f"[{i + 1}/"
                    f"{num_chunks}] "
                    f"running PPL = "
                    f"{running_ppl:.4f}"
                )

    mean_nll = (
        total_nll
        / total_target_tokens
    )

    ppl = math.exp(
        mean_nll
    )

    return (
        mean_nll,
        ppl,
        num_chunks,
        total_target_tokens,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
    )

    parser.add_argument(
        "--sparsity",
        type=float,
        default=None,
        help="Ignored for N:M patterns (implied by N/M).",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="unstructured",
        help="unstructured, N:M (e.g. 2:4) or blockR[xC] (e.g. block16)",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--calib-samples",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON file for the result record",
    )

    args = parser.parse_args()

    args.sparsity = pattern_sparsity(
        parse_pattern(args.pattern),
        args.sparsity,
    )

    if not (
        0.0 <= args.sparsity < 1.0
    ):
        raise ValueError(
            "--sparsity must be in [0, 1)"
        )

    torch.set_num_threads(
        args.threads
    )

    torch.manual_seed(
        args.seed
    )

    print(
        f"PyTorch threads: "
        f"{torch.get_num_threads()}"
    )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    print("Loading dense model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
    )

    model.eval()

    linears = get_target_linears(
        model
    )

    print(
        f"Target Linear modules: "
        f"{len(linears)}"
    )

    print(
        "Preparing calibration data..."
    )

    calibration_samples = (
        get_calibration_samples(
            tokenizer=tokenizer,
            nsamples=args.calib_samples,
            seq_len=args.seq_len,
            seed=args.seed,
        )
    )

    stats = collect_activation_stats(
        model,
        calibration_samples,
    )

    actual_sparsity = prune_wanda(
        model,
        stats,
        args.sparsity,
        args.pattern,
    )

    save_model(
        model,
        tokenizer,
        args.save_path,
    )

    (
        mean_nll,
        ppl,
        chunks,
        target_tokens,
    ) = evaluate_ppl(
        model,
        tokenizer,
        args.seq_len,
        args.max_samples,
    )

    print()
    print("===== Result =====")

    print(
        "Method:          wanda"
    )

    print(
        f"Pattern:         "
        f"{args.pattern}"
    )

    print(
        f"Target sparsity: "
        f"{args.sparsity:.4f}"
    )

    print(
        f"Actual sparsity: "
        f"{actual_sparsity:.4f}"
    )

    print(
        f"Calibration:     "
        f"{args.calib_samples} samples"
    )

    print(
        f"Sequence len:    "
        f"{args.seq_len}"
    )

    print(
        f"Chunks:          "
        f"{chunks}"
    )

    print(
        f"Target tokens:   "
        f"{target_tokens:,}"
    )

    print(
        f"Mean NLL:        "
        f"{mean_nll:.6f}"
    )

    print(
        f"Perplexity:      "
        f"{ppl:.4f}"
    )

    if args.save_path is not None:

        print(
            f"Saved model:     "
            f"{os.path.expandvars(args.save_path)}"
        )

    record = {
        "experiment": "pruning",
        "method": "wanda",
        "args": vars(args),
        "model_revision": model_revision(model),
        "pattern": args.pattern,
        "target_sparsity": args.sparsity,
        "actual_sparsity": actual_sparsity,
        "calib_samples": args.calib_samples,
        "seq_len": args.seq_len,
        "eval_chunks": chunks,
        "eval_tokens": target_tokens,
        "full_eval": args.max_samples is None,
        "mean_nll": mean_nll,
        "ppl": ppl,
    }

    write_json(args.output, record)

    if args.save_path is not None:
        write_json(
            os.path.join(
                os.path.expandvars(args.save_path),
                "pruning_meta.json",
            ),
            record,
        )


if __name__ == "__main__":
    main()