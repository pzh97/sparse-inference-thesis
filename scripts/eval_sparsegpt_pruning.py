import argparse
import math
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------
# SparseGPT import
# ---------------------------------------------------------------------

SPARSEGPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party",
    "sparsegpt",
)

if SPARSEGPT_DIR not in sys.path:
    sys.path.insert(0, SPARSEGPT_DIR)

from sparsegpt import SparseGPT

from pruning_patterns import parse_pattern, pattern_sparsity
from result_io import model_revision, write_json


# ---------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------

def get_calibration_samples(
    tokenizer,
    nsamples=16,
    seq_len=2048,
    seed=0,
):
    """
    Sample random fixed-length sequences from WikiText-2 train split.

    Returns:
        list of tensors with shape [1, seq_len]
    """

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
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

    if tokens.shape[1] <= seq_len:
        raise ValueError(
            f"Calibration corpus too short: "
            f"{tokens.shape[1]} tokens "
            f"for seq_len={seq_len}"
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

        sample = tokens[
            :,
            start:start + seq_len
        ]

        samples.append(sample)

    return samples


# ---------------------------------------------------------------------
# Linear selection
# ---------------------------------------------------------------------

def get_target_linears(model):
    """
    Select Transformer Linear modules only.

    Excludes lm_head.

    Qwen2.5-0.5B should give 168 modules:
        24 layers * 7 Linear modules
    """

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
# SparseGPT calibration statistics
# ---------------------------------------------------------------------

def collect_sparsegpt_stats(
    model,
    calibration_samples,
):
    """
    Create one SparseGPT object per target Linear and collect
    activation statistics using forward hooks.
    """

    linears = get_target_linears(
        model
    )

    print(
        f"Preparing SparseGPT objects "
        f"for {len(linears)} Linear modules..."
    )

    sparsegpt_objects = {}

    for name, module in linears.items():
        sparsegpt_objects[name] = (
            SparseGPT(module)
        )

    hooks = []

    for name, module in linears.items():

        def make_hook(layer_name):

            def hook(
                module,
                inputs,
                output,
            ):
                inp = inputs[0].detach()

                out = (
                    output.detach()
                    if isinstance(
                        output,
                        torch.Tensor,
                    )
                    else output
                )

                sparsegpt_objects[
                    layer_name
                ].add_batch(
                    inp,
                    out,
                )

            return hook

        handle = (
            module.register_forward_hook(
                make_hook(name)
            )
        )

        hooks.append(handle)

    print(
        f"Collecting calibration statistics "
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

    for handle in hooks:
        handle.remove()

    return sparsegpt_objects


# ---------------------------------------------------------------------
# SparseGPT pruning
# ---------------------------------------------------------------------

def prune_sparsegpt(
    model,
    sparsegpt_objects,
    sparsity,
    percdamp=0.01,
    blocksize=128,
    pattern="unstructured",
):
    """
    Apply SparseGPT pruning to all Transformer Linear modules.

    SparseGPT supports unstructured and N:M sparsity natively
    (prunen = weights removed per group of prunem columns).
    """

    spec = parse_pattern(pattern)

    if spec["kind"] == "block":
        raise ValueError(
            "SparseGPT does not support block patterns; "
            "use magnitude or Wanda for block pruning."
        )

    if spec["kind"] == "nm":
        prunen = spec["m"] - spec["n"]
        prunem = spec["m"]
    else:
        prunen = 0
        prunem = 0

    print()
    print(
        "Applying SparseGPT pruning..."
    )

    total_objects = len(
        sparsegpt_objects
    )

    for i, (
        name,
        sgpt,
    ) in enumerate(
        sparsegpt_objects.items()
    ):

        print(
            f"[{i + 1}/{total_objects}] "
            f"{name}"
        )

        sgpt.fasterprune(
            sparsity=sparsity,
            prunen=prunen,
            prunem=prunem,
            blocksize=blocksize,
            percdamp=percdamp,
        )

        sgpt.free()

    total_weights = 0
    total_zeros = 0

    linears = get_target_linears(
        model
    )

    for _, module in linears.items():

        W = module.weight.data

        total_weights += (
            W.numel()
        )

        total_zeros += (
            W == 0
        ).sum().item()

    actual_sparsity = (
        total_zeros
        / total_weights
    )

    print()
    print(
        "===== SparseGPT pruning ====="
    )

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
# Saving
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

    print(
        "Save complete."
    )


# ---------------------------------------------------------------------
# WikiText-2 PPL
# ---------------------------------------------------------------------

def evaluate_ppl(
    model,
    tokenizer,
    seq_len,
    max_samples=None,
):
    """
    Evaluate perplexity using non-overlapping fixed-length chunks.

    Matches our previous dense / magnitude / Wanda evaluation style.
    """

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

    total_tokens = (
        tokens.shape[1]
    )

    num_chunks = (
        total_tokens
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

            end = (
                start
                + seq_len
            )

            batch = tokens[
                :,
                start:end
            ]

            outputs = model(
                input_ids=batch,
                labels=batch,
                use_cache=False,
            )

            # Causal LM predicts tokens 1..L-1
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

                running_nll = (
                    total_nll
                    / total_target_tokens
                )

                running_ppl = (
                    math.exp(
                        running_nll
                    )
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

    parser = argparse.ArgumentParser(
        description=(
            "SparseGPT pruning + "
            "WikiText-2 evaluation "
            "for Qwen2.5"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=(
            "Qwen/Qwen2.5-0.5B"
        ),
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
        help="unstructured or N:M (e.g. 2:4)",
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
        help=(
            "Maximum number of "
            "WikiText-2 evaluation chunks. "
            "Omit for full evaluation."
        ),
    )

    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
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
        help=(
            "Directory for saving "
            "the pruned Hugging Face model."
        ),
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
        0.0
        <= args.sparsity
        < 1.0
    ):
        raise ValueError(
            "--sparsity must be "
            "in [0, 1)"
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

    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model
        )
    )

    print(
        "Loading dense model..."
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float32,
        )
    )

    model.eval()

    linears = get_target_linears(
        model
    )

    print(
        f"Target Linear modules: "
        f"{len(linears)}"
    )

    if len(linears) != 168:
        print(
            "WARNING: expected 168 "
            "Transformer Linear modules "
            "for Qwen2.5-0.5B."
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

    sparsegpt_objects = (
        collect_sparsegpt_stats(
            model,
            calibration_samples,
        )
    )

    actual_sparsity = (
        prune_sparsegpt(
            model=model,
            sparsegpt_objects=
                sparsegpt_objects,
            sparsity=args.sparsity,
            percdamp=args.percdamp,
            blocksize=args.blocksize,
            pattern=args.pattern,
        )
    )

    save_model(
        model=model,
        tokenizer=tokenizer,
        save_path=args.save_path,
    )

    (
        mean_nll,
        ppl,
        chunks,
        target_tokens,
    ) = evaluate_ppl(
        model=model,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
    )

    print()
    print(
        "===== Result ====="
    )

    print(
        "Method:          "
        "sparsegpt"
    )

    print(
        f"Model:           "
        f"{args.model}"
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
        f"Percdamp:        "
        f"{args.percdamp}"
    )

    print(
        f"Blocksize:       "
        f"{args.blocksize}"
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

        expanded_path = (
            os.path.expandvars(
                os.path.expanduser(
                    args.save_path
                )
            )
        )

        print(
            f"Saved model:     "
            f"{expanded_path}"
        )

    record = {
        "experiment": "pruning",
        "method": "sparsegpt",
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