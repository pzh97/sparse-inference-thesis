"""
WikiText-2 calibration data, perplexity and benchmark prompts.
"""

import math

import torch


def _wikitext2(split):
    from datasets import load_dataset

    return load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split=split,
    )


def _tokenize_split(tokenizer, split):
    text = "\n\n".join(_wikitext2(split)["text"])

    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        verbose=False,
    ).input_ids


def get_calibration_samples(tokenizer, nsamples, seq_len=2048, seed=0):
    """
    Random fixed-length sequences from the WikiText-2 train split.

    Returns a list of [1, seq_len] tensors.
    """
    tokens = _tokenize_split(tokenizer, "train")

    if tokens.shape[1] <= seq_len:
        raise ValueError(
            f"Calibration corpus too short: {tokens.shape[1]} tokens "
            f"for seq_len={seq_len}"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    max_start = tokens.shape[1] - seq_len - 1

    samples = []
    for _ in range(nsamples):
        start = torch.randint(
            low=0,
            high=max_start,
            size=(1,),
            generator=generator,
        ).item()
        samples.append(tokens[:, start:start + seq_len])

    return samples


def evaluate_ppl(model, tokenizer, seq_len, max_samples=None):
    """
    WikiText-2 test perplexity on non-overlapping seq_len chunks.

    Returns (mean_nll, ppl, num_chunks, target_tokens).
    """
    tokens = _tokenize_split(tokenizer, "test")

    num_chunks = tokens.shape[1] // seq_len
    if max_samples is not None:
        num_chunks = min(num_chunks, max_samples)

    print()
    print(f"Evaluating {num_chunks} chunks...")

    total_nll = 0.0
    total_target_tokens = 0

    model.eval()

    with torch.inference_mode():
        for i in range(num_chunks):
            batch = tokens[:, i * seq_len:(i + 1) * seq_len]

            outputs = model(input_ids=batch, labels=batch, use_cache=False)

            # The HF loss is the mean over the seq_len - 1 predicted tokens.
            valid_tokens = batch.numel() - 1
            total_nll += outputs.loss.item() * valid_tokens
            total_target_tokens += valid_tokens

            if (i + 1) % 20 == 0 or i + 1 == num_chunks:
                running_ppl = math.exp(total_nll / total_target_tokens)
                print(f"[{i + 1}/{num_chunks}] running PPL = {running_ppl:.4f}")

    mean_nll = total_nll / total_target_tokens

    return mean_nll, math.exp(mean_nll), num_chunks, total_target_tokens


def make_prompt_tokens(tokenizer, prompt_len):
    """
    Deterministic repeated text truncated to prompt_len tokens.
    """
    base_text = (
        "The history of computer science and artificial intelligence "
        "contains many important developments in algorithms, systems, "
        "hardware, programming languages, and machine learning. "
    )

    tokens = tokenizer(
        base_text * 500,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids

    if tokens.shape[1] < prompt_len:
        raise RuntimeError("Could not generate enough prompt tokens.")

    return tokens[:, :prompt_len]
