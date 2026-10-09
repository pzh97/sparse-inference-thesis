import argparse
import math
import os
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM

from result_io import read_json, write_json


def layer_type_from_name(name):
    for t in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]:
        if name.endswith(t):
            return t

    return "other"


def run_lengths(binary_row):
    """
    binary_row:
        1 = nonzero
        0 = zero

    returns:
        zero runs
        nonzero runs
    """

    zero_runs = []
    nonzero_runs = []

    current_value = int(binary_row[0])
    current_len = 1

    for value in binary_row[1:]:
        value = int(value)

        if value == current_value:
            current_len += 1
        else:
            if current_value == 0:
                zero_runs.append(current_len)
            else:
                nonzero_runs.append(current_len)

            current_value = value
            current_len = 1

    if current_value == 0:
        zero_runs.append(current_len)
    else:
        nonzero_runs.append(current_len)

    return zero_runs, nonzero_runs


def row_run_lengths(mask):
    """
    Vectorized equivalent of run_lengths() applied to every row of a
    boolean mask. Returns (zero_runs, nonzero_runs) as Python lists.
    """

    rows, _ = mask.shape

    # Separator value 2 at the end of every row so runs never cross rows.
    x = torch.cat(
        [
            mask.to(torch.int8),
            torch.full((rows, 1), 2, dtype=torch.int8),
        ],
        dim=1,
    ).flatten()

    change = torch.ones_like(x, dtype=torch.bool)
    change[1:] = x[1:] != x[:-1]

    starts = torch.nonzero(change).flatten()

    lengths = torch.diff(
        starts,
        append=torch.tensor([x.numel()]),
    )

    values = x[starts]

    return (
        lengths[values == 0].tolist(),
        lengths[values == 1].tolist(),
    )


def nm_2_4_ratio(mask):
    """
    Fraction of groups of 4 that contain exactly 2 nonzeros.
    mask shape: [rows, cols]
    """

    rows, cols = mask.shape

    usable_cols = (cols // 4) * 4

    if usable_cols == 0:
        return float("nan")

    grouped = mask[:, :usable_cols].reshape(
        rows,
        -1,
        4,
    )

    counts = grouped.sum(dim=2)

    return (
        (counts == 2)
        .float()
        .mean()
        .item()
    )


def block_occupancy(mask, block_size=16):
    """
    Fraction of blocks containing at least one nonzero.
    Lower occupancy at fixed sparsity may indicate stronger clustering.
    """

    rows, cols = mask.shape

    usable_rows = (rows // block_size) * block_size
    usable_cols = (cols // block_size) * block_size

    if usable_rows == 0 or usable_cols == 0:
        return float("nan")

    x = mask[
        :usable_rows,
        :usable_cols,
    ]

    blocks = x.reshape(
        usable_rows // block_size,
        block_size,
        usable_cols // block_size,
        block_size,
    )

    blocks = blocks.permute(
        0,
        2,
        1,
        3,
    )

    occupied = (
        blocks
        .reshape(
            -1,
            block_size * block_size,
        )
        .any(dim=1)
    )

    return occupied.float().mean().item()


def analyze_matrix(weight):
    mask = weight != 0

    rows, cols = mask.shape

    total = mask.numel()
    nnz = mask.sum().item()

    sparsity = 1.0 - nnz / total

    row_nnz = mask.sum(dim=1).float()
    col_nnz = mask.sum(dim=0).float()

    zero_runs, nonzero_runs = row_run_lengths(
        mask.cpu()
    )

    avg_zero_run = (
        sum(zero_runs) / len(zero_runs)
        if zero_runs
        else 0.0
    )

    max_zero_run = (
        max(zero_runs)
        if zero_runs
        else 0
    )

    avg_nonzero_run = (
        sum(nonzero_runs) / len(nonzero_runs)
        if nonzero_runs
        else 0.0
    )

    max_nonzero_run = (
        max(nonzero_runs)
        if nonzero_runs
        else 0
    )

    return {
        "sparsity": sparsity,

        "row_nnz_mean":
            row_nnz.mean().item(),

        "row_nnz_std":
            row_nnz.std(unbiased=False).item(),

        "col_nnz_mean":
            col_nnz.mean().item(),

        "col_nnz_std":
            col_nnz.std(unbiased=False).item(),

        "avg_zero_run":
            avg_zero_run,

        "max_zero_run":
            max_zero_run,

        "avg_nonzero_run":
            avg_nonzero_run,

        "max_nonzero_run":
            max_nonzero_run,

        "block16_occupancy":
            block_occupancy(
                mask,
                block_size=16,
            ),

        "two_of_four_ratio":
            nm_2_4_ratio(mask),
    }


def mean(values):
    values = [
        x for x in values
        if not math.isnan(x)
    ]

    if not values:
        return float("nan")

    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to pruned Hugging Face model",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON file with per-layer statistics",
    )

    args = parser.parse_args()

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float32,
    )

    model.eval()

    grouped = defaultdict(list)

    per_layer = {}

    num_layers = 0

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

        stats = analyze_matrix(
            module.weight.data
        )

        layer_type = (
            layer_type_from_name(name)
        )

        grouped[layer_type].append(
            stats
        )

        per_layer[name] = stats

        num_layers += 1

    print()
    print(
        f"Analyzed Linear modules: "
        f"{num_layers}"
    )

    print()
    print(
        "===== Pattern summary ====="
    )

    header = (
        f"{'type':12s} "
        f"{'count':>5s} "
        f"{'sparsity':>9s} "
        f"{'row_std':>9s} "
        f"{'col_std':>9s} "
        f"{'zero_run':>9s} "
        f"{'max_zero':>9s} "
        f"{'block16':>9s} "
        f"{'2:4':>9s}"
    )

    print(header)

    print(
        "-" * len(header)
    )

    for layer_type, items in grouped.items():

        print(
            f"{layer_type:12s} "
            f"{len(items):5d} "
            f"{mean([x['sparsity'] for x in items]):9.4f} "
            f"{mean([x['row_nnz_std'] for x in items]):9.3f} "
            f"{mean([x['col_nnz_std'] for x in items]):9.3f} "
            f"{mean([x['avg_zero_run'] for x in items]):9.3f} "
            f"{mean([x['max_zero_run'] for x in items]):9.1f} "
            f"{mean([x['block16_occupancy'] for x in items]):9.4f} "
            f"{mean([x['two_of_four_ratio'] for x in items]):9.4f}"
        )


    pruning_meta_path = os.path.join(
        args.model_path,
        "pruning_meta.json",
    )

    write_json(
        args.output,
        {
            "experiment": "sparsity_patterns",
            "args": vars(args),
            "pruning": (
                read_json(pruning_meta_path)
                if os.path.exists(pruning_meta_path)
                else None
            ),
            "per_layer": per_layer,
        },
    )


if __name__ == "__main__":
    main()
