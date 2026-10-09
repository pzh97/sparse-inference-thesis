import argparse
import os
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM

from sparse_inference.pattern_stats import analyze_matrix, layer_type_from_name, mean
from sparse_inference.results import read_json, write_json


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
