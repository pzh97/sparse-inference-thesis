import argparse
import copy
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from result_io import read_json, write_json


# ============================================================
# CSR Linear
# ============================================================

class CSRLinear(torch.nn.Module):
    """
    Drop-in CPU replacement for nn.Linear using a CSR weight matrix.

    Original nn.Linear:
        y = x @ W.T + b

    CSR implementation:
        y.T = W @ x.T

    Weight shape:
        [out_features, in_features]
    """

    def __init__(
        self,
        linear: torch.nn.Linear,
        index_dtype=torch.int64,
    ):
        super().__init__()

        if linear.weight.device.type != "cpu":
            raise ValueError(
                "CSRLinear currently expects CPU weights."
            )

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        weight = linear.weight.detach().contiguous()

        self.register_buffer(
            "weight_csr",
            to_csr(
                weight,
                index_dtype,
            ),
        )

        if linear.bias is not None:
            self.register_buffer(
                "bias",
                linear.bias.detach().clone(),
            )
        else:
            self.bias = None

    def forward(self, x):
        """
        Supports x with shape:
            [..., in_features]

        Flatten all leading dimensions, perform sparse mm, then reshape.
        """

        original_shape = x.shape

        x2d = x.reshape(
            -1,
            self.in_features,
        )

        # W: [out, in]
        # x2d.T: [in, M]
        #
        # result.T:
        # [M, out]
        y2d = torch.sparse.mm(
            self.weight_csr,
            x2d.T,
        ).T

        if self.bias is not None:
            y2d = y2d + self.bias

        output_shape = (
            *original_shape[:-1],
            self.out_features,
        )

        return y2d.reshape(
            output_shape
        )


def to_csr(
    weight,
    index_dtype=torch.int64,
):
    """
    CSR copy of a dense weight.

    torch creates int64 indices, but the oneMKL kernel used on CPU
    (mkl_sparse_s_csr_ng_n_mm_*_i4) takes int32 indices, so with int64
    the index arrays are converted on every call. int32 avoids that.
    """

    csr = weight.to_sparse_csr()

    if index_dtype == torch.int64:
        return csr

    return torch.sparse_csr_tensor(
        csr.crow_indices().to(index_dtype),
        csr.col_indices().to(index_dtype),
        csr.values(),
        csr.shape,
        check_invariants=False,
    )


class CSRMLP(torch.nn.Module):
    """
    Qwen2 MLP with CSR weights and a transposed dataflow:

        h.T = act(W_gate @ x.T) * (W_up @ x.T)
        y.T = W_down @ h.T

    The activations stay as [features, tokens] (row-major), which lets
    oneMKL use its row-major kernel (2-3x faster than the column-major
    one used by CSRLinear for M > 1). The elementwise ops do not care
    about the layout, so only the 896-wide input and output are
    transposed, once per MLP block.
    """

    def __init__(
        self,
        mlp,
        index_dtype=torch.int32,
    ):
        super().__init__()

        self.hidden_size = mlp.gate_proj.in_features
        self.act_fn = mlp.act_fn

        for name in TARGET_SUFFIXES:
            self.register_buffer(
                f"{name}_csr",
                to_csr(
                    getattr(mlp, name).weight.detach().contiguous(),
                    index_dtype,
                ),
            )

    def forward(self, x):
        original_shape = x.shape

        xT = x.reshape(
            -1,
            self.hidden_size,
        ).T.contiguous()

        hT = self.act_fn(
            torch.sparse.mm(self.gate_proj_csr, xT)
        ) * torch.sparse.mm(self.up_proj_csr, xT)

        yT = torch.sparse.mm(
            self.down_proj_csr,
            hT,
        )

        return yT.T.reshape(original_shape)


# Implementations compared in the end-to-end benchmark:
#   v0      CSRLinear, int64 indices (2026-10-08 sweep)
#   int32   CSRLinear, int32 indices
#   mlp_t   CSRMLP (int32, transposed dataflow) for the MLP blocks,
#           CSRLinear int32 for the attention projections
CSR_IMPLS = (
    "v0",
    "int32",
    "mlp_t",
)


# ============================================================
# Module replacement
# ============================================================

TARGET_SUFFIXES = (
    "gate_proj",
    "up_proj",
    "down_proj",
)

TARGET_GROUPS = {
    "mlp": TARGET_SUFFIXES,
    "attn": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ),
}

TARGET_GROUPS["all"] = (
    TARGET_GROUPS["attn"]
    + TARGET_GROUPS["mlp"]
)


def get_parent_module(
    model,
    module_name,
):
    """
    Example:
        module_name =
        model.layers.0.mlp.gate_proj

    returns:
        parent = model.layers.0.mlp
        child_name = gate_proj
    """

    parts = module_name.split(".")

    parent = model

    for part in parts[:-1]:
        parent = getattr(
            parent,
            part,
        )

    return (
        parent,
        parts[-1],
    )


def replace_mlp_linears_with_csr(
    model,
):
    num_replaced, _ = replace_linears_with_csr(
        model,
        TARGET_SUFFIXES,
    )

    return num_replaced


def replace_linears_with_csr(
    model,
    suffixes,
    impl="v0",
):
    if impl not in CSR_IMPLS:
        raise ValueError(
            f"Unknown CSR implementation {impl}"
        )

    index_dtype = (
        torch.int64
        if impl == "v0"
        else torch.int32
    )

    suffixes = tuple(suffixes)

    num_mlp_modules = 0
    mlp_weights = 0
    mlp_zeros = 0

    if impl == "mlp_t" and set(TARGET_SUFFIXES) <= set(suffixes):

        mlp_names = [
            name
            for name, module in model.named_modules()
            if name.startswith("model.layers.")
            and name.endswith(".mlp")
        ]

        for name in mlp_names:
            mlp = model.get_submodule(name)

            for proj in TARGET_SUFFIXES:
                W = getattr(mlp, proj).weight.data
                mlp_weights += W.numel()
                mlp_zeros += (W == 0).sum().item()

            parent, child_name = get_parent_module(
                model,
                name,
            )

            setattr(
                parent,
                child_name,
                CSRMLP(
                    mlp,
                    index_dtype,
                ),
            )

            num_mlp_modules += len(TARGET_SUFFIXES)

        print(
            f"Replaced {len(mlp_names)} MLP blocks "
            f"with transposed-dataflow CSRMLP"
        )

        suffixes = tuple(
            x
            for x in suffixes
            if x not in TARGET_SUFFIXES
        )

    targets = []

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

        if not name.endswith(
            tuple(suffixes)
        ):
            continue

        targets.append(
            (
                name,
                module,
            )
        )

    print(
        f"Replacing {len(targets)} "
        f"Linear modules with CSR..."
    )

    total_weights = 0
    total_zeros = 0

    for i, (
        name,
        module,
    ) in enumerate(
        targets
    ):

        W = module.weight.data

        total_weights += W.numel()
        total_zeros += (
            W == 0
        ).sum().item()

        parent, child_name = (
            get_parent_module(
                model,
                name,
            )
        )

        csr_module = CSRLinear(
            module,
            index_dtype,
        )

        setattr(
            parent,
            child_name,
            csr_module,
        )

        print(
            f"[{i + 1}/{len(targets)}] "
            f"{name}"
        )

    total_weights += mlp_weights
    total_zeros += mlp_zeros

    sparsity = (
        total_zeros
        / total_weights
    )

    print()
    print(
        f"CSR target sparsity: "
        f"{sparsity:.4f}"
    )

    return (
        len(targets)
        + num_mlp_modules,
        sparsity,
    )


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
# Prompt creation
# ============================================================

def make_prompt_tokens(
    tokenizer,
    prompt_len,
):
    """
    Use deterministic repeated text and truncate to the requested length.
    """

    base_text = (
        "The history of computer science and artificial intelligence "
        "contains many important developments in algorithms, systems, "
        "hardware, programming languages, and machine learning. "
    )

    text = (
        base_text * 500
    )

    tokens = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids

    if tokens.shape[1] < prompt_len:
        raise RuntimeError(
            "Could not generate enough prompt tokens."
        )

    return tokens[
        :,
        :prompt_len
    ]


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
            )

        times = []

        for _ in range(repeats):

            synchronize()

            t0 = time.perf_counter()

            model(
                input_ids=input_ids,
                use_cache=True,
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