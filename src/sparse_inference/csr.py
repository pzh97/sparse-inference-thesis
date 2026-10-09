"""
CSR execution of the Transformer Linear modules on CPU.

torch.sparse.mm(W_csr, B) runs oneMKL (mkl_sparse_s_csr_ng_n_mm_*_i4_avx512).
Because the sparse operand must be on the left, y = x W^T is computed as
y^T = W x^T, which makes the activation layout part of the cost.

Implementations compared in the end-to-end benchmark:

    v0      CSRLinear, int64 indices (2026-10-08 sweep)
    int32   CSRLinear, int32 indices
    mlp_t   CSRMLP (int32, transposed dataflow) for the MLP blocks,
            CSRLinear int32 for the attention projections
"""

import torch

from sparse_inference.model_utils import get_parent_module


MLP_SUFFIXES = (
    "gate_proj",
    "up_proj",
    "down_proj",
)

ATTN_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)

TARGET_GROUPS = {
    "mlp": MLP_SUFFIXES,
    "attn": ATTN_SUFFIXES,
    "all": ATTN_SUFFIXES + MLP_SUFFIXES,
}

CSR_IMPLS = (
    "v0",
    "int32",
    "mlp_t",
)


def to_csr(weight, index_dtype=torch.int64):
    """
    CSR copy of a dense weight.

    torch creates int64 indices, but the oneMKL kernel takes int32
    indices, so with int64 the index arrays are converted on every call.
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


class CSRLinear(torch.nn.Module):
    """
    Drop-in replacement for nn.Linear with a CSR weight:

        y = x W^T + b   computed as   y^T = W x^T

    x^T is a column-major view, so oneMKL uses its column-major kernel.
    """

    def __init__(self, linear, index_dtype=torch.int64):
        super().__init__()

        if linear.weight.device.type != "cpu":
            raise ValueError("CSRLinear currently expects CPU weights.")

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.register_buffer(
            "weight_csr",
            to_csr(linear.weight.detach().contiguous(), index_dtype),
        )

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x):
        original_shape = x.shape

        x2d = x.reshape(-1, self.in_features)

        y2d = torch.sparse.mm(self.weight_csr, x2d.T).T

        if self.bias is not None:
            y2d = y2d + self.bias

        return y2d.reshape(*original_shape[:-1], self.out_features)


class CSRMLP(torch.nn.Module):
    """
    Qwen2 MLP with CSR weights and a transposed dataflow:

        h^T = act(W_gate x^T) * (W_up x^T)
        y^T = W_down h^T

    The activations stay [features, tokens] row-major, so oneMKL uses
    its row-major kernel (2-3x faster than the column-major one for
    M > 1). The elementwise ops do not care about the layout; only the
    896-wide input and output are transposed, once per MLP block.
    """

    def __init__(self, mlp, index_dtype=torch.int32):
        super().__init__()

        self.hidden_size = mlp.gate_proj.in_features
        self.act_fn = mlp.act_fn

        for name in MLP_SUFFIXES:
            self.register_buffer(
                f"{name}_csr",
                to_csr(getattr(mlp, name).weight.detach().contiguous(), index_dtype),
            )

    def forward(self, x):
        original_shape = x.shape

        xT = x.reshape(-1, self.hidden_size).T.contiguous()

        hT = self.act_fn(
            torch.sparse.mm(self.gate_proj_csr, xT)
        ) * torch.sparse.mm(self.up_proj_csr, xT)

        yT = torch.sparse.mm(self.down_proj_csr, hT)

        return yT.T.reshape(original_shape)


def _count_zeros(weights):
    total = sum(W.numel() for W in weights)
    zeros = sum((W == 0).sum().item() for W in weights)
    return total, zeros


def replace_linears_with_csr(model, suffixes, impl="v0"):
    """
    Convert the Transformer Linear modules ending in `suffixes` to CSR.

    Returns (number of converted Linear modules, their sparsity).
    """
    if impl not in CSR_IMPLS:
        raise ValueError(f"Unknown CSR implementation {impl}")

    index_dtype = torch.int64 if impl == "v0" else torch.int32
    suffixes = tuple(suffixes)

    converted = []

    if impl == "mlp_t" and set(MLP_SUFFIXES) <= set(suffixes):
        mlp_names = [
            name
            for name, _ in model.named_modules()
            if name.startswith("model.layers.") and name.endswith(".mlp")
        ]

        for name in mlp_names:
            mlp = model.get_submodule(name)
            converted += [getattr(mlp, s).weight.data for s in MLP_SUFFIXES]

            parent, child = get_parent_module(model, name)
            setattr(parent, child, CSRMLP(mlp, index_dtype))

        print(f"Replaced {len(mlp_names)} MLP blocks with transposed-dataflow CSRMLP")

        suffixes = tuple(s for s in suffixes if s not in MLP_SUFFIXES)

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("model.layers.")
        and name.endswith(suffixes)
    ] if suffixes else []

    print(f"Replacing {len(targets)} Linear modules with CSR...")

    for name, module in targets:
        converted.append(module.weight.data)

        parent, child = get_parent_module(model, name)
        setattr(parent, child, CSRLinear(module, index_dtype))

    total, zeros = _count_zeros(converted)
    sparsity = zeros / total

    print(f"CSR target sparsity: {sparsity:.4f}")

    return len(converted), sparsity
