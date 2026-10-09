"""
Sparsity patterns shared by the pruning scripts.

A pattern is given as a short string:

    unstructured   no constraint on where the zeros go
    N:M            N nonzeros kept in every group of M consecutive
                   input columns (e.g. 2:4, 4:8, 1:4)
    blockR[xC]     whole R x C blocks are removed (e.g. block16 = 16x16,
                   block4x16 = 4 rows x 16 columns)

All helpers take an importance metric with the shape of the weight
matrix ([out_features, in_features], higher = more important) and return
a boolean mask where True marks the weights to REMOVE.
"""

import torch


def parse_pattern(pattern):
    """
    Returns a dict:
        {"kind": "unstructured"}
        {"kind": "nm", "n": N, "m": M}
        {"kind": "block", "rows": R, "cols": C}
    """
    pattern = pattern.strip().lower()

    if pattern == "unstructured":
        return {"kind": "unstructured"}

    if ":" in pattern:
        n, m = (int(x) for x in pattern.split(":"))

        if not 0 < n < m:
            raise ValueError(f"Invalid N:M pattern {pattern}")

        return {"kind": "nm", "n": n, "m": m}

    if pattern.startswith("block"):
        shape = pattern[len("block"):]

        if "x" in shape:
            rows, cols = (int(x) for x in shape.split("x"))
        else:
            rows = cols = int(shape)

        return {"kind": "block", "rows": rows, "cols": cols}

    raise ValueError(
        f"Unknown pattern '{pattern}'. "
        f"Use unstructured, N:M or blockR[xC]."
    )


def pattern_sparsity(spec, requested):
    """
    N:M fixes the sparsity; the other patterns use the requested value.
    """
    if spec["kind"] == "nm":
        return 1.0 - spec["n"] / spec["m"]

    if requested is None:
        raise ValueError("--sparsity is required for this pattern")

    return requested


def pattern_tag(pattern, sparsity):
    """
    Short name used for checkpoint directories and result files.
    """
    spec = parse_pattern(pattern)

    if spec["kind"] == "nm":
        return f"nm{spec['n']}-{spec['m']}"

    pct = int(round(sparsity * 100))

    if spec["kind"] == "block":
        return f"block{spec['rows']}x{spec['cols']}-{pct}"

    return f"u{pct}"


def unstructured_row_mask(metric, sparsity):
    """
    Remove the lowest-metric weights inside each output row
    (Wanda's per-output comparison group).
    """
    num_prune = int(metric.shape[1] * sparsity)

    mask = torch.zeros_like(metric, dtype=torch.bool)

    if num_prune > 0:
        idx = torch.topk(metric, k=num_prune, dim=1, largest=False).indices
        mask.scatter_(1, idx, True)

    return mask


def unstructured_matrix_mask(metric, sparsity):
    """
    Remove the lowest-metric weights over the whole matrix.
    """
    num_prune = int(metric.numel() * sparsity)

    mask = torch.zeros(metric.numel(), dtype=torch.bool, device=metric.device)

    if num_prune > 0:
        idx = torch.topk(metric.flatten(), k=num_prune, largest=False).indices
        mask[idx] = True

    return mask.view_as(metric)


def nm_mask(metric, n, m):
    """
    Keep the n largest-metric weights in every group of m consecutive
    input columns.
    """
    rows, cols = metric.shape

    if cols % m != 0:
        raise ValueError(
            f"in_features={cols} is not a multiple of M={m}"
        )

    groups = metric.reshape(rows, cols // m, m)

    idx = torch.topk(groups, k=m - n, dim=2, largest=False).indices

    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(2, idx, True)

    return mask.reshape(rows, cols)


def block_mask(metric, sparsity, block_rows, block_cols):
    """
    Score each block by the sum of its metric and remove the lowest
    `sparsity` fraction of blocks of the matrix.
    """
    rows, cols = metric.shape

    if rows % block_rows != 0 or cols % block_cols != 0:
        raise ValueError(
            f"Matrix {rows}x{cols} is not divisible into "
            f"{block_rows}x{block_cols} blocks"
        )

    br = rows // block_rows
    bc = cols // block_cols

    scores = (
        metric
        .reshape(br, block_rows, bc, block_cols)
        .sum(dim=(1, 3))
    )

    block_prune = unstructured_matrix_mask(scores, sparsity)

    return (
        block_prune
        .repeat_interleave(block_rows, dim=0)
        .repeat_interleave(block_cols, dim=1)
    )


def build_mask(metric, pattern, sparsity, unstructured_scope="matrix"):
    """
    Dispatch on the pattern string.

    unstructured_scope:
        "matrix"  magnitude pruning (global per matrix)
        "row"     Wanda (per output row)
    """
    spec = parse_pattern(pattern)

    if spec["kind"] == "nm":
        return nm_mask(metric, spec["n"], spec["m"])

    if spec["kind"] == "block":
        return block_mask(metric, sparsity, spec["rows"], spec["cols"])

    if unstructured_scope == "row":
        return unstructured_row_mask(metric, sparsity)

    return unstructured_matrix_mask(metric, sparsity)
