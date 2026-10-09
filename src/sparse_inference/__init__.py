"""
Shared code for the sparse inference experiments.

    results       JSON result records with run metadata
    masks         pruning patterns (unstructured, N:M, block)
    model_utils   Transformer Linear selection, module replacement, saving
    data          WikiText-2 calibration samples, perplexity, prompts
    csr           CSR Linear / MLP modules (oneMKL through torch.sparse)
    pattern_stats sparsity pattern statistics of a weight matrix

The scripts in scripts/ find this package through PYTHONPATH, which
scripts/run_cpu_benchmark.sh sets.
"""
