# Sparse Inference

Experiments on efficient inference of pruned language models.

## Hardware
CESGA FinisTerrae III
Intel Xeon Platinum 8352Y (Ice Lake-SP, 2 x 32 cores, 4 NUMA nodes of 16 cores)

## Initial scope
- Dense CPU baseline
- CSR sparse baseline
- Pruning experiments
- Profiling

## Pinned versions

| Component | Version |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Container | `intel/pytorch:cpu-2.13.0` (image 20260907), `$STORE/sparse-inference/containers/intel-pytorch-cpu.sif` |
| PyTorch | 2.13.0+cpu, oneMKL 2024.2, oneDNN 3.12.0, AVX-512 |
| transformers / datasets | 4.57.6 / 3.6.0 (in `$STORE/sparse-inference/pydeps`) |
| SparseGPT | `third_party/sparsegpt` (sha256 of `sparsegpt.py` stored in every result) |
| Dataset | `Salesforce/wikitext`, `wikitext-2-raw-v1` |

`scripts/run_cpu_benchmark.sh` runs with `HF_HUB_OFFLINE=1`, so only the
cached revision is used. Every result JSON also records the git commit,
library versions, CPU binding and SLURM job.

The CSR baseline is oneMKL: `torch.sparse.mm(W_csr, x.T)` executes
`mkl_sparse_s_csr_ng_n_mm_c_ker_i4_avx512` (checked with `perf record`).

### CSR implementations (`src/sparse_inference/csr.py`, `--csr-impl`)

| Name | What it does |
|---|---|
| `v0` | `CSRLinear`, int64 indices: oneMKL takes int32, so the index arrays are converted on every call (2026-10-08 sweep) |
| `int32` | `CSRLinear` with int32 indices |
| `mlp_t` | int32 + `CSRMLP`: the MLP block runs in a transposed dataflow ([features, tokens]), so oneMKL uses its row-major kernel (2-3x faster than the column-major one for M > 1); attention projections use `int32` |

The sparse operand must be on the left (y^T = W x^T), so every CSR layer
pays layout conversions; in `mlp_t` they are ~13% of prefill time.

## Layout

```
configs/prune_sweep.txt     one line per pruned checkpoint (method, pattern, sparsity)
src/sparse_inference/       shared code (results, masks, data/PPL, CSR modules, pattern stats)
scripts/                    experiment scripts (run through run_cpu_benchmark.sh,
                            which puts src/ on PYTHONPATH)
slurm/                      batch jobs for the sweep
results-meta/               small summary tables (committed)
report/                     first-iteration report (LaTeX) and its figures
$STORE/sparse-inference/
    checkpoints/<name>/     pruned models + pruning_meta.json
    results/                raw JSON/CSV results (not committed)
```

Checkpoint names: `<method>-<tag>`, tag = `u50` (unstructured 50%),
`nm2-4` (2:4), `block16x16-70` (16x16 blocks, 70%).

## Workflow

```bash
mkdir -p slurm-logs

# 1. Prune every configuration + full WikiText-2 PPL (seq len 2048)
P=$(sbatch --parsable --array=0-39 slurm/prune.sbatch)

# 2. Pattern analysis, per-layer and end-to-end dense vs CSR, profiler
B=$(sbatch --parsable --dependency=afterany:$P --array=1-39 slurm/bench.sbatch)

# 3. TMA of dense vs CSR kernels (perf, user mode)
sbatch --dependency=afterany:$P slurm/tma.sbatch

# 4. Summary tables
python3 scripts/summarize_results.py --out results-meta

# 5. Figures and report (plain python3 + matplotlib, pdflatex)
python3 scripts/make_figures.py --meta results-meta --out report/figures
cd report && latexmk -pdf report.tex
```

The first-iteration report is `report/report.pdf`.

Every step skips outputs that already exist (`FORCE=1` to redo).
`SMOKE=1` runs a tiny version of each job into `$STORE/sparse-inference/smoke/`
(`SMOKE_ROOT` to change it). `slurm/bench.sbatch` options: `STEPS`,
`CSR_IMPLS`, `LAYER_MODES` (hot/cold), `E2E_REPS`, `E2E_DIR`; see the file.

### Scripts

| Script | What it measures |
|---|---|
| `eval_{magnitude,wanda,sparsegpt}_pruning.py` | prune (`--pattern unstructured / N:M / blockR[xC]`), save, PPL |
| `eval_wikitext2_ppl.py` | dense reference PPL |
| `analyze_sparsity_patterns.py` | row/col nnz spread, zero runs, block occupancy, 2:4 ratio |
| `bench_layer_formats.py` | per-layer dense vs CSR (int64/int32, column/row-major) vs CSR on a random mask with the same nnz, for several M; `--cold` cycles through weight copies larger than the L3 |
| `bench_qwen_csr_inference.py` | end-to-end prefill/decode, `--targets mlp/attn/all`, `--csr-impl v0/int32/mlp_t` |
| `check_timing_stability.py` | dense timing in fresh processes (`slurm/stability.sbatch`) |
| `profile_qwen.py` | PyTorch profiler, dense or CSR, prefill or decode |
| `run_tma.sh` + `tma_kernel.py` + `tma_report.py` | Top-down analysis of a single kernel in steady state |

## Measurement notes

- Results in `e2e/` computed lm_head logits for every prompt token (HF
  default, ~25% of prefill time); `e2e_lk1/` uses `logits_to_keep=1` as in
  real inference. Decode is unaffected.
- Dense timings are not identical across runs: dense `F.linear` at M=32 is
  2x faster on some nodes (node property, stable per node), and full-model
  prefill at 512 tokens lands at ~400 or ~505 ms per process (~27% of
  processes are fast). Thread pinning does not change this. Compare CSR and
  dense within the same process and use several processes (`E2E_REPS`).
- Per-layer benchmarks without `--cold` keep one matrix in the L3 and
  favour dense; `--cold` matches the end-to-end decode results.

## Known limitations

- Wanda and SparseGPT are applied in one shot: the calibration statistics
  of all layers come from the unpruned model. The reference implementations
  prune layer by layer and feed the pruned outputs to the next layer,
  which gives somewhat better PPL at high sparsity.
- The TMA Level 2/3 memory breakdown uses simplified Ice Lake formulas;
  Level 1 is exact (PERF_METRICS). Cross-check with VTune
  (`module load cesga/2020 vtune/2022.1.0`).
