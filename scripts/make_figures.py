"""
Figures for the first-iteration report (report/figures/*.pdf).

Reads the summary tables in results-meta/ (scripts/summarize_results.py).
Runs with plain python3 + matplotlib, no torch needed:

    python3 scripts/make_figures.py --meta results-meta --out report/figures
"""

import argparse
import csv
import os
import re
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


# Categorical slots in fixed order (validated palette, light mode).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"

# Sequential blue ramp, light -> dark, for ordered categories (M, versions).
BLUE_RAMP = ["#9cc3f0", "#6aa3e6", "#2a78d6", "#1b5aa8", "#103c73"]

INK = "#1d1d1b"
INK_2 = "#55544f"
GRID = "#e4e3de"
DENSE_REF = "#8a8984"

METHOD_COLORS = {"magnitude": ORANGE, "wanda": AQUA, "sparsegpt": BLUE}
METHOD_LABELS = {"magnitude": "Magnitude", "wanda": "Wanda", "sparsegpt": "SparseGPT"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

FULL_WIDTH = 6.3  # inches, matches the LaTeX text width


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def base_name(name):
    """sparsegpt-u50-mlp-mlp_t-r2 -> sparsegpt-u50"""
    return re.sub(r"-(mlp|all)(-int32|-mlp_t)?(-r\d+)?$", "", name)


def unstructured_sparsity(name):
    # sparsegpt-u50 -> 0.5
    return int(name.rsplit("-u", 1)[1]) / 100.0


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path)
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(
            os.path.join(os.environ["PREVIEW_DIR"], name.replace(".pdf", ".png")),
            dpi=130,
        )
    plt.close(fig)
    print(f"wrote {path}")


def speedup_label(ax):
    ax.axhline(1.0, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.text(
        ax.get_xlim()[0], 1.0, " dense", va="bottom", ha="left",
        color=INK_2, fontsize=7.5,
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_ppl(meta):
    ppl = {}
    for r in read_csv(os.path.join(meta, "ppl_sweep.csv")):
        ppl[r["name"]] = float(r["ppl"])
    return ppl


def load_e2e(meta):
    """
    {(checkpoint, targets, impl, lk1): [rows]}  lk1 = run with logits_to_keep=1
    """
    groups = defaultdict(list)
    for r in read_csv(os.path.join(meta, "e2e_dense_vs_csr.csv")):
        lk1 = r.get("logits_to_keep") not in (None, "", "None")
        groups[(base_name(r["name"]), r["csr_targets"], r["csr_impl"], lk1)].append(r)
    return groups


def e2e_speedup(groups, ckpt, targets, impl, prompt_len, field):
    """
    Median over processes; prefers the logits_to_keep=1 runs.
    Returns (value, used_lk1) or (None, None).
    """
    for lk1 in (True, False):
        rows = [
            r for r in groups.get((ckpt, targets, impl, lk1), [])
            if int(r["prompt_len"]) == prompt_len
        ]
        if rows:
            return statistics.median(float(r[field]) for r in rows), lk1
    return None, None


def load_layers(meta, cold=True):
    """
    {(checkpoint, M, variant): mean speedup over the benchmarked layers/types}
    """
    acc = defaultdict(list)
    for r in read_csv(os.path.join(meta, "layer_formats_summary.csv")):
        if (r.get("cold") == "True") != cold:
            continue
        acc[(r["name"], int(r["M"]), r["variant"])].append(float(r["speedup_vs_dense"]))
    return {k: statistics.mean(v) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_quality_speed(meta, out):
    ppl = load_ppl(meta)
    e2e = load_e2e(meta)
    levels = [30, 50, 60, 70, 80, 90]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FULL_WIDTH, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )

    dense_ppl = ppl["dense"]

    # (a) quality
    for method in ["magnitude", "wanda", "sparsegpt"]:
        xs, ys = [0.0], [dense_ppl]
        for s in levels:
            name = f"{method}-u{s}"
            if name in ppl:
                xs.append(s / 100)
                ys.append(ppl[name])
        ax1.plot(xs, ys, marker="o", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax1.annotate(
            METHOD_LABELS[method], (xs[-1], ys[-1]), xytext=(4, 0),
            textcoords="offset points", va="center", fontsize=7.5, color=INK,
        )

    ax1.set_yscale("log")
    ax1.axhline(dense_ppl, color=DENSE_REF, linewidth=0.9, linestyle=(0, (4, 3)))
    ax1.text(0.97, dense_ppl * 1.25, f"dense {dense_ppl:.1f}", ha="right", color=INK_2, fontsize=7.5)
    ax1.set_ylabel("WikiText-2 perplexity (log)")
    ax1.set_ylim(5, 2e8)
    ax1.legend(loc="upper left", ncol=1)

    # (b) speed, SparseGPT checkpoints, mlp_t, MLP layers in CSR
    series = [
        ("Decode (1 token)", "decode_speedup", 128, BLUE),
        ("Prefill, 128 tokens", "prefill_speedup", 128, ORANGE),
        ("Prefill, 512 tokens", "prefill_speedup", 512, AQUA),
    ]
    any_old = False
    for label, field, plen, color in series:
        xs, ys = [], []
        for s in levels:
            v, lk1 = e2e_speedup(e2e, f"sparsegpt-u{s}", "mlp", "mlp_t", plen, field)
            if v is not None:
                xs.append(s / 100)
                ys.append(v)
                any_old |= (lk1 is False and field == "prefill_speedup")
        ax2.plot(xs, ys, marker="o", color=color, label=label)

    ax2.set_ylabel("CSR speedup over dense")
    ax2.set_xlabel("Sparsity of the Transformer Linear layers")
    ax2.set_xlim(-0.02, 0.97)
    ax2.set_ylim(0, 1.6)
    ax2.legend(loc="upper left")
    speedup_label(ax2)
    ax1.set_xlim(-0.02, 0.97)

    # Region where the model is still usable (SparseGPT PPL < 2x dense)
    for ax in (ax1, ax2):
        ax.axvspan(0.0, 0.55, color="#2a78d6", alpha=0.06, linewidth=0, zorder=0)
    ax2.text(
        0.275, 0.08, "usable quality: SparseGPT PPL < 2x dense",
        ha="center", color=INK_2, fontsize=7.5,
    )

    ax1.text(-0.1, 1.0, "(a)", transform=ax1.transAxes, fontsize=9, va="top")
    ax2.text(-0.1, 1.0, "(b)", transform=ax2.transAxes, fontsize=9, va="top")

    if any_old:
        print("note: some prefill points in fig_quality_speed predate the lm_head fix")

    save(fig, out, "quality_speed.pdf")


def fig_baseline(meta, out):
    e2e = load_e2e(meta)
    levels = [30, 50, 60, 70, 80, 90]
    impls = [
        ("v0", "int64 indices (initial)", BLUE_RAMP[0]),
        ("int32", "int32 indices", BLUE_RAMP[2]),
        ("mlp_t", "int32 + transposed MLP", BLUE_RAMP[4]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5), sharey=True)
    panels = [
        ("Decode (1 token)", "decode_speedup", 128),
        ("Prefill, 128 tokens", "prefill_speedup", 128),
    ]

    for ax, (title, field, plen) in zip(axes, panels):
        for impl, label, color in impls:
            xs, ys = [], []
            for s in levels:
                v, _ = e2e_speedup(e2e, f"sparsegpt-u{s}", "all", impl, plen, field)
                if v is not None:
                    xs.append(s / 100)
                    ys.append(v)
            ax.plot(xs, ys, marker="o", color=color, label=label)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Sparsity")
        ax.set_xlim(0.25, 0.95)
        ax.set_ylim(0, 1.6)
        speedup_label(ax)

    axes[0].set_ylabel("CSR speedup over dense")
    axes[0].legend(loc="upper left")

    save(fig, out, "baseline_progress.pdf")


def fig_layers(meta, out):
    layers = load_layers(meta, cold=True)
    levels = [30, 50, 60, 70, 80, 90]
    ms = [1, 8, 32, 128, 512]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.6), sharey=True)
    panels = [
        ("csr_i32", "CSR, column-major activations"),
        ("csr_i32_rowmajor", "CSR, row-major activations"),
    ]

    for ax, (variant, title) in zip(axes, panels):
        for M, color in zip(ms, BLUE_RAMP):
            xs, ys = [], []
            for s in levels:
                v = layers.get((f"sparsegpt-u{s}", M, variant))
                if v is not None:
                    xs.append(s / 100)
                    ys.append(v)
            ax.plot(xs, ys, marker="o", color=color, label=f"M = {M}")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Sparsity")
        ax.set_xlim(0.25, 0.95)
        ax.set_yscale("log")
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
        ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
        ax.set_ylim(0.12, 6)
        speedup_label(ax)

    axes[0].set_ylabel("Speedup over dense (log)")
    axes[0].legend(loc="upper left", ncol=2, title="Activation rows M", title_fontsize=7.5)

    save(fig, out, "layers_cold.pdf")


def fig_patterns(meta, out):
    layers = load_layers(meta, cold=True)

    rows = [
        ("Unstructured (Magnitude)", "magnitude-u50"),
        ("Unstructured (Wanda)", "wanda-u50"),
        ("Unstructured (SparseGPT)", "sparsegpt-u50"),
        ("2:4 (SparseGPT)", "sparsegpt-nm2-4"),
        ("4:8 (SparseGPT)", "sparsegpt-nm4-8"),
        ("4x4 blocks (Wanda)", "wanda-block4x4-50"),
        ("16x16 blocks (Wanda)", "wanda-block16x16-50"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.7), sharey=True)

    for ax, M in zip(axes, [1, 128]):
        labels, real, rand = [], [], []
        for label, ckpt in rows:
            v = layers.get((ckpt, M, "csr"))
            r = layers.get((ckpt, M, "csr_random"))
            if v is None:
                continue
            labels.append(label)
            real.append(v)
            rand.append(r)

        ys = list(range(len(labels)))[::-1]
        ax.barh(
            [y + 0.19 for y in ys], real, height=0.36, color=BLUE,
            label="Pruned matrix",
        )
        ax.barh(
            [y - 0.19 for y in ys], rand, height=0.36, color=ORANGE,
            label="Random positions, same nnz",
        )
        for y, v in zip(ys, real):
            ax.text(v + 0.01, y + 0.19, f"{v:.2f}", va="center", fontsize=7, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.set_title(f"M = {M}", loc="left")
        ax.set_xlabel("CSR speedup over dense")
        ax.set_xlim(0, max(real + rand) * 1.25)
        ax.grid(axis="y", visible=False)

    axes[0].legend(loc="lower left", ncol=2, bbox_to_anchor=(-0.02, 1.08))

    save(fig, out, "patterns.pdf")


def fig_tma(meta, out):
    path = os.path.join(meta, "tma.csv")
    rows = read_csv(path)

    cats = [
        ("retiring", "Retiring", BLUE),
        ("bad_speculation", "Bad speculation", ORANGE),
        ("frontend_bound", "Frontend bound", AQUA),
        ("memory_bound", "Backend: memory", YELLOW),
        ("core_bound", "Backend: core", MAGENTA),
    ]
    formats = [
        ("dense", "dense"),
        ("csr", "CSR int64"),
        ("csr_i32", "CSR int32"),
        ("csr_i32_rowmajor", "CSR int32, row-major"),
    ]
    ckpts = [("sparsegpt-u50", "50%"), ("sparsegpt-u90", "90%")]

    index = {}
    for r in rows:
        ckpt = r["run"].split("/")[0]
        index[(ckpt, int(r["threads"]), int(r["m"]), r["format"])] = r

    fig, axes = plt.subplots(2, 1, figsize=(FULL_WIDTH, 5.0), sharex=True)

    for ax, M in zip(axes, [1, 128]):
        labels, data = [], []
        for ckpt, pct in ckpts:
            for fmt, fmt_label in formats:
                r = index.get((ckpt, 16, M, fmt))
                if r is None:
                    continue
                labels.append(f"{pct}  {fmt_label}" if fmt != "dense" else f"{pct}  dense")
                data.append(r)

        ys = list(range(len(labels)))[::-1]
        for y, r in zip(ys, data):
            left = 0.0
            for key, _, color in cats:
                w = float(r[key]) * 100
                ax.barh(y, w, left=left, height=0.7, color=color, edgecolor="white", linewidth=0.8)
                left += w
            ax.text(
                101, y, f"IPC {float(r['ipc']):.2f}  {float(r['ms_per_call']):.2f} ms",
                va="center", fontsize=6.5, color=INK_2,
            )
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlim(0, 100)
        ax.set_title(f"M = {M}, 16 threads", loc="left")
        ax.grid(axis="y", visible=False)

    axes[1].set_xlabel("Share of pipeline slots (%)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in cats]
    axes[0].legend(
        handles, [l for _, l, _ in cats], loc="lower left", ncol=5,
        bbox_to_anchor=(-0.02, 1.12), handlelength=1.2, columnspacing=1.2,
    )
    fig.subplots_adjust(hspace=0.3)

    save(fig, out, "tma.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="results-meta")
    parser.add_argument("--out", default="report/figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    fig_quality_speed(args.meta, args.out)
    fig_baseline(args.meta, args.out)
    fig_layers(args.meta, args.out)
    fig_patterns(args.meta, args.out)
    fig_tma(args.meta, args.out)


if __name__ == "__main__":
    main()
