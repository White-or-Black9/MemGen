"""Generate AAAI-ready MemGen result figures from results_data.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "results_data.csv"
METHODS_OVERALL = ["Recent-text", "Rolling summary", "BM25 top-2", "Dense E5 top-2", "Matched-16", "Ours"]
METHODS_EFFICIENCY = ["Recent-text", "Rolling summary", "BM25 top-2", "Dense E5 top-2", "Matched-16", "Ours"]
COLORS = {
    "Recent-text": "#34495E",
    "Rolling summary": "#B3B7BD",
    "BM25 top-2": "#898F98",
    "Dense E5 top-2": "#A6ABB3",
    "Matched-16": "#D0D3D8",
    "Ours": "#7B5AA6",
}


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 7.6,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.9,
    "ytick.labelsize": 7.0,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def load_rows() -> list[dict[str, str]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values(rows: list[dict[str, str]], figure: str, panel: str, metric: str, methods: list[str]) -> dict[str, tuple[float, float | None]]:
    selected = [r for r in rows if r["figure"] == figure and r["panel"] == panel and r["metric"] == metric]
    out = {}
    for r in selected:
        std = float(r["std"]) if r["std"].strip() else None
        out[r["method"]] = (float(r["value"]), std)
    if set(out) != set(methods):
        raise ValueError(f"Incomplete values for Figure {figure}, panel {panel}, metric {metric}")
    return out


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#D9DEE6", linewidth=0.5, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7B8491")
    ax.spines["bottom"].set_color("#7B8491")
    ax.tick_params(axis="both", length=2.5, width=0.6, color="#657080", pad=2)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(0.0, 1.045, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=8.5)
    ax.text(0.115, 1.045, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)


def save(fig: plt.Figure, name: str) -> None:
    base = HERE / name
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def figure2(rows: list[dict[str, str]]) -> None:
    specs = [
        ("a", "Exact Match", "Exact Match ↑", (0, 0.26), ".3f"),
        ("b", "Recall", "Recall ↑", (0, 0.30), ".3f"),
        ("c", "Format failures", "Format Failures ↓", (0, 410), ".1f"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.75), sharey=True, constrained_layout=True)
    y = np.arange(len(METHODS_OVERALL))
    for ax, (letter, metric, title, xlim, formatter) in zip(axes, specs):
        metric_values = values(rows, "2", letter, metric, METHODS_OVERALL)
        means = [metric_values[m][0] for m in METHODS_OVERALL]
        bars = ax.barh(y, means, height=0.62, color=[COLORS[m] for m in METHODS_OVERALL], edgecolor="none", zorder=2)
        for idx, method in enumerate(METHODS_OVERALL):
            mean, std = metric_values[method]
            if std and std > 0:
                ax.errorbar(mean, idx, xerr=std, fmt="none", ecolor="#303945", elinewidth=0.8,
                            capsize=2.0, capthick=0.8, zorder=3)
            offset = (xlim[1] - xlim[0]) * 0.008
            label_x = mean + (std if std and std > 0 else 0) + offset
            ax.text(label_x, idx, format(mean, formatter), va="center", ha="left", fontsize=6.7, color="#253247")
        ax.set_xlim(*xlim)
        ax.set_yticks(y, METHODS_OVERALL)
        ax.invert_yaxis()
        style_axis(ax)
        panel_label(ax, f"({letter})", title)
        if metric == "Format failures":
            ax.set_xticks([0, 100, 200, 300, 400])
        else:
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    save(fig, "figure2_overall_results")


def figure3(rows: list[dict[str, str]]) -> None:
    seconds = values(rows, "3", "a", "Seconds per question", METHODS_EFFICIENCY)
    em = values(rows, "3", "a", "Exact Match", METHODS_EFFICIENCY)
    memory = values(rows, "3", "b", "Peak incremental GPU memory (GiB)", METHODS_EFFICIENCY)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.85), constrained_layout=True, gridspec_kw={"width_ratios": [1.05, 1]})

    ax = axes[0]
    label_offsets = {
        "Recent-text": (-6, 6, "right"),
        "Rolling summary": (6, 6, "left"),
        "BM25 top-2": (-6, 7, "right"),
        "Dense E5 top-2": (6, 10, "left"),
        "Matched-16": (6, 4, "left"),
        "Ours": (6, 5, "left"),
    }
    for method in METHODS_EFFICIENCY:
        x, y = seconds[method][0], em[method][0]
        size = 45 if method == "Ours" else 28
        edge = "#4E2B70" if method == "Ours" else "#FFFFFF"
        linewidth = 0.8 if method == "Ours" else 0.55
        ax.scatter(x, y, s=size, color=COLORS[method], edgecolor=edge, linewidth=linewidth, zorder=3)
        dx, dy, ha = label_offsets[method]
        ax.annotate(method, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.8, color="#253247")
    ax.set_xscale("log")
    ax.set_xlim(0.65, 7.2)
    ax.set_ylim(0, 0.215)
    ax.set_xlabel("Seconds per question ↓")
    ax.set_ylabel("Exact Match ↑")
    ax.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([0.7, 1, 2, 5]))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    style_axis(ax)
    panel_label(ax, "(a)", "Exact Match vs. Seconds per Question")

    ax = axes[1]
    y = np.arange(len(METHODS_EFFICIENCY))
    means = [memory[m][0] for m in METHODS_EFFICIENCY]
    ax.barh(y, means, height=0.62, color=[COLORS[m] for m in METHODS_EFFICIENCY], edgecolor="none", zorder=2)
    for idx, method in enumerate(METHODS_EFFICIENCY):
        value = memory[method][0]
        ax.text(value * 1.14, idx, f"{value:.3f}", va="center", ha="left", fontsize=6.8, color="#253247")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 20)
    ax.set_yticks(y, METHODS_EFFICIENCY)
    ax.invert_yaxis()
    ax.set_xlabel("Peak incremental GPU memory (GiB) ↓")
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    style_axis(ax)
    panel_label(ax, "(b)", "Peak Incremental GPU Memory")
    save(fig, "figure4_efficiency")


if __name__ == "__main__":
    data = load_rows()
    figure2(data)
    figure3(data)
