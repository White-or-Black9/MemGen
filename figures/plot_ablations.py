"""Generate query-time memory-use ablation results for an AAAI paper."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "ablation_results.csv"
METHODS = ["No retrieval", "No conditioning", "Direct injection", "Ours"]
COLORS = {
    "No retrieval": "#D0D3D8",
    "No conditioning": "#E0E2E5",
    "Direct injection": "#63788D",
    "Ours": "#7B5AA6",
}


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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
    "hatch.linewidth": 0.55,
})


def load_data() -> list[dict[str, str]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_values(rows: list[dict[str, str]], panel: str, metric: str) -> dict[str, tuple[float, float]]:
    out = {
        row["method"]: (float(row["value"]), float(row["std"]))
        for row in rows
        if row["panel"] == panel and row["metric"] == metric
    }
    if set(out) != set(METHODS):
        raise ValueError(f"Missing data for panel {panel}: {metric}")
    return out


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#D9DEE6", linewidth=0.5, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7B8491")
    ax.spines["bottom"].set_color("#7B8491")
    ax.tick_params(axis="both", length=2.5, width=0.6, color="#657080", pad=2)


def add_panel_heading(ax: plt.Axes, letter: str, heading: str) -> None:
    ax.text(0.0, 1.045, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, fontweight="bold")
    ax.text(0.115, 1.045, heading, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)


def draw_bar(ax: plt.Axes, y: int, value: float, method: str) -> None:
    hatch = "///" if method == "No conditioning" else None
    edge = "#888F98" if method == "No conditioning" else "none"
    ax.barh(y, value, height=0.62, color=COLORS[method], edgecolor=edge, linewidth=0.55, hatch=hatch, zorder=2)


def main() -> None:
    specs = [
        ("a", "Exact Match", "Exact Match ↑", (0, 0.26), ".3f"),
        ("b", "Recall", "Recall ↑", (0, 0.28), ".3f"),
        ("c", "Format failures", "Format Failures ↓", (0, 430), ".1f"),
    ]
    rows = load_data()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), sharey=True, constrained_layout=True)
    y = np.arange(len(METHODS))

    for ax, (letter, metric, title, xlim, formatter) in zip(axes, specs):
        data = metric_values(rows, letter, metric)
        for idx, method in enumerate(METHODS):
            mean, std = data[method]
            draw_bar(ax, idx, mean, method)
            if std > 0:
                ax.errorbar(mean, idx, xerr=std, fmt="none", ecolor="#303945", elinewidth=0.8,
                            capsize=2.0, capthick=0.8, zorder=3)
            offset = (xlim[1] - xlim[0]) * 0.008
            label_x = mean + (std if std > 0 else 0) + offset
            ax.text(label_x, idx, format(mean, formatter), va="center", ha="left", fontsize=6.7, color="#253247")
        ax.set_xlim(*xlim)
        ax.set_yticks(y, METHODS)
        ax.invert_yaxis()
        style_axis(ax)
        add_panel_heading(ax, letter, title)
        if metric == "Format failures":
            ax.set_xticks([0, 100, 200, 300, 400])
        else:
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

    output = HERE / "figure3_query_time_ablations"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output.with_suffix(".png"), dpi=270, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    main()
