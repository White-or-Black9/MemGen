#!/usr/bin/env python3
"""Render the paper's method architecture and frozen-bank protocol figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/memgen_matplotlib_config")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

COLORS = {
    "ink": "#28323C",
    "muted": "#66727D",
    "line": "#9AA6B2",
    "frozen": "#E8EBEF",
    "frozen_edge": "#7A8793",
    "blue": "#4E79A7",
    "blue_light": "#DCE8F4",
    "teal": "#3F8F8B",
    "teal_light": "#D9ECEA",
    "bank": "#EDF5F5",
    "red": "#B94A48",
    "red_light": "#F7E1DF",
    "white": "#FFFFFF",
    "phase": "#F6F7F9",
}

FIGURE_STEMS = ("fig1_method_architecture", "fig2_frozen_bank_protocol")
FORMATS = ("svg", "pdf", "tiff", "png")


def _setup_axis(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["white"])
    return fig, ax


def _box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 8.0,
    weight: str = "normal",
    sublabel: str | None = None,
    zorder: int = 2,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    center_y = y + height * (0.59 if sublabel else 0.5)
    ax.text(
        x + width / 2,
        center_y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        fontweight=weight,
        zorder=zorder + 1,
    )
    if sublabel:
        ax.text(
            x + width / 2,
            y + height * 0.25,
            sublabel,
            ha="center",
            va="center",
            fontsize=6.2,
            color=COLORS["muted"],
            zorder=zorder + 1,
        )
    return patch


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["ink"],
    width: float = 1.25,
    style: str = "-|>",
    connection: str = "arc3",
    zorder: int = 1,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=width,
        color=color,
        connectionstyle=connection,
        shrinkA=1,
        shrinkB=1,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def _lock(ax, center: tuple[float, float], scale: float = 1.0):
    x, y = center
    w, h = 0.020 * scale, 0.020 * scale
    ax.add_patch(
        Rectangle(
            (x - w / 2, y - h / 2),
            w,
            h,
            linewidth=0.8,
            edgecolor=COLORS["frozen_edge"],
            facecolor=COLORS["white"],
            zorder=5,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - w * 0.30, y + h * 0.25),
            w * 0.60,
            h * 0.55,
            boxstyle="round,pad=0.001,rounding_size=0.008",
            linewidth=0.8,
            edgecolor=COLORS["frozen_edge"],
            facecolor="none",
            zorder=4,
        )
    )


def _stop(ax, center: tuple[float, float], radius: float = 0.018):
    x, y = center
    ax.add_patch(
        Circle(
            (x, y),
            radius,
            facecolor=COLORS["white"],
            edgecolor=COLORS["red"],
            linewidth=1.5,
            zorder=6,
        )
    )
    ax.plot(
        [x - radius * 0.65, x + radius * 0.65],
        [y + radius * 0.65, y - radius * 0.65],
        color=COLORS["red"],
        linewidth=1.5,
        zorder=7,
    )


def _slots(ax, x: float, y: float, width: float, height: float):
    cols, rows = 8, 2
    gap = 0.004
    slot_w = (width - gap * (cols - 1)) / cols
    slot_h = (height - gap * (rows - 1)) / rows
    for row in range(rows):
        for col in range(cols):
            ax.add_patch(
                FancyBboxPatch(
                    (x + col * (slot_w + gap), y + (rows - row - 1) * (slot_h + gap)),
                    slot_w,
                    slot_h,
                    boxstyle="round,pad=0.001,rounding_size=0.003",
                    linewidth=0.55,
                    edgecolor=COLORS["teal"],
                    facecolor=COLORS["teal_light"] if (row + col) % 3 else COLORS["blue_light"],
                    zorder=4,
                )
            )


def draw_method_architecture():
    fig, ax = _setup_axis((7.20, 4.55))
    ax.text(0.02, 0.965, "a", fontsize=9, fontweight="bold", va="top")
    ax.text(
        0.055,
        0.965,
        "Inference-time session-local latent memory architecture",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )

    ax.add_patch(Rectangle((0.025, 0.57), 0.95, 0.31, facecolor=COLORS["phase"], edgecolor="none"))
    ax.text(0.04, 0.845, "Frozen MemGen inference path", fontsize=7.3, color=COLORS["muted"], fontweight="bold")

    _box(ax, (0.045, 0.665), 0.12, 0.095, "Input state", face=COLORS["white"], edge=COLORS["line"], fontsize=7.6)
    component_specs = [
        (0.225, "Trigger", "when to invoke"),
        (0.425, "Weaver", "generate memory"),
        (0.695, "Reasoner", "use latent support"),
    ]
    for x, label, sublabel in component_specs:
        _box(
            ax,
            (x, 0.65),
            0.15,
            0.125,
            label,
            face=COLORS["frozen"],
            edge=COLORS["frozen_edge"],
            fontsize=8.4,
            weight="bold",
            sublabel=sublabel,
        )
        _lock(ax, (x + 0.132, 0.762), 0.72)
    _box(ax, (0.89, 0.665), 0.075, 0.095, "Answer", face=COLORS["white"], edge=COLORS["line"], fontsize=7.6)

    for start, end in [((0.165, 0.712), (0.225, 0.712)), ((0.375, 0.712), (0.425, 0.712)), ((0.575, 0.712), (0.695, 0.712)), ((0.845, 0.712), (0.89, 0.712))]:
        _arrow(ax, start, end)

    ax.text(0.50, 0.60, "all learned parameters remain frozen", ha="center", fontsize=6.8, color=COLORS["muted"])

    _box(
        ax,
        (0.255, 0.115),
        0.49,
        0.27,
        "Session-local latent bank",
        face=COLORS["bank"],
        edge=COLORS["teal"],
        fontsize=8.8,
        weight="bold",
    )
    ax.text(0.50, 0.335, "Weaver space  •  max 16 slots  •  reset between sessions", ha="center", fontsize=6.8, color=COLORS["muted"])
    _slots(ax, 0.29, 0.195, 0.42, 0.085)
    ax.text(0.50, 0.155, "insert  |  matched update  |  bounded replacement", ha="center", fontsize=6.6, color=COLORS["teal"])

    _arrow(ax, (0.50, 0.65), (0.50, 0.385), color=COLORS["blue"], width=1.5)
    ax.text(0.515, 0.50, "construction-time\nlatent write", fontsize=6.8, color=COLORS["blue"], va="center")

    _box(ax, (0.79, 0.30), 0.17, 0.13, "Retrieve", face=COLORS["blue_light"], edge=COLORS["blue"], fontsize=8.2, weight="bold", sublabel="similarity × decay\nthreshold 0.05 → top-k 2")
    _arrow(ax, (0.745, 0.25), (0.79, 0.345), color=COLORS["blue"], width=1.5)
    _arrow(ax, (0.875, 0.43), (0.785, 0.65), color=COLORS["blue"], width=1.5, connection="arc3,rad=-0.18")
    ax.text(0.90, 0.515, "selected latents\nto Reasoner", fontsize=6.7, color=COLORS["blue"], ha="center")

    _box(ax, (0.79, 0.105), 0.17, 0.10, "Query state", face=COLORS["white"], edge=COLORS["line"], fontsize=7.8, sublabel="mean-pooled key")
    _arrow(ax, (0.875, 0.205), (0.875, 0.30), color=COLORS["line"], width=1.1)

    _arrow(ax, (0.555, 0.65), (0.60, 0.385), color=COLORS["red"], width=1.1, style="-|>", connection="arc3,rad=-0.08")
    _stop(ax, (0.58, 0.505), 0.017)
    ax.text(0.68, 0.525, "query-time writes blocked", fontsize=6.7, color=COLORS["red"], ha="center")

    ax.text(0.03, 0.045, "No retraining", fontsize=6.7, color=COLORS["muted"])
    ax.text(0.20, 0.045, "No global registry", fontsize=6.7, color=COLORS["muted"])
    ax.text(0.40, 0.045, "No cross-session sharing", fontsize=6.7, color=COLORS["muted"])
    ax.text(0.67, 0.045, "No forced fallback when no slot passes", fontsize=6.7, color=COLORS["muted"])
    return fig


def draw_frozen_bank_protocol():
    fig, ax = _setup_axis((7.20, 4.35))
    ax.text(0.02, 0.965, "b", fontsize=9, fontweight="bold", va="top")
    ax.text(
        0.055,
        0.965,
        "Frozen-context-bank evaluation protocol",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )

    ax.add_patch(Rectangle((0.025, 0.59), 0.47, 0.28, facecolor=COLORS["blue_light"], alpha=0.43, edgecolor="none"))
    ax.add_patch(Rectangle((0.515, 0.20), 0.46, 0.67, facecolor=COLORS["teal_light"], alpha=0.35, edgecolor="none"))
    ax.text(0.04, 0.83, "CONSTRUCTION PHASE", fontsize=7.2, fontweight="bold", color=COLORS["blue"])
    ax.text(0.53, 0.83, "QUERY PHASE — independent branches", fontsize=7.2, fontweight="bold", color=COLORS["teal"])

    _box(ax, (0.045, 0.675), 0.10, 0.09, "Reset", face=COLORS["white"], edge=COLORS["blue"], fontsize=7.7, sublabel="one context")
    _box(ax, (0.185, 0.66), 0.145, 0.12, "Ordered chunks", face=COLORS["white"], edge=COLORS["blue"], fontsize=7.7, sublabel="c₁ → c₂ → … → cₘ")
    _box(ax, (0.37, 0.66), 0.105, 0.12, "Build bank", face=COLORS["bank"], edge=COLORS["teal"], fontsize=7.7, sublabel="write / update")
    _arrow(ax, (0.145, 0.72), (0.185, 0.72), color=COLORS["blue"])
    _arrow(ax, (0.33, 0.72), (0.37, 0.72), color=COLORS["blue"])

    _box(ax, (0.35, 0.43), 0.205, 0.105, "Snapshot and freeze", face=COLORS["frozen"], edge=COLORS["frozen_edge"], fontsize=7.7, weight="bold", sublabel="immutable context bank")
    _arrow(ax, (0.425, 0.66), (0.452, 0.535), color=COLORS["frozen_edge"], width=1.4)
    _lock(ax, (0.535, 0.52), 0.75)

    branch_ys = [0.68, 0.47, 0.26]
    branch_labels = ["Question 1", "Question 2", "Question N"]
    for y, label in zip(branch_ys, branch_labels):
        _box(ax, (0.61, y), 0.115, 0.09, label, face=COLORS["white"], edge=COLORS["teal"], fontsize=7.2)
        _box(ax, (0.76, y), 0.09, 0.09, "Retrieve", face=COLORS["blue_light"], edge=COLORS["blue"], fontsize=7.0)
        _box(ax, (0.89, y), 0.07, 0.09, "Answer", face=COLORS["white"], edge=COLORS["teal"], fontsize=6.8)
        _arrow(ax, (0.555, 0.482), (0.61, y + 0.045), color=COLORS["frozen_edge"], connection="arc3,rad=0.10" if y > 0.48 else "arc3,rad=-0.10")
        _arrow(ax, (0.725, y + 0.045), (0.76, y + 0.045), color=COLORS["teal"])
        _arrow(ax, (0.85, y + 0.045), (0.89, y + 0.045), color=COLORS["teal"])
        ax.text(0.585, y + 0.092, "restore", fontsize=5.7, color=COLORS["muted"], ha="center")
        _arrow(ax, (0.915, y), (0.84, y - 0.045), color=COLORS["red"], width=0.95, connection="arc3,rad=0.2")
        _stop(ax, (0.857, y - 0.025), 0.012)

    ax.text(0.86, 0.15, "query-time write", fontsize=6.2, color=COLORS["red"], ha="center")
    ax.text(0.34, 0.355, "same snapshot for every question", fontsize=6.6, color=COLORS["frozen_edge"], ha="center")

    ax.add_patch(
        FancyBboxPatch(
            (0.06, 0.055),
            0.88,
            0.075,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=COLORS["phase"],
            edgecolor=COLORS["line"],
            linewidth=0.9,
        )
    )
    ax.text(0.19, 0.092, "Protocol invariants", fontsize=7.4, fontweight="bold", color=COLORS["ink"], ha="center", va="center")
    ax.text(0.47, 0.092, "query_write_count = 0", fontsize=7.1, color=COLORS["red"], ha="center", va="center")
    ax.text(0.75, 0.092, "bank_after_query = frozen_snapshot", fontsize=7.1, color=COLORS["teal"], ha="center", va="center")
    return fig


def _export(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in FORMATS:
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "tiff":
            kwargs["dpi"] = 600
        elif suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        if suffix == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n",
                encoding="utf-8",
            )
        paths.append(path)
    plt.close(fig)
    return paths


def render_all(output_dir: Path) -> list[Path]:
    outputs = []
    outputs.extend(_export(draw_method_architecture(), output_dir, FIGURE_STEMS[0]))
    outputs.extend(_export(draw_frozen_bank_protocol(), output_dir, FIGURE_STEMS[1]))
    return outputs


def expected_outputs(output_dir: Path) -> Iterable[Path]:
    for stem in FIGURE_STEMS:
        for suffix in FORMATS:
            yield output_dir / f"{stem}.{suffix}"


def check_outputs(output_dir: Path) -> list[str]:
    errors = []
    for path in expected_outputs(output_dir):
        if not path.is_file():
            errors.append(f"missing output: {path}")
            continue
        if path.stat().st_size <= 1_000:
            errors.append(f"output is unexpectedly small: {path}")
        if path.suffix == ".svg":
            svg = path.read_text(encoding="utf-8")
            if "<text" not in svg:
                errors.append(f"SVG does not contain editable text: {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        errors = check_outputs(args.output_dir)
        if errors:
            for error in errors:
                print(error)
            return 1
        print(f"validated {sum(1 for _ in expected_outputs(args.output_dir))} figure exports")
        return 0

    outputs = render_all(args.output_dir)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
