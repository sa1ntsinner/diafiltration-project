"""
dfp.viz.style
=============
House style for every figure in the report – Matplotlib only, so the whole
graphical output stays inside the package set allowed by the task sheet.

Design decisions
----------------
* One accent colour per *controller*, kept identical across all figures, so a
  reader recognises a curve without reading the legend twice.
* Specification and constraint information is drawn as *annotated regions*
  rather than bare horizontal lines: a red hatch band above ``cL_max``, a green
  band inside the product window.  A glance is enough to see whether a batch
  is on spec.
* No chart junk: no top/right spines, hairline grid behind the data, units in
  the axis label, values in the annotation.
"""

from __future__ import annotations

from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ["PALETTE", "CTRL_COLORS", "use_style", "figure", "annotate_value",
           "spec_band", "limit_band", "phase_span", "finish_marker", "save"]

#: Core palette (dark ink, lawn green, orange, yellow + three cool accents).
PALETTE: Dict[str, str] = {
    "ink": "#1B1C1E",
    "green": "#79BD03",
    "orange": "#EA6700",
    "yellow": "#FEC500",
    "blue": "#2A6F97",
    "violet": "#6C4AB6",
    "teal": "#009C8F",
    "red": "#C1121F",
    "grey": "#8A8F98",
    "light": "#EDEFF2",
}

#: Fixed colour per controller family – used by every figure.
CTRL_COLORS: Dict[str, str] = {
    "analytic": PALETTE["ink"],
    "optimum": PALETTE["ink"],
    "threshold": PALETTE["orange"],
    "tracking": PALETTE["blue"],
    "tracking_scaled": PALETTE["violet"],
    "l1_time": PALETTE["teal"],
    "min_time": PALETTE["green"],
    "economic": PALETTE["yellow"],
    "multistage": PALETTE["violet"],
    "adaptive": PALETTE["teal"],
    "backoff": PALETTE["grey"],
}

_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8.0,
    "axes.labelsize": 9.5,
    "axes.labelcolor": PALETTE["ink"],
    "axes.edgecolor": "#C7CCD3",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "both",
    "grid.color": "#E3E6EA",
    "grid.linewidth": 0.7,
    "xtick.color": PALETTE["grey"],
    "ytick.color": PALETTE["grey"],
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.handlelength": 1.8,
    "lines.linewidth": 1.9,
    "lines.solid_capstyle": "round",
    "mathtext.fontset": "dejavusans",
    "axes.prop_cycle": mpl.cycler(color=[
        PALETTE["green"], PALETTE["orange"], PALETTE["blue"], PALETTE["violet"],
        PALETTE["teal"], PALETTE["yellow"], PALETTE["red"], PALETTE["grey"]]),
}


def use_style() -> None:
    """Activate the house style (idempotent)."""
    mpl.rcParams.update(_RC)


def figure(nrows: int = 1, ncols: int = 1, *, width: float = 7.2,
           height: float | None = None, sharex: bool = False, **kw):
    """A styled figure sized for a two-column report page."""
    use_style()
    height = height if height is not None else 2.4 * nrows + 0.5
    fig, ax = plt.subplots(nrows, ncols, figsize=(width, height), sharex=sharex,
                           constrained_layout=True, **kw)
    return fig, ax


# ── decorations ─────────────────────────────────────────────────────────────
def limit_band(ax, level: float, *, label: str | None = None, top: float | None = None):
    """Hatched forbidden region above a hard constraint."""
    ylim = ax.get_ylim()
    top = top if top is not None else max(ylim[1], level * 1.08)
    ax.axhspan(level, top, facecolor=PALETTE["red"], alpha=0.07, zorder=0)
    ax.axhline(level, color=PALETTE["red"], lw=1.1, ls=(0, (5, 2)), zorder=1)
    if label:
        ax.annotate(label, xy=(0.995, level), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8, color=PALETTE["red"])
    ax.set_ylim(ylim[0], top)


def spec_band(ax, level: float, *, label: str | None = None, side: str = "below",
              color: str | None = None):
    """Shade the admissible product window and mark the set point."""
    color = color or PALETTE["green"]
    lo, hi = ax.get_ylim()
    if side == "below":
        ax.axhspan(lo, level, facecolor=color, alpha=0.08, zorder=0)
    else:
        ax.axhspan(level, hi, facecolor=color, alpha=0.08, zorder=0)
    ax.axhline(level, color=color, lw=1.1, ls=(0, (4, 2)), zorder=1)
    if label:
        ax.annotate(label, xy=(0.995, level), xycoords=("axes fraction", "data"),
                    ha="right", va="top" if side == "below" else "bottom",
                    fontsize=8, color=color)
    ax.set_ylim(lo, hi)


def phase_span(ax, t0: float, t1: float, *, label: str | None = None,
               color: str | None = None):
    """Shade a time window (filter-cake tear, tariff peak, …)."""
    color = color or PALETTE["yellow"]
    ax.axvspan(t0, t1, facecolor=color, alpha=0.22, zorder=0, lw=0)
    if label:
        ax.annotate(label, xy=(0.5 * (t0 + t1), 0.97), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=8, color="#8a6d00")


def finish_marker(ax, t_end: float, *, color: str, label: str | None = None,
                  y: float = 0.02):
    """Vertical marker at the end of a batch."""
    ax.axvline(t_end, color=color, lw=1.0, ls=":", alpha=0.9, zorder=1)
    if label:
        ax.annotate(label, xy=(t_end, y), xycoords=("data", "axes fraction"),
                    rotation=90, ha="right", va="bottom", fontsize=7.5, color=color)


def annotate_value(ax, x, y, text, *, color=None, dx=6, dy=6, **kw):
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                color=color or PALETTE["ink"], fontsize=8.5, fontweight="bold", **kw)


def save(fig, path, *, formats=("png", "pdf")) -> list:
    """Save a figure in every requested format; returns the written paths."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        p = path.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out
