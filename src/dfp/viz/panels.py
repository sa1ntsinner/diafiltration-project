"""
dfp.viz.panels
==============
Reusable, self-documenting figure builders.  Every study in
:mod:`dfp.experiments` composes its figure from these blocks, so all figures
share axes, colours and annotations.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import NOMINAL, ProcessParams
from ..simulate import ClosedLoopResult
from .style import (CTRL_COLORS, PALETTE, annotate_value, figure, finish_marker,
                    limit_band, phase_span, spec_band, use_style)

__all__ = ["trajectory_figure", "input_axis", "metric_table_figure",
           "horizon_figure", "pareto_figure", "histogram_figure",
           "process_schematic", "tariff_figure", "switching_figure",
           "estimator_figure"]


_FALLBACK_COLORS = [PALETTE[k] for k in ("green", "orange", "blue", "violet",
                                         "teal", "yellow", "red", "grey", "ink")]


def _pick_colors(labels: Sequence[str]) -> List[str]:
    """Assign one distinct colour per curve.

    A controller keeps the colour registered in :data:`CTRL_COLORS` (so the same
    controller looks the same in every figure of the report), unless that colour
    is already taken in *this* figure - e.g. when the same MPC is shown on two
    different plants.  Then the next free accent is used, which guarantees that
    two curves are never drawn in the same colour.
    """
    used: set = set()
    out: List[str] = []
    for lab in labels:
        low = lab.lower().replace("-", "_").replace(" ", "_")
        col = next((c for k, c in CTRL_COLORS.items() if k in low and c not in used),
                   None)
        if col is None:
            col = next((c for c in _FALLBACK_COLORS if c not in used),
                       _FALLBACK_COLORS[len(out) % len(_FALLBACK_COLORS)])
        used.add(col)
        out.append(col)
    return out


# ─────────────────────────────────────────────────────────────────────────────
def trajectory_figure(
    results: Sequence[ClosedLoopResult],
    *,
    title: str = "",
    params: ProcessParams = NOMINAL,
    reference: Optional[Dict[str, np.ndarray]] = None,
    reference_label: str = "time-optimal solution",
    tear_window: Optional[Tuple[float, float]] = None,
    colors: Optional[Sequence[str]] = None,
    show_volume: bool = True,
    log_cL: bool = True,
    mark_finish: bool = True,
    width: float = 7.4,
):
    """Stacked trajectory plot: ``cP``, ``cL``, ``V`` and ``u`` against time.

    Conventions used throughout the report
    --------------------------------------
    * ``cP`` panel – solid line = set point ``cP_f``; the light red band above
      it marks **over-concentration**, which is off-spec even though
      ``cP > cP_f`` (the specification is an equality).
    * ``cL`` panel – logarithmic, because the washing phase is exponential; red
      hatch = crystallisation limit, green = product window.
    * legend – carries the achieved batch time, so no extra table is needed.
    """
    use_style()
    n = 4 if show_volume else 3
    fig, axes = figure(n, 1, width=width, height=1.72 * n + 1.15, sharex=True)
    ax_cP, ax_cL = axes[0], axes[1]
    ax_V = axes[2] if show_volume else None
    ax_u = axes[-1]
    handles: list = []
    labels: list = []

    if reference is not None:
        t_ref = np.asarray(reference["t"]) / 3600.0
        ln, = ax_cP.plot(t_ref, reference["cP"], color=PALETTE["ink"], lw=1.4,
                         ls=(0, (5, 2)), zorder=4)
        ax_cL.plot(t_ref, reference["cL"], color=PALETTE["ink"], lw=1.4,
                   ls=(0, (5, 2)), zorder=4)
        if ax_V is not None and "V" in reference:
            ax_V.plot(t_ref, np.asarray(reference["V"]) * 1e3, color=PALETTE["ink"],
                      lw=1.4, ls=(0, (5, 2)), zorder=4)
        if "u" in reference:
            ax_u.plot(t_ref, reference["u"], color=PALETTE["ink"], lw=1.4,
                      ls=(0, (5, 2)), zorder=4)
        handles.append(ln)
        labels.append(f"{reference_label}  —  {t_ref[-1]:.3f} h")

    auto = _pick_colors([r.label for r in results])
    for i, r in enumerate(results):
        col = colors[i] if colors is not None else auto[i]
        ln, = ax_cP.plot(r.t_h, r.cP, color=col)
        ax_cL.plot(r.t_h, np.maximum(r.cL, 1e-3), color=col)
        if ax_V is not None:
            ax_V.plot(r.t_h, r.V * 1e3, color=col)
        input_axis(ax_u, r, color=col)
        if mark_finish and r.finished:
            for ax in axes:
                ax.axvline(r.batch_time_h, color=col, lw=0.9, ls=":", alpha=0.8,
                           zorder=1)
        tag = (f"{r.batch_time_h:.3f} h" if r.finished else "not finished")
        flag = "" if r.spec_ok else "  ⚠"
        handles.append(ln)
        labels.append(f"{r.label}  —  {tag}{flag}")

    # ── protein panel ─────────────────────────────────────────────────────
    cP_top = max(params.cP_f * 1.30,
                 max((float(r.cP.max()) for r in results), default=params.cP_f) * 1.08)
    ax_cP.set_ylim(0, cP_top)
    ax_cP.axhspan(params.cP_f * 1.01, cP_top, facecolor=PALETTE["red"], alpha=0.06,
                  zorder=0)
    ax_cP.axhline(params.cP_f, color=PALETTE["green"], lw=1.2, ls=(0, (4, 2)), zorder=2)
    ax_cP.annotate(f"$c_{{P,f}}={params.cP_f:g}$  —  over-concentration above",
                   xy=(0.008, params.cP_f), xycoords=("axes fraction", "data"),
                   ha="left", va="bottom", fontsize=8, color=PALETTE["green"])
    ax_cP.set_ylabel(r"$c_P$  [mol m$^{-3}$]")
    ax_cP.set_title(title or "closed-loop trajectories")

    # ── lactose panel ─────────────────────────────────────────────────────
    cl_top = max(params.cL_max * 1.15,
                 max((float(r.cL.max()) for r in results), default=params.cL_max) * 1.2)
    if log_cL:
        ax_cL.set_yscale("log")
        ax_cL.set_ylim(max(1.0, params.cL_f * 0.35), cl_top)
    else:
        ax_cL.set_ylim(0, cl_top)
    lo = ax_cL.get_ylim()[0]
    ax_cL.axhspan(params.cL_max, cl_top, facecolor=PALETTE["red"], alpha=0.08, zorder=0)
    ax_cL.axhline(params.cL_max, color=PALETTE["red"], lw=1.2, ls=(0, (5, 2)), zorder=2)
    ax_cL.annotate(f"$c_{{L,\\max}}={params.cL_max:g}$",
                   xy=(0.995, params.cL_max), xycoords=("axes fraction", "data"),
                   ha="right", va="bottom", fontsize=8, color=PALETTE["red"])
    ax_cL.axhspan(lo, params.cL_f, facecolor=PALETTE["green"], alpha=0.10, zorder=0)
    ax_cL.axhline(params.cL_f, color=PALETTE["green"], lw=1.2, ls=(0, (4, 2)), zorder=2)
    ax_cL.annotate(f"$c_{{L,f}}={params.cL_f:g}$",
                   xy=(0.995, params.cL_f), xycoords=("axes fraction", "data"),
                   ha="right", va="bottom", fontsize=8, color=PALETTE["green"])
    ax_cL.set_ylabel(r"$c_L$  [mol m$^{-3}$]")

    # ── volume panel ──────────────────────────────────────────────────────
    if ax_V is not None:
        ax_V.set_ylabel(r"$V$  [L]")
        ax_V.axhline(params.V_f * 1e3, color=PALETTE["grey"], lw=0.9, ls=":")
        ax_V.annotate(f"$V_f={params.V_f*1e3:.0f}$ L", xy=(0.995, params.V_f * 1e3),
                      xycoords=("axes fraction", "data"), ha="right", va="bottom",
                      fontsize=8, color=PALETTE["grey"])

    ax_u.set_ylabel(r"$u = d/p$  [–]")
    ax_u.set_xlabel("time  [h]")
    ax_u.set_ylim(-0.06, 1.10)

    if tear_window is not None:
        for ax in axes:
            phase_span(ax, *tear_window,
                       label="filter-cake tear" if ax is ax_u else None)

    fig.legend(handles, labels, loc="outside lower center",
               ncol=min(2, max(1, len(labels))), fontsize=8.4)
    return fig, axes


def input_axis(ax, r: ClosedLoopResult, *, color: str) -> None:
    """Piecewise-constant input drawn on its true (possibly non-uniform) grid."""
    t = np.repeat(r.t_h, 2)[1:-1]
    u = np.repeat(r.u, 2)
    ax.plot(t, u, color=color, lw=1.7, solid_joinstyle="miter")


# ─────────────────────────────────────────────────────────────────────────────
def horizon_figure(runs: Dict[int, ClosedLoopResult], *, title: str,
                   params: ProcessParams = NOMINAL, reference_T: Optional[float] = None):
    """Trajectories for several prediction horizons + batch-time bar chart."""
    use_style()
    fig, axes = figure(2, 2, width=7.4, height=5.0)
    (ax_cP, ax_cL), (ax_u, ax_bar) = axes
    cols = [PALETTE["blue"], PALETTE["teal"], PALETTE["violet"], PALETTE["orange"]]
    for i, (N, r) in enumerate(sorted(runs.items())):
        c = cols[i % len(cols)]
        ax_cP.plot(r.t_h, r.cP, color=c, label=f"$N={N}$")
        ax_cL.plot(r.t_h, r.cL, color=c)
        input_axis(ax_u, r, color=c)
    ax_cP.set_ylabel(r"$c_P$  [mol m$^{-3}$]")
    ax_cP.set_ylim(0, params.cP_f * 1.25)
    spec_band(ax_cP, params.cP_f, label="$c_{P,f}$", side="above")
    ax_cP.legend(loc="lower right")
    ax_cP.set_title(title)
    ax_cL.set_ylabel(r"$c_L$  [mol m$^{-3}$]")
    ax_cL.set_yscale("log")
    ax_cL.axhline(params.cL_f, color=PALETTE["green"], ls=(0, (4, 2)), lw=1.1)
    ax_cL.annotate("$c_{L,f}$", xy=(0.99, params.cL_f), xycoords=("axes fraction", "data"),
                   ha="right", va="bottom", fontsize=8, color=PALETTE["green"])
    ax_cL.set_title("lactose (log scale)")
    ax_u.set_ylabel(r"$u$  [–]"); ax_u.set_xlabel("time  [h]"); ax_u.set_ylim(-0.06, 1.09)
    ax_cP.set_xlabel("time  [h]"); ax_cL.set_xlabel("time  [h]")

    names = [f"N={N}" for N in sorted(runs)]
    vals = [runs[N].batch_time_h if runs[N].finished else np.nan for N in sorted(runs)]
    fails = [not runs[N].finished for N in sorted(runs)]
    bars = ax_bar.bar(names, [v if np.isfinite(v) else params.t_max / 3600 for v in vals],
                      color=[PALETTE["red"] if f else PALETTE["green"] for f in fails],
                      width=0.6)
    for b, v, f in zip(bars, vals, fails):
        ax_bar.annotate("not finished" if f else f"{v:.2f} h",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, fontweight="bold",
                        color=PALETTE["red"] if f else PALETTE["ink"])
    if reference_T is not None:
        ax_bar.axhline(reference_T, color=PALETTE["ink"], ls="--", lw=1.2)
        ax_bar.annotate(f"optimum {reference_T:.3f} h", xy=(0.02, reference_T),
                        xycoords=("axes fraction", "data"), va="bottom", fontsize=8)
    ax_bar.set_ylabel("batch time  [h]")
    ax_bar.set_title("batch time vs. horizon")
    ax_bar.grid(axis="x", visible=False)
    return fig, axes


def metric_table_figure(rows: List[List[str]], header: Sequence[str], *,
                        title: str = "", width: float = 7.4,
                        highlight: Optional[Sequence[int]] = None):
    """Render a small results table as a figure (goes straight into the report)."""
    use_style()
    h = 0.30 * (len(rows) + 1) + 0.55
    fig, ax = figure(width=width, height=h)
    ax.axis("off")
    ax.set_title(title)
    #  bbox=[0,0,1,1] makes the table fill the axes exactly, so the figure height
    #  alone controls the row height and no empty band is left below the table
    tbl = ax.table(cellText=rows, colLabels=list(header), cellLoc="center",
                   bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.6)
    ncol = len(header)
    for j in range(ncol):
        cell = tbl[0, j]
        cell.set_facecolor(PALETTE["ink"])
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")
    for i in range(1, len(rows) + 1):
        for j in range(ncol):
            cell = tbl[i, j]
            cell.set_edgecolor("#DCE0E5")
            if highlight and (i - 1) in highlight:
                cell.set_facecolor("#F0F7DF")
            elif i % 2 == 0:
                cell.set_facecolor("#F7F8FA")
    return fig, ax


def pareto_figure(points: Sequence[Tuple[float, float, str]], *, xlabel: str,
                  ylabel: str, title: str = "", annotate: bool = True):
    use_style()
    fig, ax = figure(width=6.4, height=4.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=PALETTE["grey"], lw=1.1, ls="-", zorder=1, alpha=0.7)
    ax.scatter(xs, ys, s=52, color=PALETTE["green"], edgecolor=PALETTE["ink"],
               linewidth=0.8, zorder=3)
    if annotate:
        for x, y, lab in points:
            annotate_value(ax, x, y, lab, dx=7, dy=5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    return fig, ax


def histogram_figure(data: Dict[str, np.ndarray], *, xlabel: str, title: str = "",
                     vline: Optional[float] = None, vline_label: str = ""):
    use_style()
    fig, ax = figure(width=6.8, height=3.6)
    cols = [PALETTE["green"], PALETTE["orange"], PALETTE["blue"], PALETTE["violet"]]
    for i, (name, arr) in enumerate(data.items()):
        ax.hist(arr, bins=22, alpha=0.6, label=name, color=cols[i % len(cols)],
                edgecolor="white", linewidth=0.6)
    if vline is not None:
        ax.axvline(vline, color=PALETTE["red"], ls="--", lw=1.3)
        ax.annotate(vline_label, xy=(vline, 0.96), xycoords=("data", "axes fraction"),
                    rotation=90, ha="right", va="top", fontsize=8, color=PALETTE["red"])
    ax.set_xlabel(xlabel); ax.set_ylabel("number of plants"); ax.set_title(title)
    ax.legend()
    return fig, ax


def tariff_figure(lam: Callable[[float], float], *, title: str = "day-ahead tariff"):
    use_style()
    fig, ax = figure(width=6.4, height=2.8)
    h = np.linspace(0, 24, 1441)
    price = np.array([lam(x * 3600.0) for x in h])
    ax.fill_between(h, 0, price, color=PALETTE["yellow"], alpha=0.35)
    ax.plot(h, price, color=PALETTE["orange"])
    ax.set_xlabel("hour of day"); ax.set_ylabel("price  [€ kWh$^{-1}$]")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_title(title)
    return fig, ax


def switching_figure(cP: np.ndarray, price: np.ndarray, *, params: ProcessParams = NOMINAL):
    """The linear-programming argument behind the analytic optimum."""
    use_style()
    fig, ax = figure(width=6.4, height=3.2)
    ax.plot(cP, price / 3600.0, color=PALETTE["green"])
    i = int(np.argmin(price))
    ax.scatter([cP[i]], [price[i] / 3600.0], s=60, color=PALETTE["orange"],
               edgecolor=PALETTE["ink"], zorder=4)
    annotate_value(ax, cP[i], price[i] / 3600.0,
                   f"cheapest washing at $c_P={cP[i]:.0f}$", dx=-160, dy=6)
    ax.set_xlabel(r"$c_P$  [mol m$^{-3}$]")
    ax.set_ylabel(r"$M_P/(p\,c_P r_L)$  [h]")
    ax.set_title("price of one unit of washing along the reachable set")
    return fig, ax


def estimator_figure(kappa_logs: Dict[str, Sequence[float]], truths: Dict[str, float],
                     *, dt_h: float, title: str = "online identification of $k_{M,L}$"):
    use_style()
    fig, ax = figure(width=6.8, height=3.6)
    cols = [PALETTE["green"], PALETTE["blue"], PALETTE["violet"], PALETTE["orange"]]
    for i, (name, log) in enumerate(kappa_logs.items()):
        t = np.arange(len(log)) * dt_h
        ax.plot(t, log, color=cols[i % len(cols)], label=name)
        if name in truths:
            ax.axhline(truths[name], color=cols[i % len(cols)], ls=":", lw=1.1)
    ax.set_xlabel("time  [h]")
    ax.set_ylabel(r"$\hat k_{M,L}/k_{M,L}^{\rm nom}$")
    ax.set_ylim(0, 1.35)
    ax.set_title(title)
    ax.legend(ncol=2)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
def process_schematic(params: ProcessParams = NOMINAL):
    """Flow sheet of the batch-diafiltration plant, drawn with Matplotlib.

    Reproduces Figure 1 of the task sheet and adds the model symbols, so the
    report is self-contained.
    """
    import matplotlib.patches as mp

    use_style()
    fig, ax = figure(width=7.4, height=3.9)
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
    ink, green, orange, blue = (PALETTE["ink"], PALETTE["green"], PALETTE["orange"],
                                PALETTE["blue"])

    def box(x, y, w, h, fc, ec=ink, lw=1.3, r=1.4, alpha=1.0, **kw):
        patch = mp.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                  facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha,
                                  **kw)
        ax.add_patch(patch)
        return patch

    def arrow(x0, y0, x1, y1, color=ink, lw=1.6, ls="-", label=None, dy=1.6, dx=0.0):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    linestyle=ls, shrinkA=0, shrinkB=0,
                                    mutation_scale=13))
        if label:
            ax.text(0.5 * (x0 + x1) + dx, 0.5 * (y0 + y1) + dy, label, color=color,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # feed tank
    box(8, 12, 22, 26, "#F4F8EA", green, 1.6)
    ax.add_patch(mp.Rectangle((8.6, 12.6), 20.8, 15.5, facecolor=green, alpha=0.22, lw=0))
    ax.text(19, 40, "feed tank", ha="center", fontsize=10, fontweight="bold", color=ink)
    ax.text(19, 22.5, "$V,\\; M_L,\\; M_P$", ha="center", fontsize=11, color=ink)
    ax.text(19, 17.5, "$c_P=M_P/V$\n$c_L=M_L/V$", ha="center", va="center",
            fontsize=8.6, color="#4b5560")
    # stirrer
    ax.plot([19, 19], [38, 30], color=ink, lw=1.4)
    ax.plot([15.5, 22.5], [30, 30], color=ink, lw=2.2)

    # membrane module
    box(56, 20, 26, 15, "#FFF3E8", orange, 1.6)
    ax.text(69, 36.6, "membrane module", ha="center", fontsize=10, fontweight="bold",
            color=ink)
    for i in range(6):
        ax.plot([58.5 + i * 4.0, 58.5 + i * 4.0], [21.4, 33.6], color=orange, lw=1.1,
                alpha=0.75)
    ax.text(69, 27.5, "$A = %g\\,$m$^2$" % params.A, ha="center", fontsize=9, color=ink)

    # pump
    ax.add_patch(mp.Circle((43, 27.5), 4.6, facecolor="white", edgecolor=blue, lw=1.6))
    ax.plot([40.2, 45.6], [27.5, 27.5], color=blue, lw=1.5)
    ax.plot([43, 43], [24.6, 30.4], color=blue, lw=1.5)
    ax.text(43, 20.4, "recirculation\npump", ha="center", va="top", fontsize=8.4,
            color=blue)

    # flows
    arrow(30, 27.5, 38.4, 27.5, blue)
    arrow(47.6, 27.5, 56, 27.5, blue)
    arrow(82, 27.5, 97, 27.5, orange)
    ax.text(89.5, 30.4, "permeate  $p(t)$", ha="center", fontsize=9,
            fontweight="bold", color=orange)
    ax.text(89.5, 23.6, "$c_{L,p}=r_L c_L$", ha="center", fontsize=8.6, color=orange)
    # retentate return
    ax.annotate("", xy=(19, 38.2), xytext=(69, 47),
                arrowprops=dict(arrowstyle="-|>", color=green, lw=1.6,
                                connectionstyle="arc3,rad=0.18", mutation_scale=13))
    ax.plot([69, 69], [35, 47], color=green, lw=1.6)
    ax.text(45, 48.4, "retentate (recycle)", ha="center", fontsize=9, color=green,
            fontweight="bold")

    # solvent inlet
    arrow(2.5, 44, 2.5, 38.4, ink)
    ax.plot([2.5, 8.6], [38.4, 38.4], color=ink, lw=1.6)
    ax.text(3.6, 45.4, "solvent  $d(t)=u\\,p(t)$", ha="left", fontsize=9,
            fontweight="bold", color=ink)
    ax.add_patch(mp.Circle((2.5, 34.5), 2.4, facecolor="white", edgecolor=ink, lw=1.4))
    ax.plot([0.5, 4.5], [32.9, 36.1], color=ink, lw=1.3)
    ax.text(2.5, 29.6, "$0\\leq u\\leq 1$", fontsize=8.6, color=ink, ha="center")

    ax.text(50, 6.2,
            r"$p = kA\ln(c_g/c_P)$        $\dot V=(u-1)p$        "
            r"$\dot M_L=-r_L c_L p$        $\dot M_P=-r_P c_P p$",
            ha="center", fontsize=10, color=ink)
    ax.text(50, 1.6,
            r"$r_\bullet=\gamma/[\,1+(\gamma-1)e^{p/(k_M A)}\,]$,   "
            r"$\gamma=\alpha$ (lactose) or $\beta$ (protein)",
            ha="center", fontsize=8.6, color="#4b5560")
    return fig, ax
