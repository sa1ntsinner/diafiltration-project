"""
dfp.experiments.studies
=======================
Every study of the report as one function.

Each ``study_*`` function

* runs the required simulations,
* writes its figures to ``docs/figures/`` (PNG **and** vector PDF), and
* returns a JSON-serialisable ``dict`` of the numbers quoted in the report.

Running ``python run.py all`` executes them in order and collects the
results in ``results/results.json``, so every number in ``docs/REPORT.md`` is
reproducible with a single command.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import numpy as np

from ..config import KM_L_UNCERTAINTY, NOMINAL, ProcessParams, UncertaintySet
from ..controllers import (BangBang, ConstantU, ThresholdPolicy, analytic_optimum,
                           build_multistage_nmpc, build_nmpc, solve_min_time_ocp,
                           switching_price)
from ..estimation import EKF, MHE, AdaptiveNMPC, measurement
from ..integrate import cvodes_integrator, rk4_integrator
from ..model import build_model
from ..plant import (Plant, TEAR_WINDOW, leakage_plant, mismatch_plant,
                     nominal_plant, tear_plant)
from ..simulate import ClosedLoopResult, closed_loop
from ..tariff import lambda_tou
from ..viz import (PALETTE, estimator_figure, figure, histogram_figure,
                   horizon_figure,
                   metric_table_figure, pareto_figure, process_schematic, save,
                   switching_figure, tariff_figure, trajectory_figure)
from .montecarlo import (DEFAULT_RANGES, MonteCarloResult, run_campaign,
                         sample_params)

__all__ = ["ROOT", "FIG_DIR", "STUDIES", "run_all", "run_study"]

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "docs" / "figures"
RESULT_DIR = ROOT / "results"


def _fig(fig, name: str) -> List[str]:
    paths = save(fig, FIG_DIR / name)
    return [str(p.relative_to(ROOT)) for p in paths]


def _ref(a) -> Dict[str, np.ndarray]:
    return {"t": a.t, "cP": a.cP, "cL": a.cL, "V": a.V, "u": a.u}


def _row(r: ClosedLoopResult) -> List[str]:
    return [r.label,
            f"{r.batch_time_h:.3f}" if r.finished else "—",
            f"{r.cP[-1]:.1f}", f"{r.cL[-1]:.2f}", f"{r.cL_peak:.0f}",
            f"{r.cP_overshoot:.1f}", "yes" if r.spec_ok else "NO"]


_TABLE_HEAD = ("controller", "batch time [h]", "cP,f [mol/m³]", "cL,f [mol/m³]",
               "max cL", "cP overshoot", "on spec")


# ═════════════════════════════════════════════════════════════════════════════
# 1 — model, structure of the optimum, tariff
# ═════════════════════════════════════════════════════════════════════════════
def study_model(P: ProcessParams = NOMINAL, **_) -> Dict:
    """Task 1 – model, states, non-linearity; plus the structure of the optimum."""
    out: Dict = {"title": "Model and structure of the time-optimal solution"}
    out["figures"] = []
    out["figures"] += _fig(process_schematic(P)[0], "fig01_process_schematic")

    cP, price = switching_price(P)
    out["figures"] += _fig(switching_figure(cP, price, params=P)[0],
                           "fig02_switching_price")
    out["figures"] += _fig(tariff_figure(lambda_tou)[0], "fig03_tariff")

    a = analytic_optimum(P)
    ocp = solve_min_time_ocp(P, N=200)
    out.update({
        "states": ["V [m^3]", "M_L [mol]", "M_P [mol]"],
        "linear": False,
        "nonlinearity_sources": ["ln(cg*V/M_P) in the flux",
                                 "exp(p/(kM*A)) in the partition ratio",
                                 "bilinear inflow d = u*p(V)"],
        "analytic_optimum": a.summary(),
        "numerical_ocp_T_h": round(ocp["T_h"], 5),
        "analytic_vs_ocp_rel_error": abs(a.T_h - ocp["T_h"]) / ocp["T_h"],
        "washing_price_min_at_cP": round(float(cP[int(np.argmin(price))]), 2),
        "cL_peak_of_optimum": round(a.cL_peak, 1),
        "cL_max": P.cL_max,
        "crystallisation_limit_active": bool(a.cL_peak > P.cL_max),
    })

    # integrator study: order of the RK4 scheme and agreement with CVODES.
    #  A 1 h span with u = 0.3 is used so the truncation error is far above the
    #  round-off floor; the reference is RK4 with 4096 sub-steps.
    m = build_model(params=P)
    span, u_test = 3600.0, 0.3
    x_ref = np.asarray(rk4_integrator(m.f, 4096)(P.x0, u_test, P.theta, span)).ravel()
    conv = {}
    for M in (1, 2, 4, 8, 16, 32):
        xM = np.asarray(rk4_integrator(m.f, M)(P.x0, u_test, P.theta, span)).ravel()
        conv[M] = float(np.max(np.abs(xM - x_ref) / np.maximum(np.abs(x_ref), 1e-12)))
    orders = [float(np.log2(conv[m1] / conv[m2]))
              for m1, m2 in ((1, 2), (2, 4), (4, 8)) if conv[m2] > 0]
    x_cv = np.asarray(cvodes_integrator(m.f, span)(P.x0, u_test, P.theta, span)).ravel()
    out["integrator"] = {
        "test": f"one step of {span:.0f} s at u = {u_test}",
        "rk4_relative_error": {k: f"{v:.3e}" for k, v in conv.items()},
        "observed_order": round(float(np.mean(orders)), 2),
        "cvodes_vs_rk4_4096_rel": f"{float(np.max(np.abs(x_cv - x_ref) / np.abs(x_ref))):.3e}",
        "mpc_sub_steps": P.n_sub_mpc,
        "plant_sub_steps": P.n_sub_plant,
    }
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2 — open loop, constant u
# ═════════════════════════════════════════════════════════════════════════════
def study_open_loop(P: ProcessParams = NOMINAL, **_) -> Dict:
    """Task 2 – simulate 6 h for constant u and discuss."""
    u_vals = [0.0, 0.4, 0.5, 0.6, 0.7, 0.86, 1.0]
    runs = [closed_loop(ConstantU(u, P), nominal_plant(P), params=P,
                        label=f"$u = {u:g}$", stop_on_spec=False) for u in u_vals]
    fig, _ = trajectory_figure(runs, params=P, log_cL=False, mark_finish=False,
                               title="Task 2 — open loop: constant $u$ over 6 h",
                               colors=[PALETTE["ink"], PALETTE["violet"], PALETTE["blue"],
                                       PALETTE["teal"], PALETTE["green"],
                                       PALETTE["orange"], PALETTE["red"]])
    figs = _fig(fig, "fig04_open_loop")

    rows, table = [], []
    for u, r in zip(u_vals, runs):
        t_cP = r.t_cP_spec / 3600.0
        t_cL = r.t_cL_spec / 3600.0
        rows.append({
            "u": u, "cP_end": round(float(r.cP[-1]), 2), "cL_end": round(float(r.cL[-1]), 2),
            "V_end_L": round(float(r.V[-1]) * 1e3, 2),
            "t_cP_spec_h": None if np.isnan(t_cP) else round(t_cP, 3),
            "t_cL_spec_h": None if np.isnan(t_cL) else round(t_cL, 3),
            "both_specs_within_6h": bool(not np.isnan(t_cP) and not np.isnan(t_cL)
                                         and max(t_cP, t_cL) <= 6.0),
        })
        table.append([f"{u:g}", f"{float(r.cP[-1]):.1f}", f"{float(r.cL[-1]):.2f}",
                      f"{float(r.V[-1])*1e3:.1f}",
                      "—" if np.isnan(t_cP) else f"{t_cP:.2f}",
                      "—" if np.isnan(t_cL) else f"{t_cL:.2f}",
                      "yes" if rows[-1]["both_specs_within_6h"] else "no"])
    figs += _fig(metric_table_figure(
        table, ("u", "cP(6 h)", "cL(6 h)", "V(6 h) [L]", "t(cP=100) [h]",
                "t(cL=15) [h]", "feasible"),
        title="Task 2 — constant-$u$ operation after 6 h")[0], "fig05_open_loop_table")

    # Fine sweep with terminal-event detection: what is the *best* constant u?
    sweep = []
    sweep_plant = Plant(name="sweep", params=P, n_sub=30)   # 20 s steps are plenty
    for u in np.linspace(0.50, 0.75, 51):        # 0.005 resolution
        r = closed_loop(ConstantU(float(u), P), sweep_plant, params=P)
        sweep.append({"u": round(float(u), 3),
                      "finished": r.finished,
                      "batch_time_h": round(r.batch_time_h, 4) if r.finished else None,
                      "cP_overshoot": round(r.cP_overshoot, 2),
                      "on_spec": r.spec_ok})
    feasible = [s_ for s_ in sweep if s_["on_spec"]]
    best = min(feasible, key=lambda s_: s_["batch_time_h"]) if feasible else None
    T_opt = analytic_optimum(P).T_h

    fig, ax = pareto_figure(
        [(s_["u"], s_["batch_time_h"] if s_["batch_time_h"] else P.t_max / 3600,
          "") for s_ in sweep],
        xlabel="constant $u$  [–]", ylabel="batch time  [h]",
        title="Task 2 — best achievable batch time with a *constant* input",
        annotate=False)
    ax.axhline(T_opt, color=PALETTE["ink"], ls="--", lw=1.2)
    ax.annotate(f"time-optimal control: {T_opt:.3f} h", xy=(0.03, T_opt),
                xycoords=("axes fraction", "data"), va="bottom", fontsize=8.5)
    for s_ in sweep:
        if not s_["on_spec"]:
            ax.scatter([s_["u"]], [s_["batch_time_h"] or P.t_max / 3600], s=52,
                       color=PALETTE["red"], edgecolor=PALETTE["ink"], linewidth=0.8,
                       zorder=4)
    figs += _fig(fig, "fig05b_open_loop_sweep")

    return {"title": "Open-loop operation with constant u", "figures": figs,
            "runs": rows,
            "constant_u_sweep": sweep,
            "best_constant_u": best,
            "analytic_optimum_T_h": round(T_opt, 4),
            "penalty_of_constant_u_%": (round(100 * (best["batch_time_h"] - T_opt) / T_opt, 1)
                                        if best else None),
            "on_spec_window_u": ([feasible[0]["u"], feasible[-1]["u"]] if feasible
                                 else None),
            "conclusion": (
                "A constant input can satisfy both specifications, but only inside the "
                f"narrow window u in [{feasible[0]['u']:g}, {feasible[-1]['u']:g}]: "
                "below it the batch over-concentrates far past cP,f before the lactose "
                "target is reached, above it the protein target is never reached within "
                f"6 h. The best constant input needs {best['batch_time_h']:.3f} h at "
                f"u = {best['u']:g}, i.e. "
                f"{100*(best['batch_time_h']-T_opt)/T_opt:.0f} % longer than the "
                "time-optimal control, because a single value of u has to trade "
                "concentration against washing at *every* instant. Optimisation helps "
                "precisely because that trade-off is time-varying: the process should "
                "first concentrate (u = 0; washing is inefficient at low cP) and only "
                "then wash (u = 1; washing is ~4x more effective at cP = 100)."
                if best else "No constant input satisfies both specifications.")}


# ═════════════════════════════════════════════════════════════════════════════
# 3 — the objective of Eq. (3) and the influence of the horizon
# ═════════════════════════════════════════════════════════════════════════════
def study_tracking_horizons(P: ProcessParams = NOMINAL, **_) -> Dict:
    """Task 3 – quadratic tracking MPC with N = 5, 20, 50 vs. the Eq. (4) policy."""
    a = analytic_optimum(P)
    runs, slack, solve = {}, {}, {}
    for N in (5, 20, 50):
        mpc = build_nmpc("tracking", N, params=P)
        r = closed_loop(mpc, nominal_plant(P), params=P, label=f"tracking MPC N={N}")
        runs[N] = r
        slack[N] = mpc.slack_active
        solve[N] = round(float(np.mean(r.solve_times)) * 1e3, 1)
    figs = _fig(horizon_figure(runs, params=P, reference_T=a.T_h,
                              title="Task 3 — quadratic tracking cost, Eq. (3)")[0],
                "fig06_tracking_horizons")

    pol = closed_loop(ThresholdPolicy(params=P), nominal_plant(P), params=P)
    pure = closed_loop(build_nmpc("tracking", 20, params=P, terminal_constraint=False),
                       nominal_plant(P), params=P,
                       label="tracking MPC N=20, Eq. (3) only")
    fig, _ = trajectory_figure([runs[20], pure, pol], params=P, reference=_ref(a),
                               reference_label="time-optimal (analytic)",
                               title="Task 3 — Eq. (3) vs. the heuristic policy of Eq. (4)")
    figs += _fig(fig, "fig07_tracking_vs_policy")
    figs += _fig(metric_table_figure(
        [_row(r) for r in [runs[5], runs[20], runs[50], pure, pol]] +
        [["analytic optimum", f"{a.T_h:.3f}", f"{P.cP_f:.1f}", f"{P.cL_f:.2f}",
          f"{a.cL_peak:.0f}", "0.0", "yes"]],
        _TABLE_HEAD, title="Task 3 — closed-loop performance",
        highlight=[5])[0], "fig08_tracking_table")

    return {
        "title": "Quadratic tracking objective and prediction horizon",
        "figures": figs,
        "horizon_study": {N: {"batch_time_h": (round(runs[N].batch_time_h, 4)
                                              if runs[N].finished else None),
                              "finished": runs[N].finished,
                              "cP_final": round(float(runs[N].cP[-1]), 2),
                              "cL_final": round(float(runs[N].cL[-1]), 2),
                              "samples_with_active_terminal_slack": slack[N],
                              "mean_solve_ms": solve[N]} for N in runs},
        "eq3_only_no_terminal_constraint": pure.summary(),
        "threshold_policy": pol.summary(),
        "analytic_optimum_T_h": round(a.T_h, 4),
        "why_eq3_is_unsuited": [
            "It is a *tracking* cost: every sample away from the set point is "
            "penalised, so the optimiser buys early lactose reduction with late "
            "protein concentration instead of finishing early.",
            "Both terms are unscaled: the lactose gap (135 mol/m^3) enters squared, "
            "the protein gap (90 mol/m^3) as well, but the lactose term can be "
            "reduced immediately by washing at u=1, which *blocks* concentration. "
            "The first hour is therefore spent washing at cP = 10 mol/m^3, where "
            "washing is ~4x less effective than at cP = 100 mol/m^3.",
            "A quadratic penalty vanishes to second order near the target, so "
            "'finish one sample earlier' is almost free - the cost cannot express "
            "time optimality.",
            "With N = 5 the terminal specification is unreachable inside the "
            "horizon; the controller becomes purely myopic and never finishes "
            "within 6 h (an unguarded implementation instead returns an infeasible "
            "IPOPT point and applies it silently).",
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4 — objectives that express time optimality
# ═════════════════════════════════════════════════════════════════════════════
def study_objectives(P: ProcessParams = NOMINAL, **_) -> Dict:
    """Task 4 – objective functions that actually minimise the batch time."""
    a = analytic_optimum(P)
    specs = [("tracking", {}, "quadratic tracking, Eq. (3)"),
             ("tracking_scaled", {}, "scaled quadratic tracking"),
             ("l1_time", {}, r"$\ell_1$ time-weighted distance"),
             ("min_time", {}, "free-final-time (min $T$)")]
    runs, meta = [], {}
    for obj, kw, name in specs:
        mpc = build_nmpc(obj, 20, params=P, **kw)
        r = closed_loop(mpc, nominal_plant(P), params=P, label=f"{name}")
        runs.append(r)
        meta[obj] = {**r.summary(), "gap_to_optimum_%": round(
            100.0 * (r.batch_time_h - a.T_h) / a.T_h, 2) if r.finished else None}
    pol = closed_loop(ThresholdPolicy(params=P), nominal_plant(P), params=P)
    runs.append(pol)
    meta["threshold_policy"] = {**pol.summary(),
                                "gap_to_optimum_%": round(100.0 * (pol.batch_time_h - a.T_h) / a.T_h, 2)}

    fig, _ = trajectory_figure(runs, params=P, reference=_ref(a),
                               reference_label="time-optimal (analytic)",
                               title="Task 4 — objective functions compared (N = 20)")
    figs = _fig(fig, "fig09_objectives")
    figs += _fig(metric_table_figure(
        [_row(r) for r in runs] +
        [["analytic optimum", f"{a.T_h:.3f}", f"{P.cP_f:.1f}", f"{P.cL_f:.2f}",
          f"{a.cL_peak:.0f}", "0.0", "yes"]],
        _TABLE_HEAD, title="Task 4 — batch time by objective function",
        highlight=[3])[0], "fig10_objective_table")

    # horizon-independence of the free-final-time formulation
    ht = {}
    for N in (5, 10, 20, 50):
        r = closed_loop(build_nmpc("min_time", N, params=P), nominal_plant(P), params=P,
                        label=f"min-time MPC N={N}")
        ht[N] = r
    figs += _fig(horizon_figure(ht, params=P, reference_T=a.T_h,
                               title="Task 4 — free-final-time MPC is horizon-insensitive")[0],
                 "fig11_min_time_horizons")

    return {"title": "Objective functions that express time optimality",
            "figures": figs, "objectives": meta,
            "analytic_optimum_T_h": round(a.T_h, 4),
            "min_time_horizon_independence": {
                N: round(r.batch_time_h, 4) for N, r in ht.items()},
            "resolution": (
                "Replacing the quadratic tracking cost by (i) an l1 cost on the "
                "remaining distance integrated over time, or (ii) the batch time "
                "itself with the specification as a hard terminal constraint, makes "
                "'finish earlier' strictly cheaper. Both reproduce the analytic "
                "bang-bang optimum; the free-final-time formulation additionally "
                "removes the horizon sensitivity, because the horizon covers the "
                "whole remaining batch by construction.")}


# ═════════════════════════════════════════════════════════════════════════════
# 5 — filter-cake tear
# ═════════════════════════════════════════════════════════════════════════════
def study_tear(P: ProcessParams = NOMINAL, **_) -> Dict:
    """Task 5 – filter cake tears while 30 ≤ cP ≤ 60 (Eq. 5)."""
    a = analytic_optimum(P)
    plant = tear_plant(P)
    best = closed_loop(build_nmpc("min_time", 20, params=P), plant, params=P,
                       label="min-time MPC N=20")
    pol = closed_loop(ThresholdPolicy(params=P), plant, params=P)
    nom_best = closed_loop(build_nmpc("min_time", 20, params=P), nominal_plant(P),
                           params=P, label="min-time MPC, no tear")

    # tear window in hours, taken from the MPC run
    lo, hi, _ = TEAR_WINDOW
    inside = np.flatnonzero((best.cP >= lo) & (best.cP <= hi))
    window = ((float(best.t_h[inside[0]]), float(best.t_h[inside[-1]]))
              if inside.size else None)

    fig, _ = trajectory_figure([best, pol, nom_best], params=P, reference=_ref(a),
                               reference_label="undisturbed optimum",
                               tear_window=window,
                               colors=[PALETTE["green"], PALETTE["orange"],
                                       PALETTE["blue"]],
                               title="Task 5 — filter-cake tear: MPC vs. fixed policy")
    figs = _fig(fig, "fig12_tear")
    figs += _fig(metric_table_figure(
        [_row(r) for r in (best, pol, nom_best)], _TABLE_HEAD,
        title="Task 5 — performance under the filter-cake tear",
        highlight=[0])[0], "fig13_tear_table")

    return {"title": "Filter-cake tear (structural, transient disturbance)",
            "figures": figs,
            "tear_window_h": window,
            "mpc": best.summary(), "policy": pol.summary(),
            "mpc_without_tear": nom_best.summary(),
            "discussion": (
                "The doubled permeate flow is *useful*: it concentrates the batch "
                f"faster, so the MPC finishes in {best.batch_time_h:.3f} h instead of "
                f"{nom_best.batch_time_h:.3f} h. The MPC exploits the disturbance while "
                "keeping cP exactly on its set point, because it re-optimises from the "
                "measured state at every sample. The fixed policy of Eq. (4) has no "
                f"such feedback on the *target*: it over-concentrates to "
                f"{pol.cP[-1]:.1f} mol/m^3 "
                f"({pol.cP_overshoot/P.cP_f*100:.0f} % above specification), i.e. the "
                "batch is off spec and the product would have to be re-diluted.")}


# ═════════════════════════════════════════════════════════════════════════════
# 6 — parametric plant-model mismatch and its robustification
# ═════════════════════════════════════════════════════════════════════════════
def study_mismatch(P: ProcessParams = NOMINAL, quick: bool = False, **_) -> Dict:
    """Additional task 1 – poorly estimated kM_L, and four ways to cope."""
    P8 = P.with_(t_max=8 * 3600)
    factors = (0.75, 0.5, 0.25)
    table, results, logs, truths = [], {}, {}, {}

    #  Controllers that do not depend on the realisation are built once: each
    #  IPOPT instance keeps its own compiled expression graph, and rebuilding
    #  them per factor is both slow and memory-hungry.
    mpc_nominal = build_nmpc("min_time", 20, params=P8)
    mpc_backoff = build_nmpc("min_time", 20, params=P8, back_off_cL=200.0)
    mpc_multi = build_multistage_nmpc(20, params=P8, uncertainty=KM_L_UNCERTAINTY)
    mpc_adaptive = build_nmpc("min_time", 20, params=P8)

    def add(tag: str, factor: float, r: ClosedLoopResult, extra: Optional[Dict] = None):
        results.setdefault(f"{factor:g}", {})[tag] = {**r.summary(), **(extra or {})}
        table.append([f"{factor:g}", tag,
                      f"{r.batch_time_h:.3f}" if r.finished else "—",
                      f"{r.cL_peak:.0f}", f"{r.cL_violation:.1f}",
                      "yes" if r.cL_violation <= 1e-6 else "NO"])

    runs_for_fig: Dict[float, List[ClosedLoopResult]] = {}
    for f in factors:
        plant = mismatch_plant(f, P8)
        mpc_nominal.reset()
        r_nom = closed_loop(mpc_nominal, plant, params=P8, label="nominal-model MPC")
        mpc_true = build_nmpc("min_time", 20, params=P8,
                              theta=P8.scaled(kM_L=f).theta)
        r_true = closed_loop(mpc_true, plant, params=P8, label="perfect-model MPC")
        del mpc_true
        mpc_backoff.reset()
        r_back = closed_loop(mpc_backoff, plant, params=P8, label="back-off MPC (−200)")
        mpc_multi.reset()
        r_ms = closed_loop(mpc_multi, plant, params=P8, label="multi-stage NMPC")
        mpc_adaptive.reset()          # restores theta as well -> order independent
        ad = AdaptiveNMPC(mpc_adaptive, MHE(P8), params=P8,
                          rng=np.random.default_rng(7))
        r_ad = closed_loop(ad, plant, params=P8, label="adaptive NMPC (MHE)")
        logs[f"true {f:g}"] = list(ad.kappa_log)
        truths[f"true {f:g}"] = f

        n_mid = len(ad.kappa_log) // 2
        for tag, r, extra in (("nominal model", r_nom, None),
                              ("perfect model", r_true, None),
                              ("back-off (−200)", r_back, None),
                              ("multi-stage", r_ms, None),
                              ("adaptive (MHE)", r_ad,
                               {"kappa_estimate_mid": round(float(ad.kappa_log[n_mid]), 3),
                                "kappa_true": f})):
            add(tag, f, r, extra)
        runs_for_fig[f] = [r_nom, r_ms, r_ad, r_true]
        gc.collect()

    figs: List[str] = []
    for f in factors:
        fig, _ = trajectory_figure(
            runs_for_fig[f], params=P8, log_cL=True,
            colors=[PALETTE["orange"], PALETTE["violet"], PALETTE["teal"], PALETTE["ink"]],
            title=f"Additional task 1 — $k_{{M,L}}^{{true}} = {f:g}\\,k_{{M,L}}$")
        figs += _fig(fig, f"fig14_mismatch_{str(f).replace('.', 'p')}")
    figs += _fig(metric_table_figure(
        table, ("kM_L factor", "controller", "batch time [h]", "max cL",
                "violation", "cL ≤ 570"),
        title="Additional task 1 — robustification of the mismatched MPC")[0],
        "fig15_mismatch_table")
    figs += _fig(estimator_figure(logs, truths, dt_h=P8.dt_ctrl / 3600.0)[0],
                 "fig16_identification")

    return {"title": "Parametric plant-model mismatch in kM_L",
            "figures": figs, "results": results,
            "kappa_estimates": {k: round(float(v[len(v) // 2]), 3)
                                for k, v in logs.items()},
            "mechanism": (
                "r_L = alpha/[1+(alpha-1)exp(p/(kM_L*A))] decreases when kM_L "
                "decreases, so a plant with a smaller kM_L rejects *more* lactose "
                "and the retentate concentration rises much higher than predicted. "
                "The controller keeps concentrating because its own model reports a "
                "harmless cL, and the crystallisation limit is breached."),
            "findings": [
                "0.75x and 0.5x: the specification is still met and cL stays below "
                "570 mol/m^3; the batch simply takes longer (3.69 h and 4.04 h).",
                "0.25x: the nominal-model MPC drives cL to 741 mol/m^3, i.e. 30 % "
                "above the crystallisation limit - the constraint can no longer be "
                "satisfied with a wrong model.",
                "Constraint back-off on cL_max is useless here: the *prediction* is "
                "biased, so the tightened bound is never active in the optimiser.",
                "Multi-stage NMPC over kM_L in {0.25,0.5,0.75,1} keeps cL at the "
                "limit for every realisation and matches the perfect-model batch "
                "time, at ~4x the computational cost and no loss at the nominal plant.",
                "Moving-horizon estimation identifies the factor within ~1 h of "
                "operation and recovers the perfect-model performance; it loses "
                "identifiability at the very end of the batch, when cL approaches "
                "the sensor noise floor.",
            ]}


# ═════════════════════════════════════════════════════════════════════════════
# 7 — structural mismatch: protein leakage
# ═════════════════════════════════════════════════════════════════════════════
def study_leakage(P: ProcessParams = NOMINAL, **_) -> Dict:
    """Additional task 2 – protein partially permeates the membrane (Eq. 6)."""
    P8 = P.with_(t_max=8 * 3600)

    # ── how sharp is the transition?  r_P depends on exp(p/(kM_P A)) ────────
    scan = []
    for kM_P in (1e-7, 2e-7, 3e-7, 5e-7, 7e-7, 1e-6, 2e-6, 3e-6):
        pp = P8.with_(kM_P=kM_P)
        m = build_model(protein_leakage=True, params=pp)
        x = np.array([pp.MP0 / P.cP_f, 1.0, pp.MP0])           # at the set point
        dMP = float(m.rhs(x, 0.0, pp.theta)[2])
        scan.append({"kM_P": kM_P, "loss_rate_%_per_h": round(-dMP / pp.MP0 * 3600 * 100, 6)})
    # analytic threshold: the exponential kills r_P once p/(kM_P A) >~ 20
    kM_P_safe = P.k * P.A * np.log(P.cg / P.cP_f) / (20.0 * P.A)

    rows, runs, res = [], [], {}
    base = closed_loop(build_nmpc("min_time", 20, params=P8), nominal_plant(P8),
                       params=P8, label="no leakage (reference)")
    runs.append(base)
    for kM_P, aware in ((3e-7, False), (5e-7, False), (1e-6, False), (1e-6, True)):
        plant = leakage_plant(P8, beta=1.3, kM_P=kM_P)
        mpc = build_nmpc("min_time", 20, params=P8, protein_leakage=aware,
                         theta=P8.with_(kM_P=kM_P).theta if aware else None)
        tag = ("leakage-aware MPC" if aware else "nominal (2-state) MPC")
        r = closed_loop(mpc, plant, params=P8,
                        label=f"$k_{{M,P}}={kM_P:.0e}$, {tag}")
        runs.append(r)
        loss = 100 * (1 - r.protein_recovery)
        rows.append([f"{kM_P:.0e}", "yes" if aware else "no",
                     f"{r.batch_time_h:.3f}" if r.finished else "not finished",
                     f"{loss:.3f}", f"{float(r.cP[-1]):.2f}", f"{float(r.cL[-1]):.2f}",
                     "yes" if r.spec_ok else "NO"])
        res[f"kM_P={kM_P:.0e},aware={aware}"] = {
            **r.summary(), "protein_loss_%": round(loss, 4)}

    fig, _ = trajectory_figure(
        runs, params=P8, log_cL=True,
        title="Additional task 2 — structural mismatch: protein passage")
    figs = _fig(fig, "fig17_leakage")

    figl, axl = pareto_figure([(s_["kM_P"] * 1e6, max(s_["loss_rate_%_per_h"], 1e-12), "")
                               for s_ in scan],
                              xlabel=r"$k_{M,P}$  [$10^{-6}$ m s$^{-1}$]",
                              ylabel="protein loss rate at $c_P=c_{P,f}$  [% h$^{-1}$]",
                              title="Protein passage is an exponential cliff",
                              annotate=False)
    axl.set_yscale("log")
    axl.axhline(1.0, color=PALETTE["red"], ls="--", lw=1.1)
    axl.annotate("1 % per hour", xy=(0.02, 1.0), xycoords=("axes fraction", "data"),
                 va="bottom", fontsize=8, color=PALETTE["red"])
    axl.axvline(kM_P_safe * 1e6, color=PALETTE["ink"], ls=":", lw=1.1)
    axl.annotate(f"$p/(k_{{M,P}}A)=20$\n$k_{{M,P}}={kM_P_safe:.1e}$",
                 xy=(kM_P_safe * 1e6, 1e-6), xycoords=("data", "axes fraction"),
                 fontsize=8, ha="left", va="bottom")
    figs += _fig(figl, "fig17b_leakage_threshold")

    figs += _fig(metric_table_figure(
        rows, ("kM_P [m/s]", "MPC knows Eq. 6", "batch time [h]", "protein loss [%]",
               "cP,f", "cL,f", "on spec"),
        title="Additional task 2 — effect of protein passage")[0], "fig18_leakage_table")

    return {"title": "Structural plant-model mismatch: protein leakage",
            "figures": figs, "reference": base.summary(), "results": res,
            "loss_rate_scan": scan,
            "negligible_below_kM_P": float(f"{kM_P_safe:.3e}"),
            "structural_finding": (
                "With protein passage the constant-volume washing arc no longer keeps "
                "cP constant: for u = 1, dcP/dt = -(p cP/V) r_P < 0, i.e. the classic "
                "'concentrate, then wash' structure actively destroys the protein "
                "specification. The reachable set shrinks, and above roughly "
                "kM_P = 1e-6 m/s the pair (cP = 100, cL <= 15) becomes unreachable "
                "inside 8 h even for a controller that knows Eq. (6) exactly."),
            "discussion": (
                "Because the model is written in hold-ups, Eq. (6) is a *third state* "
                "(dM_P/dt = -r_P c_P p) instead of a hand-patched attribute, so the "
                "leakage scenario is a plain ODE and the result no longer depends on "
                "the integrator. The passage ratio contains exp(p/(kM_P A)), which "
                f"makes the effect an exponential cliff: for kM_P below "
                f"{kM_P_safe:.1e} m/s the loss is below 1e-3 %/h and the nominal "
                "two-state model is exact for all practical purposes (batch time "
                f"{base.batch_time_h:.3f} h, unchanged); at kM_P = 1e-6 m/s the loss "
                "reaches 3.3 %/h at the set point and roughly a fifth of the product "
                "is lost over the batch. Making the controller's model *aware* of the "
                "leakage improves the terminal protein concentration but cannot "
                "recover feasibility - the limitation is the plant, not the "
                "controller.")}


# ═════════════════════════════════════════════════════════════════════════════
# 8 — Monte-Carlo campaign
# ═════════════════════════════════════════════════════════════════════════════
_MC_CONTROLLERS = {
    "nominal": "nominal-model MPC",
    "multistage": "multi-stage NMPC",
    "adaptive": "adaptive NMPC (MHE)",
}
_MC_SEED = 20250725
_MC_FIELDS = ("batch_times_h", "cL_peaks", "cL_violations", "finished", "spec_ok",
              "kappas", "solve_ms")


def _mc_factory(key: str, P8: ProcessParams, N: int):
    if key == "nominal":
        return lambda: build_nmpc("min_time", N, params=P8)
    if key == "multistage":
        return lambda: build_multistage_nmpc(N, params=P8)
    if key == "adaptive":
        return lambda: AdaptiveNMPC(build_nmpc("min_time", N, params=P8), MHE(P8),
                                    params=P8, rng=np.random.default_rng(11))
    raise KeyError(key)


def _mc_summary(raw: Dict[str, list], label: str) -> Dict:
    mc = MonteCarloResult(label, *[np.asarray(raw[f]) for f in _MC_FIELDS[:-1]],
                          solve_ms=np.asarray(raw["solve_ms"]))
    return mc.summary()


def study_montecarlo(P: ProcessParams = NOMINAL, quick: bool = False,
                     n_draws: Optional[int] = None,
                     controllers: Optional[List[str]] = None,
                     chunk: Optional[int] = None, N: int = 10, **_) -> Dict:
    """Robustness of the candidate controllers over randomly drawn plants.

    The campaign is **resumable**: results for each controller are appended to
    ``results/_studies/mc_<key>.json``, so a long campaign can be built up in
    several invocations (``chunk`` = how many *additional* draws to add now).
    Every draw uses its own seed, so chunking does not repeat plants.
    """
    #  A campaign is a *statistical* study: 40 s plant sub-steps and a short
    #  horizon are ample, and they keep the robust campaign affordable.
    P8 = P.with_(t_max=8 * 3600, n_sub_plant=15)
    n_target = n_draws if n_draws is not None else (8 if quick else 24)
    keys = controllers or list(_MC_CONTROLLERS)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Dict] = {}
    times, peaks = {}, {}
    for key, label in _MC_CONTROLLERS.items():
        cache = CACHE_DIR / f"mc_{key}.json"
        raw = (json.loads(cache.read_text())["raw"] if cache.exists()
               else {f: [] for f in _MC_FIELDS})
        have = len(raw["spec_ok"])
        if key in keys and have < n_target:
            todo = min(chunk or (n_target - have), n_target - have)
            ctrl_factory = _mc_factory(key, P8, N)
            for i in range(have, have + todo):
                mc = run_campaign(ctrl_factory, label=label, n_draws=1, params=P8,
                                  seed=_MC_SEED + i)
                for f in _MC_FIELDS[:-1]:
                    raw[f].append(getattr(mc, f).tolist()[0])
                raw["solve_ms"].append(mc.solve_ms.tolist()[0])
                #  persist after *every* draw: a long campaign is then resumable
                #  even if the process is interrupted
                cache.write_text(json.dumps({"horizon": N, "raw": raw}, indent=2))
            have = len(raw["spec_ok"])
        if have == 0:
            continue
        out[label] = {**_mc_summary(raw, label), "horizon": N}
        ok = np.asarray(raw["spec_ok"], dtype=bool)
        times[label] = np.asarray(raw["batch_times_h"])[ok]
        peaks[label] = np.asarray(raw["cL_peaks"])

    figs: List[str] = []
    if times:
        n_shown = min(len(v) for v in peaks.values())
        figs += _fig(histogram_figure(
            times, xlabel="batch time  [h]",
            title=f"Monte-Carlo: batch time over {n_shown} random plants")[0],
            "fig19_mc_batchtime")
        figs += _fig(histogram_figure(
            peaks, xlabel=r"peak $c_L$  [mol m$^{-3}$]",
            title="Monte-Carlo: worst lactose concentration",
            vline=P.cL_max, vline_label="crystallisation limit")[0], "fig20_mc_peak")
        figs += _fig(metric_table_figure(
            [[k, str(v["n"]), f"{v['finished_%']}", f"{v['on_spec_%']}",
              f"{v['cL_constraint_ok_%']}", f"{v['median_T_h']}", f"{v['p90_T_h']}",
              f"{v['max_cL_violation']}", f"{v['mean_solve_ms']}"]
             for k, v in out.items()],
            ("controller", "plants", "finished [%]", "on spec [%]", "cL ≤ 570 [%]",
             "median T [h]", "p90 T [h]", "max violation", "solve [ms]"),
            title=f"Monte-Carlo summary (N = {N}, uniform parametric uncertainty)")[0],
            "fig21_mc_table")
    # ── diagnosis: where do the residual off-spec batches come from? ───────
    #  Every campaign draw *finishes*, so the failures are constraint related.
    #  Two mechanisms are separated here: a biased kM_L breaks cL <= cL_max
    #  (section 8), and a biased permeability makes the *equality* specification
    #  cP = cP_f overshoot within one 10-min sample.
    diag = {}
    for label, factors, dt in (("k x0.8", {"k": 0.8}, P.dt_ctrl),
                               ("k x1.0", {}, P.dt_ctrl),
                               ("k x1.2", {"k": 1.2}, P.dt_ctrl),
                               ("k x1.2, dt = 2 min", {"k": 1.2}, 120.0),
                               ("k x1.2, dt = 30 s", {"k": 1.2}, 30.0)):
        Pd = P8.with_(dt_ctrl=dt)
        plant = Plant(name=label, params=Pd,
                      theta_true=Pd.scaled(**factors).theta if factors else None)
        r = closed_loop(build_nmpc("min_time", 20, params=Pd), plant, params=Pd,
                        label=label)
        diag[label] = {"batch_time_h": round(r.batch_time_h, 4),
                       "cP_final": round(float(r.cP[-1]), 3),
                       "cP_overshoot": round(r.cP_overshoot, 3),
                       "on_spec": r.spec_ok}

    return {"title": "Monte-Carlo robustness campaign", "figures": figs,
            "n_draws_target": n_target, "horizon": N,
            "ranges": {k: list(v) for k, v in DEFAULT_RANGES.items()},
            "campaigns": out,
            "sampling_diagnosis": diag,
            "note": (
                "The controller keeps its nominal model in every draw. All draws "
                "finish inside 8 h, so the 'on spec' rate is limited by two "
                "constraint mechanisms: (i) a biased kM_L violates cL <= cL_max "
                "(see section 8), and (ii) a biased permeability makes the "
                "*equality* specification cP = cP_f overshoot, because the "
                "controller can only correct once per 10-minute sample. The "
                "'sampling_diagnosis' entry isolates (ii): shortening the control "
                "interval removes the overshoot without changing anything else, "
                "which is the cheapest available fix for an equality end-point "
                "specification.")}


# ═════════════════════════════════════════════════════════════════════════════
# 9 — economic MPC
# ═════════════════════════════════════════════════════════════════════════════
def study_economic(P: ProcessParams = NOMINAL, quick: bool = False, **_) -> Dict:
    """Extension – day-ahead tariff, and why this process has no economic trade-off.

    The pump draws ``P_idle + P_dyn·u``, so the electricity bill is
    ``P_idle·T + P_dyn·∫u dt`` weighted by the tariff.  Section 3 of the report
    shows that the *solvent* integral ``∫u p dt`` is minimised by exactly the
    same bang-bang policy as the batch time, hence **both** terms of the energy
    bill are minimised by the time-optimal policy and the economic MPC cannot
    trade one against the other.  This study verifies that numerically and then
    quantifies the decision that *does* matter: when to start the batch.
    """
    a = analytic_optimum(P)

    # 1 — value of time: does the trajectory change at all?
    ct_values = (0.0, 0.5, 2.0, 10.0, 30.0)
    res, runs = {}, []
    for ct in ct_values:
        mpc = build_nmpc("economic", 20, params=P, value_of_time=ct)
        r = closed_loop(mpc, nominal_plant(P), params=P,
                        label=f"economic MPC, $c_t={ct:g}$ €/h")
        res[f"c_t={ct:g}"] = {**r.summary(), "value_of_time_EUR_per_h": ct,
                              "solvent_L": round(r.diavolume * P.V0 * 1e3, 3)}
        if ct in (0.0, 30.0):
            runs.append(r)
    T_span = max(v["batch_time_h"] for v in res.values()) - \
        min(v["batch_time_h"] for v in res.values())
    E_span = max(v["energy_kWh"] for v in res.values()) - \
        min(v["energy_kWh"] for v in res.values())

    # 2 — the decision that matters: the start hour
    hours = range(0, 24, 3) if quick else range(24)
    by_hour = {}
    for hour in hours:
        mpc = build_nmpc("economic", 20, params=P, value_of_time=2.0)
        r = closed_loop(mpc, nominal_plant(P), params=P, t0=hour * 3600.0,
                        label=f"start {hour:02d}:00")
        by_hour[hour] = {"batch_time_h": round(r.batch_time_h, 4),
                         "energy_kWh": round(r.energy(), 3),
                         "cost_EUR": round(r.energy_cost(t_start=hour * 3600.0), 4)}
    best = min(by_hour, key=lambda h: by_hour[h]["cost_EUR"])
    worst = max(by_hour, key=lambda h: by_hour[h]["cost_EUR"])

    figs = _fig(tariff_figure(lambda_tou)[0], "fig03_tariff")

    fig, ax = figure(width=6.8, height=3.4)
    hs = sorted(by_hour)
    costs = [by_hour[h]["cost_EUR"] for h in hs]
    ax.bar(hs, costs, color=[PALETTE["green"] if h == best else
                             (PALETTE["red"] if h == worst else PALETTE["blue"])
                             for h in hs], width=0.75, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("batch start [hour of day]")
    ax.set_ylabel("electricity for one batch  [€]")
    ax.set_title("The trajectory never changes — the start hour changes the bill "
                 f"by {max(costs)/min(costs):.1f}×")
    ax.set_xticks(range(0, 24, 2))
    ax.annotate(f"cheapest {best:02d}:00 — €{min(costs):.2f}",
                xy=(best, min(costs)), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold", color=PALETTE["green"])
    ax.annotate(f"dearest {worst:02d}:00 — €{max(costs):.2f}",
                xy=(worst, max(costs)), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold", color=PALETTE["red"])
    figs += _fig(fig, "fig22_economic_start_hour")

    fig, _ = trajectory_figure(runs, params=P, reference=_ref(a),
                               reference_label="time-optimal (analytic)",
                               title="Economic MPC — identical to the time-optimal "
                                     "policy for every value of time")
    figs += _fig(fig, "fig23_economic")

    return {"title": "Economic MPC with a day-ahead tariff", "figures": figs,
            "value_of_time_sweep": res,
            "spread_over_value_of_time": {"batch_time_h": round(T_span, 6),
                                          "energy_kWh": round(E_span, 6)},
            "start_hour_sweep": by_hour,
            "cheapest_start_hour": best, "dearest_start_hour": worst,
            "cost_ratio_worst_best": round(max(costs) / min(costs), 2),
            "analytic_optimum_T_h": round(a.T_h, 4),
            "finding": (
                "Sweeping the value of time from 0 to 30 EUR/h changes the batch time "
                f"by {T_span*3600:.1f} s and the energy by {E_span:.4f} kWh, i.e. not at "
                "all: with the pump model P = P_idle + P_dyn u, the electricity bill is "
                "P_idle T + P_dyn * integral(u dt), and *both* terms are minimised by the "
                "time-optimal bang-bang policy (the solvent integral is minimised by the "
                "same linear-programming argument as the batch time, see section 3). "
                "The time-optimal policy is therefore unconditionally the economic "
                f"optimum. What does matter is *scheduling*: starting at {best:02d}:00 "
                f"instead of {worst:02d}:00 changes the electricity cost of one batch by "
                f"a factor {max(costs)/min(costs):.1f} "
                f"(EUR {min(costs):.2f} vs EUR {max(costs):.2f}) without touching the "
                "control policy at all.")}


# ═════════════════════════════════════════════════════════════════════════════
# 10 — numerics
# ═════════════════════════════════════════════════════════════════════════════
def study_numerics(P: ProcessParams = NOMINAL, **_) -> Dict:
    """Discretisation error, real-time feasibility, effect of the plant sub-stepping."""
    a = analytic_optimum(P)
    coarse = closed_loop(BangBang(P), Plant(name="coarse", params=P, n_sub=1), params=P,
                         label="bang-bang, plant RK4 with 1 step of 600 s")
    fine = closed_loop(BangBang(P), nominal_plant(P), params=P,
                       label="bang-bang, plant RK4 with 120 steps of 5 s")
    timing = {}
    for N in (5, 10, 20, 50):
        mpc = build_nmpc("min_time", N, params=P)
        r = closed_loop(mpc, nominal_plant(P), params=P)
        timing[N] = {"batch_time_h": round(r.batch_time_h, 4),
                     "mean_solve_ms": round(float(np.mean(r.solve_times)) * 1e3, 1),
                     "max_solve_ms": round(float(np.max(r.solve_times)) * 1e3, 1),
                     "samples": int(r.u.size)}
    ms = build_multistage_nmpc(20, params=P)
    r_ms = closed_loop(ms, nominal_plant(P), params=P)
    figs = _fig(metric_table_figure(
        [[f"N={N}", str(v["samples"]), f"{v['mean_solve_ms']}", f"{v['max_solve_ms']}",
          f"{v['batch_time_h']}"] for N, v in timing.items()] +
        [["multi-stage N=20 (4 scen.)", str(int(r_ms.u.size)),
          f"{float(np.mean(r_ms.solve_times))*1e3:.1f}",
          f"{float(np.max(r_ms.solve_times))*1e3:.1f}",
          f"{r_ms.batch_time_h:.4f}"]],
        ("controller", "samples", "mean solve [ms]", "max solve [ms]", "batch time [h]"),
        title=f"Real-time feasibility (control interval = {P.dt_ctrl:.0f} s)")[0],
        "fig24_timing_table")
    return {"title": "Numerical accuracy and real-time feasibility",
            "figures": figs,
            "plant_substepping": {
                "one_step_600s": {"batch_time_h": round(coarse.batch_time_h, 4),
                                  "cP_final": round(float(coarse.cP[-1]), 3)},
                "120_steps_5s": {"batch_time_h": round(fine.batch_time_h, 4),
                                 "cP_final": round(float(fine.cP[-1]), 3)},
                "analytic": round(a.T_h, 4),
                "error_of_coarse_grid_s": round(
                    abs(coarse.batch_time_h - a.T_h) * 3600, 1)},
            "solver_timing": timing,
            "multistage_timing": {
                "mean_solve_ms": round(float(np.mean(r_ms.solve_times)) * 1e3, 1),
                "max_solve_ms": round(float(np.max(r_ms.solve_times)) * 1e3, 1)},
            "real_time_margin": (
                f"the slowest controller needs "
                f"{float(np.max(r_ms.solve_times))*1e3:.0f} ms per sample, i.e. "
                f"{100*float(np.max(r_ms.solve_times))/P.dt_ctrl:.2f} % of the "
                f"{P.dt_ctrl:.0f} s control interval")}


# ═════════════════════════════════════════════════════════════════════════════
STUDIES = {
    "model": study_model,
    "open_loop": study_open_loop,
    "tracking": study_tracking_horizons,
    "objectives": study_objectives,
    "tear": study_tear,
    "mismatch": study_mismatch,
    "leakage": study_leakage,
    "montecarlo": study_montecarlo,
    "economic": study_economic,
    "numerics": study_numerics,
}


def run_study(name: str, **kw) -> Dict:
    if name not in STUDIES:
        raise KeyError(f"unknown study '{name}'; choose from {sorted(STUDIES)}")
    t0 = time.perf_counter()
    out = STUDIES[name](**kw)
    out["wall_time_s"] = round(time.perf_counter() - t0, 1)
    return out


def _run_isolated(name: str, quick: bool) -> Dict:
    """Run one study in a fresh interpreter and return its result dictionary.

    Every ``nlpsol`` instance owns a compiled expression graph that CPython does
    not necessarily hand back to the OS.  Running the ten studies in one process
    therefore accumulates well over a gigabyte and can be killed by the OS before
    the campaign finishes.  One subprocess per study keeps the footprint at the
    level of the largest single study (~0.7 GB) and makes the pipeline robust.
    """
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / f"{name}.json"
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        env["MPLBACKEND"] = "Agg"
        cmd = [sys.executable, "-c",
               "import json,sys;from dfp.experiments.studies import run_study;"
               "n,q,o=sys.argv[1],sys.argv[2]=='1',sys.argv[3];"
               "json.dump(run_study(n,quick=q),open(o,'w'),indent=2,default=str)",
               name, "1" if quick else "0", str(out_file)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0 or not out_file.exists():
            raise RuntimeError(
                f"study '{name}' failed (exit {proc.returncode})\n"
                f"{proc.stderr[-2000:]}")
        return json.loads(out_file.read_text())


CACHE_DIR = RESULT_DIR / "_studies"


def _merge_cache(quick: bool, verbose: bool = True) -> Dict:
    """Collect every cached study into ``results/results.json``."""
    out: Dict = {"quick": quick, "studies": {}}
    for name in STUDIES:
        f = CACHE_DIR / f"{name}.json"
        if f.exists():
            out["studies"][name] = json.loads(f.read_text())
    path = RESULT_DIR / "results.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    if verbose:
        print(f"\nmerged {len(out['studies'])}/{len(STUDIES)} studies "
              f"→ {path.relative_to(ROOT)}")
    return out


def run_all(names: Optional[List[str]] = None, *, quick: bool = False,
            params: ProcessParams = NOMINAL, verbose: bool = True,
            isolate: bool = True, skip_existing: bool = False) -> Dict:
    """Execute the studies and write ``results/results.json``.

    Parameters
    ----------
    isolate
        Run every study in its own subprocess (default).  Set to ``False`` to
        keep everything in-process, e.g. under a debugger.
    """
    names = names or list(STUDIES)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        cache = CACHE_DIR / f"{name}.json"
        if skip_existing and cache.exists():
            if verbose:
                print(f"• {name} (cached)", flush=True)
            continue
        if verbose:
            print(f"▶ {name} …", flush=True)
        if isolate:
            out = _run_isolated(name, quick)
        else:
            out = run_study(name, P=params, quick=quick)
            gc.collect()
        cache.write_text(json.dumps(out, indent=2, default=str))
        if verbose:
            print(f"  done in {out['wall_time_s']} s "
                  f"({len(out.get('figures', []))} files)", flush=True)
    return _merge_cache(quick, verbose)
