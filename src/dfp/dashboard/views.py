"""Dashboard pages – all figures come from :mod:`dfp.viz`."""

from __future__ import annotations

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")

import numpy as np
import streamlit as st

from ..config import KM_L_UNCERTAINTY, NOMINAL, ProcessParams
from ..controllers import (BangBang, ConstantU, ThresholdPolicy, analytic_optimum,
                           build_multistage_nmpc, build_nmpc, switching_price)
from ..estimation import MHE, AdaptiveNMPC
from ..experiments.montecarlo import run_campaign
from ..plant import leakage_plant, mismatch_plant, nominal_plant, tear_plant
from ..simulate import closed_loop
from ..tariff import lambda_tou
from ..viz import (PALETTE, estimator_figure, histogram_figure, pareto_figure,
                   process_schematic, switching_figure, tariff_figure,
                   trajectory_figure)

P = NOMINAL

_CSS = f"""
<style>
  .block-container {{padding-top: 1.6rem; max-width: 1500px;}}
  h1, h2, h3 {{color: {PALETTE['ink']}; letter-spacing: -0.01em;}}
  .dfp-hero {{
      background: linear-gradient(100deg, {PALETTE['ink']} 0%, #2b3138 58%,
                                  {PALETTE['green']} 175%);
      color: #fff; border-radius: 16px; padding: 1.1rem 1.4rem; margin-bottom: 1.1rem;}}
  .dfp-hero h1 {{color: #fff; margin: 0 0 .25rem 0; font-size: 1.55rem;}}
  .dfp-hero p  {{margin: 0; opacity: .82; font-size: .92rem;}}
  .dfp-cards {{display: flex; gap: .7rem; flex-wrap: wrap; margin: .2rem 0 1rem 0;}}
  .dfp-card {{
      flex: 1 1 165px; background: #fff; border: 1px solid #E4E8ED; border-radius: 13px;
      padding: .7rem .9rem; border-left: 4px solid {PALETTE['green']};}}
  .dfp-card .k {{font-size: .72rem; text-transform: uppercase; letter-spacing: .07em;
                 color: #7A828C;}}
  .dfp-card .v {{font-size: 1.32rem; font-weight: 700; color: {PALETTE['ink']};
                 line-height: 1.25;}}
  .dfp-card .s {{font-size: .74rem; color: #7A828C;}}
  .dfp-card.warn {{border-left-color: {PALETTE['red']};}}
  .dfp-card.ok   {{border-left-color: {PALETTE['green']};}}
  .dfp-card.info {{border-left-color: {PALETTE['blue']};}}
  .dfp-note {{background: #F7F9FB; border-left: 3px solid {PALETTE['blue']};
              border-radius: 8px; padding: .65rem .9rem; font-size: .88rem;
              color: #3d454e;}}
  section[data-testid="stSidebar"] {{background: #FAFBFC; border-right: 1px solid #E4E8ED;}}
  div[data-testid="stMetricValue"] {{font-size: 1.35rem;}}
  .stTabs [data-baseweb="tab"] {{font-weight: 600;}}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_header(pages: List[str]) -> str:
    with st.sidebar:
        st.markdown(
            f"<div style='font-weight:800;font-size:1.05rem;color:{PALETTE['ink']}'>"
            "🧪 Diafiltration NMPC</div>"
            "<div style='font-size:.78rem;color:#7A828C;margin-bottom:.9rem'>"
            "Advanced Process Control · SS25 · TU Dortmund</div>",
            unsafe_allow_html=True)
        choice = st.radio("Study", pages, label_visibility="collapsed")
        a = analytic_optimum(P)
        st.markdown("---")
        st.markdown(
            f"<div class='dfp-note'><b>Reference</b><br>time-optimal batch: "
            f"<b>{a.T_h:.4f} h</b><br>switch at $c_P=100$ after "
            f"{a.t_switch/3600:.3f} h</div>", unsafe_allow_html=True)
        st.caption("Every controller on every page is scored against this optimum.")
    return choice


def _hero(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='dfp-hero'><h1>{title}</h1><p>{subtitle}</p></div>",
                unsafe_allow_html=True)


def _cards(items: List[Dict[str, str]]) -> None:
    html = "<div class='dfp-cards'>"
    for it in items:
        html += (f"<div class='dfp-card {it.get('tone','')}'>"
                 f"<div class='k'>{it['k']}</div><div class='v'>{it['v']}</div>"
                 f"<div class='s'>{it.get('s','')}</div></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def _score_cards(res, a) -> None:
    gap = 100 * (res.batch_time_h - a.T_h) / a.T_h if res.finished else None
    _cards([
        {"k": "batch time", "v": f"{res.batch_time_h:.3f} h" if res.finished
         else "not finished", "s": f"optimum {a.T_h:.3f} h",
         "tone": "ok" if res.finished else "warn"},
        {"k": "gap to optimum", "v": "—" if gap is None else f"{gap:+.1f} %",
         "s": "0 % = time-optimal", "tone": "info"},
        {"k": "final cP", "v": f"{float(res.cP[-1]):.1f}",
         "s": f"target {P.cP_f:g} mol/m³",
         "tone": "ok" if res.cP_overshoot <= 1.0 else "warn"},
        {"k": "peak cL", "v": f"{res.cL_peak:.0f}",
         "s": f"limit {P.cL_max:g} mol/m³",
         "tone": "warn" if res.cL_violation > 0 else "ok"},
        {"k": "on spec", "v": "yes" if res.spec_ok else "NO",
         "s": "cP = target and cL ≤ limit",
         "tone": "ok" if res.spec_ok else "warn"},
    ])


@st.cache_resource(show_spinner=False)
def _nmpc(objective: str, N: int, t_max: float, **kw):
    return build_nmpc(objective, N, params=P.with_(t_max=t_max), **kw)


@st.cache_resource(show_spinner=False)
def _multistage(N: int, t_max: float, worst_case: bool = True):
    return build_multistage_nmpc(N, params=P.with_(t_max=t_max), worst_case=worst_case)


# ═════════════════════════════════════════════════════════════════════════════
def page_overview() -> None:
    _hero("Time-optimal control of a batch diafiltration process",
          "Model · analytic optimum · five MPC objectives · robust and adaptive NMPC")
    a = analytic_optimum(P)
    _cards([
        {"k": "optimal batch time", "v": f"{a.T_h:.4f} h",
         "s": "analytic, verified by a 200-interval OCP"},
        {"k": "optimal structure", "v": "u = 0 → u = 1",
         "s": f"switch at t = {a.t_switch/3600:.3f} h"},
        {"k": "states", "v": "3", "s": "V, M_L, M_P (hold-ups)"},
        {"k": "model", "v": "non-linear", "s": "log flux · exp partition · bilinear u·p"},
        {"k": "peak lactose", "v": f"{a.cL_peak:.0f}",
         "s": f"crystallisation limit {P.cL_max:g}", "tone": "ok"},
    ])
    left, right = st.columns([1.25, 1.0], gap="large")
    with left:
        st.markdown("#### Process")
        st.pyplot(process_schematic(P)[0], use_container_width=True)
    with right:
        st.markdown("#### Why the optimum is bang-bang")
        cP, price = switching_price(P)
        st.pyplot(switching_figure(cP, price, params=P)[0], use_container_width=True)
        st.markdown(
            "<div class='dfp-note'>Using $c_P$ as the independent variable makes the "
            "problem <b>linear in</b> $\\sigma = 1/(1-u)$. Minimising time then becomes "
            "a linear program whose optimum places all washing where one unit of "
            "washing is cheapest — and that price decreases monotonically up to "
            "$c_{P,f}$. Hence: concentrate first, wash afterwards.</div>",
            unsafe_allow_html=True)
    with st.expander("Model equations", expanded=False):
        st.latex(r"\dot V = (u-1)\,p,\qquad \dot M_L = -r_L c_L p,"
                 r"\qquad \dot M_P = -r_P c_P p")
        st.latex(r"p = kA\ln\frac{c_g}{c_P},\qquad "
                 r"r_\bullet = \frac{\gamma}{1+(\gamma-1)\exp\!\big(p/(k_M A)\big)}")
        st.latex(r"c_P = M_P/V,\qquad c_L = M_L/V,\qquad 0\le u\le 1")
        st.markdown(
            f"Specifications: $c_{{P,f}}={P.cP_f:g}$, $c_{{L,f}}\\leq{P.cL_f:g}$, "
            f"$c_L\\leq{P.cL_max:g}$ mol m⁻³; charge "
            f"$V_0={P.V0*1e3:g}$ L, $c_{{P,0}}={P.cP0:g}$, $c_{{L,0}}={P.cL0:g}$.")


def page_open_loop() -> None:
    _hero("Open loop — constant input", "Task 2: why a single value of u cannot win")
    c1, c2 = st.columns([1, 1])
    with c1:
        u_vals = st.multiselect("constant inputs to compare",
                                [0.0, 0.3, 0.5, 0.595, 0.6, 0.7, 0.86, 1.0],
                                default=[0.0, 0.5, 0.595, 0.7, 1.0])
    with c2:
        stop = st.toggle("stop as soon as both specifications hold", value=False)
    if not u_vals:
        st.info("Pick at least one value of u.")
        return
    runs = [closed_loop(ConstantU(float(u), P), nominal_plant(P), params=P,
                        label=f"u = {u:g}", stop_on_spec=stop) for u in sorted(u_vals)]
    a = analytic_optimum(P)
    fig, _ = trajectory_figure(runs, params=P, log_cL=False, mark_finish=stop,
                               reference=({"t": a.t, "cP": a.cP, "cL": a.cL,
                                           "V": a.V, "u": a.u} if stop else None),
                               title="Constant-input operation")
    st.pyplot(fig, use_container_width=True)
    st.markdown("#### Outcome after 6 h")
    st.dataframe(
        [{"u": float(r.label.split("=")[1]),
          "cP end": round(float(r.cP[-1]), 2), "cL end": round(float(r.cL[-1]), 2),
          "V end [L]": round(float(r.V[-1]) * 1e3, 2),
          "t(cP=100) [h]": None if np.isnan(r.t_cP_spec) else round(r.t_cP_spec / 3600, 3),
          "t(cL=15) [h]": None if np.isnan(r.t_cL_spec) else round(r.t_cL_spec / 3600, 3),
          "on spec": r.spec_ok} for r in runs],
        use_container_width=True, hide_index=True)
    st.markdown(
        "<div class='dfp-note'>Only a narrow band around <b>u ≈ 0.6</b> reaches both "
        "targets, and it needs about <b>5.05 h</b> — some 42 % more than the "
        "time-optimal 3.54 h. Small u concentrates without washing; large u washes "
        "without concentrating.</div>", unsafe_allow_html=True)


def page_controllers() -> None:
    _hero("Controller laboratory", "Tasks 3–5: objective functions, horizon, disturbances")
    a = analytic_optimum(P)
    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        picks = st.multiselect(
            "controllers",
            ["min-time MPC", "ℓ₁-time MPC", "tracking MPC (Eq. 3)",
             "scaled tracking MPC", "economic MPC", "threshold policy (Eq. 4)",
             "analytic bang-bang"],
            default=["min-time MPC", "tracking MPC (Eq. 3)", "threshold policy (Eq. 4)"])
    with c2:
        N = st.select_slider("prediction horizon N", [5, 10, 20, 30, 50], value=20)
    with c3:
        scenario = st.selectbox("plant", ["nominal", "filter-cake tear",
                                          "protein leakage"])
    plant = {"nominal": nominal_plant(P), "filter-cake tear": tear_plant(P),
             "protein leakage": leakage_plant(P, beta=1.3, kM_P=3e-6)}[scenario]

    mapping = {
        "min-time MPC": lambda: _nmpc("min_time", N, P.t_max),
        "ℓ₁-time MPC": lambda: _nmpc("l1_time", N, P.t_max),
        "tracking MPC (Eq. 3)": lambda: _nmpc("tracking", N, P.t_max),
        "scaled tracking MPC": lambda: _nmpc("tracking_scaled", N, P.t_max),
        "economic MPC": lambda: _nmpc("economic", N, P.t_max),
        "threshold policy (Eq. 4)": lambda: ThresholdPolicy(params=P),
        "analytic bang-bang": lambda: BangBang(P),
    }
    if not picks:
        st.info("Select at least one controller.")
        return
    runs = []
    prog = st.progress(0.0, text="simulating …")
    for i, name in enumerate(picks):
        ctrl = mapping[name]()
        if hasattr(ctrl, "reset"):
            ctrl.reset()
        runs.append(closed_loop(ctrl, plant, params=P,
                                label=f"{name}" + (f" (N={N})" if "MPC" in name else "")))
        prog.progress((i + 1) / len(picks), text=f"simulating {name} …")
    prog.empty()

    window = None
    if scenario == "filter-cake tear":
        m = np.flatnonzero((runs[0].cP >= 30.0) & (runs[0].cP <= 60.0))
        window = (float(runs[0].t_h[m[0]]), float(runs[0].t_h[m[-1]])) if m.size else None
    fig, _ = trajectory_figure(runs, params=P, tear_window=window,
                               reference={"t": a.t, "cP": a.cP, "cL": a.cL,
                                          "V": a.V, "u": a.u},
                               reference_label="time-optimal (analytic)",
                               title=f"Closed loop on the {scenario} plant")
    st.pyplot(fig, use_container_width=True)
    st.markdown("#### Scoreboard")
    st.dataframe(
        [{"controller": r.label,
          "batch time [h]": round(r.batch_time_h, 4) if r.finished else None,
          "gap to optimum [%]": (round(100 * (r.batch_time_h - a.T_h) / a.T_h, 2)
                                 if r.finished else None),
          "cP end": round(float(r.cP[-1]), 2), "cL end": round(float(r.cL[-1]), 2),
          "peak cL": round(r.cL_peak, 1),
          "cP overshoot": round(r.cP_overshoot, 2),
          "on spec": r.spec_ok,
          "solve [ms]": (round(float(np.mean(r.solve_times)) * 1e3, 1)
                         if r.solve_times.size else None)} for r in runs],
        use_container_width=True, hide_index=True)


def page_robustness() -> None:
    _hero("Robustness", "Additional task 1: a poorly estimated $k_{M,L}$ and four remedies")
    P8 = P.with_(t_max=8 * 3600)
    c1, c2 = st.columns([1, 2])
    with c1:
        factor = st.select_slider("true $k_{M,L}$ / nominal", [0.25, 0.5, 0.75, 1.0],
                                  value=0.25)
        N = st.select_slider("horizon N", [10, 20, 30], value=20)
    with c2:
        picks = st.multiselect(
            "controllers",
            ["nominal-model MPC", "perfect-model MPC", "back-off MPC (−200)",
             "multi-stage NMPC", "adaptive NMPC (MHE)"],
            default=["nominal-model MPC", "multi-stage NMPC", "adaptive NMPC (MHE)"])
    plant = mismatch_plant(factor, P8)
    factory = {
        "nominal-model MPC": lambda: _nmpc("min_time", N, P8.t_max),
        "perfect-model MPC": lambda: build_nmpc("min_time", N, params=P8,
                                                theta=P8.scaled(kM_L=factor).theta),
        "back-off MPC (−200)": lambda: build_nmpc("min_time", N, params=P8,
                                                 back_off_cL=200.0),
        "multi-stage NMPC": lambda: _multistage(N, P8.t_max),
        "adaptive NMPC (MHE)": lambda: AdaptiveNMPC(
            build_nmpc("min_time", N, params=P8), MHE(P8), params=P8,
            rng=np.random.default_rng(7)),
    }
    if not picks:
        st.info("Select at least one controller.")
        return
    runs, kappa_logs = [], {}
    prog = st.progress(0.0, text="simulating …")
    for i, name in enumerate(picks):
        ctrl = factory[name]()
        if hasattr(ctrl, "reset"):
            ctrl.reset()
        runs.append(closed_loop(ctrl, plant, params=P8, label=name))
        if isinstance(ctrl, AdaptiveNMPC):
            kappa_logs[f"MHE, true {factor:g}"] = list(ctrl.kappa_log)
        prog.progress((i + 1) / len(picks), text=f"simulating {name} …")
    prog.empty()

    fig, _ = trajectory_figure(runs, params=P8,
                               title=f"$k_{{M,L}}^{{true}} = {factor:g}\\,k_{{M,L}}$")
    st.pyplot(fig, use_container_width=True)
    st.dataframe(
        [{"controller": r.label,
          "batch time [h]": round(r.batch_time_h, 4) if r.finished else None,
          "peak cL": round(r.cL_peak, 1),
          "violation of cL_max": round(r.cL_violation, 2),
          "cL ≤ 570": r.cL_violation <= 1e-6,
          "solve [ms]": (round(float(np.mean(r.solve_times)) * 1e3, 1)
                         if r.solve_times.size else None)} for r in runs],
        use_container_width=True, hide_index=True)
    if kappa_logs:
        st.markdown("#### Online identification")
        st.pyplot(estimator_figure(kappa_logs, {f"MHE, true {factor:g}": factor},
                                   dt_h=P8.dt_ctrl / 3600.0)[0],
                  use_container_width=True)
    st.markdown(
        "<div class='dfp-note'>A smaller $k_{M,L}$ makes the membrane reject "
        "<i>more</i> lactose, so the retentate rises far above the prediction and the "
        "crystallisation limit is breached. Constraint back-off cannot help — the "
        "prediction itself is biased. Multi-stage NMPC (four branches, robust horizon 1) "
        "and MHE-based adaptation both restore feasibility.</div>",
        unsafe_allow_html=True)


def page_montecarlo() -> None:
    _hero("Monte-Carlo campaign", "Randomised plants: how often is the batch on spec?")
    P8 = P.with_(t_max=8 * 3600)
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("random plants", 5, 80, 20, 5)
    with c2:
        picks = st.multiselect("controllers",
                               ["nominal-model MPC", "multi-stage NMPC",
                                "adaptive NMPC (MHE)"],
                               default=["nominal-model MPC", "multi-stage NMPC"])
    if not st.button("Run campaign", type="primary") or not picks:
        st.info("Choose the controllers and press **Run campaign**. "
                "kM_L, k, α and c_g are drawn uniformly around their nominal values.")
        return
    factories = {
        "nominal-model MPC": lambda: build_nmpc("min_time", 20, params=P8),
        "multi-stage NMPC": lambda: build_multistage_nmpc(20, params=P8),
        "adaptive NMPC (MHE)": lambda: AdaptiveNMPC(
            build_nmpc("min_time", 20, params=P8), MHE(P8), params=P8,
            rng=np.random.default_rng(11)),
    }
    times, peaks, rows = {}, {}, []
    for name in picks:
        with st.spinner(f"{name}: {n} plants …"):
            mc = run_campaign(factories[name], label=name, n_draws=n, params=P8)
        times[name] = mc.batch_times_h[mc.spec_ok]
        peaks[name] = mc.cL_peaks
        rows.append(mc.summary())
    st.dataframe(rows, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(histogram_figure(times, xlabel="batch time [h]",
                                   title="batch time")[0], use_container_width=True)
    with c2:
        st.pyplot(histogram_figure(peaks, xlabel="peak $c_L$ [mol m$^{-3}$]",
                                   title="worst lactose concentration",
                                   vline=P.cL_max,
                                   vline_label="crystallisation limit")[0],
                  use_container_width=True)


def page_economics() -> None:
    _hero("Economic MPC", "Batch time against a day-ahead electricity tariff")
    a = analytic_optimum(P)
    c1, c2 = st.columns([1, 1])
    with c1:
        cts = st.multiselect("value of time [€/h]", [0.0, 0.5, 2.0, 10.0, 30.0],
                             default=[0.0, 2.0, 30.0])
        hour = st.slider("batch starts at [h of day]", 0, 23, 6)
    with c2:
        st.pyplot(tariff_figure(lambda_tou)[0], use_container_width=True)
    if not cts:
        st.info("Select at least one value of time.")
        return
    runs, pareto, rows = [], [], []
    for ct in sorted(cts):
        mpc = build_nmpc("economic", 20, params=P, value_of_time=ct)
        r = closed_loop(mpc, nominal_plant(P), params=P, label=f"$c_t$ = {ct:g} €/h")
        runs.append(r)
        pareto.append((r.batch_time_h, r.energy(), f"{ct:g} €/h"))
        rows.append({"value of time [€/h]": ct,
                     "batch time [h]": round(r.batch_time_h, 4),
                     "gap to optimum [%]": round(100 * (r.batch_time_h - a.T_h) / a.T_h, 2),
                     "energy [kWh]": round(r.energy(), 3),
                     "electricity [€]": round(r.energy_cost(t_start=hour * 3600.0), 4),
                     "solvent [L]": round(r.diavolume * P.V0 * 1e3, 2)})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    c1, c2 = st.columns([1, 1.35])
    with c1:
        st.pyplot(pareto_figure(pareto, xlabel="batch time [h]",
                                ylabel="pump energy [kWh]",
                                title="time / energy trade-off")[0],
                  use_container_width=True)
    with c2:
        fig, _ = trajectory_figure(runs, params=P,
                                   reference={"t": a.t, "cP": a.cP, "cL": a.cL,
                                              "V": a.V, "u": a.u},
                                   reference_label="time-optimal",
                                   title="economic MPC")
        st.pyplot(fig, use_container_width=True)
    st.markdown(
        "<div class='dfp-note'>Minimising <i>solvent</i> leads to the same bang-bang "
        "law as minimising <i>time</i> (same linear program, same price function), so "
        "those two objectives never conflict. The only real economic lever is the "
        "tariff: with a low value of time the controller stretches the cheap, "
        "low-power concentration phase and compresses the expensive washing phase."
        "</div>", unsafe_allow_html=True)


PAGES = {
    "Overview": page_overview,
    "Open loop": page_open_loop,
    "Controller lab": page_controllers,
    "Robustness": page_robustness,
    "Monte-Carlo": page_montecarlo,
    "Economics": page_economics,
}
