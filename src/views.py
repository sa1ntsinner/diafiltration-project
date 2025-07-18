# views.py
# Streamlit app for simulating and evaluating MPC strategies in diafiltration

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Core components and controller tools
from core.params import default as P
from core.tariff import lambda_tou  # Time-of-use electricity tariff
from control.builder import build_mpc
from control import mpc_robust

# Simulation and scenario imports
from sim import (
    simulate,
    constant_u,
    threshold_policy,
    mpc_time_opt,
    Nominal,
    Tear,
    KmMismatch,
    ProteinLeakage,
)

# Monte-Carlo simulation
from experiments.montecarlo import run as mc_run

# ─────────────────────────── Utility Functions ─────────────────────────────

def batch_time(t: np.ndarray, cP: np.ndarray, cL: np.ndarray, tol: float = 1e-3) -> float:
    """Return duration (in hours) to meet spec constraints or end of run."""
    idx = np.where((cP >= P.cP_star - tol) & (cL <= P.cL_star + tol))[0]
    done = idx[0] if idx.size else len(t) - 1
    return t[done] / 3600

def spec_controller(N: int, *, rho_time: float = 0.10, params=P):
    """Returns standard spec-tracking MPC controller with quadratic objective."""
    solver, meta, LBG, UBG = build_mpc(N, weights=dict(rho_time=rho_time), params=params)

    def _ctrl(state: np.ndarray) -> float:
        x0 = np.hstack([np.tile(state, meta["N"] + 1), meta["u_init"]])
        sol = solver(x0=x0, p=state, lbg=LBG, ubg=UBG)
        return float(sol["x"].full().ravel()[meta["Uslice"]][0])

    return _ctrl

def econ_controller(N: int, *, params=P):
    """Returns economic MPC controller minimizing TOU electricity costs."""
    solver, meta, LBG, UBG = build_mpc("econ", N, params=params, weights=dict(lambda_fun=lambda_tou))

    def _ctrl(state: np.ndarray) -> float:
        x0 = np.hstack([np.tile(state, meta["N"] + 1), meta["u_init"]])
        sol = solver(x0=x0, p=state, lbg=LBG, ubg=UBG)
        return float(sol["x"].full().ravel()[meta["Uslice"]][0])

    return _ctrl

# ───────────────────────── realistic energy & cost ─────────────────────────
def energy_cost(
    t: np.ndarray,
    u: np.ndarray,
    *,
    pump_idle_kw: float = 0.5,      # stand-by draw
    pump_dyn_kw:  float = 2.5       # additional @ u = 1.0
) -> tuple[float, float]:
    """
    Integrate energy use and electricity cost over the batch.

    Pump power model
    ----------------
        P(t) = P_idle + P_dyn · u(t)

    (Roughly mimics a centrifugal feed-pump where flow ∝ u and
    hydraulic power ∝ flow.)

    Cost integration
    ----------------
    Uses the *continuous* tariff λ(t) from core.tariff.lambda_tou.
    """
    from core.tariff import lambda_tou  # local import to avoid cycles

    # guard against 1-sample mismatch (same logic used in plot_charts)
    if len(u) < len(t):
        pad = np.full(len(t) - len(u), u[-1] if len(u) else 0.0)
        u_use = np.concatenate([u, pad])
    else:
        u_use = u[: len(t)]

    # time-step vector  Δt_k  (first element = 0 → energy=0)
    dt = np.diff(t, prepend=t[0])

    # instantaneous power  [kW]  and energy  [kWh]
    power_kw = pump_idle_kw + pump_dyn_kw * u_use
    energy_kwh = np.sum(power_kw * dt) / 3600.0

    # cost: ∑  P_k · Δt_k · λ(t_k)
    price = np.vectorize(lambda_tou)(t)
    cost_eur = float(np.sum(power_kw * dt * price) / 3600.0)

    return cost_eur, energy_kwh

# ──────────────────────── Open-loop Simulation View ─────────────────────────

def show_open_loop() -> None:
    """Visualize impact of different constant control inputs."""
    st.markdown("## Open-loop simulation")

    num_u = st.slider("How many constant-u values?", 1, 5, 1)
    u_vals = [st.slider(f"u value {i+1}", 0.0, 1.0, 0.5, 0.01, key=f"u_{i}") for i in range(num_u)]
    if not u_vals:
        st.warning("Select at least one value."); return

    cols = st.columns(3, gap="small")
    fig_cP, ax_cP = plt.subplots(figsize=(4, 3))
    fig_cL, ax_cL = plt.subplots(figsize=(4, 3))
    fig_V , ax_V  = plt.subplots(figsize=(4, 3))

    for u in u_vals:
        t, V, ML, _ = simulate(constant_u(u), Nominal(P))
        cP, cL = P.MP / V, ML / V
        label = f"u = {u:.2f}"
        ax_cP.plot(t/3600, cP, label=label)
        ax_cL.plot(t/3600, cL, label=label)
        ax_V .plot(t/3600, V , label=label)

    # Draw reference lines
    ax_cP.axhline(P.cP_star, ls="--", color="k"); ax_cP.set_ylabel("$c_P$"); ax_cP.legend()
    ax_cL.axhline(P.cL_star, ls="--", color="k"); ax_cL.axhline(P.cL_max, ls=":", color="r")
    ax_cL.set_ylabel("$c_L$"); ax_cL.legend()
    ax_V.set_ylabel("V [m³]"); ax_V.legend()

    cols[0].pyplot(fig_cP, use_container_width=True)
    cols[1].pyplot(fig_cL, use_container_width=True)
    cols[2].pyplot(fig_V , use_container_width=True)

    st.markdown("✅ Constant-u policies and their effect.")

# ──────────────────────────── Generic Chart Plotting ───────────────────────

# ──────────────────────────── Unified plotting helper ──────────────────────

def plot_charts(
    title: str,
    t: np.ndarray,
    cP: np.ndarray,
    cL: np.ndarray,
    u: np.ndarray,
    *,
    highlight_tear: bool = False,
) -> None:
    """
    Show three *separate* 4×3-inch panels (cL, cP, u) in a single row.
    
    Notes:
      - One Streamlit column per metric.
      - Each panel has its own legend.
      - Control vector `u` is padded if one sample shorter than `t`.
      - Tear window shading (30 ≤ cP ≤ 60) applied before rendering.
    """

    # ─────────────────── Data preparation ────────────────────────────────
    time_h = np.asarray(t) / 3600.0  # Convert time to hours for plotting

    if len(u) < len(time_h):
        # Pad `u` if it's one sample shorter (common for MPC stopping at spec)
        pad = np.full(len(time_h) - len(u), u[-1] if len(u) else 0.0)
        u_plot = np.concatenate([u, pad])
    else:
        u_plot = u[: len(time_h)]

    # ─────────────────── Identify tear region once ───────────────────────
    t0, t1 = None, None
    if highlight_tear:
        mask = (cP >= 30.0) & (cP <= 60.0)
        if np.any(mask):
            i0 = np.argmax(mask)               # first True
            i1 = i0 + np.argmax(~mask[i0:])    # first False after i0
            t0, t1 = time_h[i0], time_h[i1]

    # ─────────────────── Streamlit layout with 3 columns ─────────────────
    st.markdown(f"**{title}**")
    col_cP, col_cL, col_u = st.columns(3, gap="small")

    # Panel 1 – Protein concentration
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(time_h, cP, color="C1", label="cP")
    ax.axhline(P.cP_star, ls="--", color="grey", label="cP target 100")
    if t0 is not None:
        ax.axvspan(t0, t1, color="yellow", alpha=0.30, label="Disturbance period")
    ax.set_ylabel("Protein cP  [mol m⁻³]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="best")
    col_cP.pyplot(fig, use_container_width=True)

    # Panel 2 – Lactose concentration
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(time_h, cL, color="C0", label="cL")
    ax.axhline(P.cL_star, ls="--", color="grey", label="cL target 15")
    ax.axhline(P.cL_max, ls=":", color="r", label="cL max 570")
    if t0 is not None:
        ax.axvspan(t0, t1, color="yellow", alpha=0.30, label="Disturbance period")
    ax.set_ylabel("Lactose cL  [mol m⁻³]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="best")
    col_cL.pyplot(fig, use_container_width=True)

    # Panel 3 – Control trajectory
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.step(time_h, u_plot, where="post", color="C2", label="u")
    if t0 is not None:
        ax.axvspan(t0, t1, color="yellow", alpha=0.30, label="Disturbance period")
    ax.set_ylabel("Control u")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="best")
    col_u.pyplot(fig, use_container_width=True)

# ──────────────────────────────── MPC Page ─────────────────────────────────

def show_mpc() -> None:
    """Main visualization page for different MPC strategies."""
    st.markdown("## Closed-loop MPC simulation")

    # 1.1. Tracking MPC
    st.markdown("---"); st.markdown("### 1.1. Tracking MPC")
    N = st.slider("Prediction horizon N", 5, 50, 20)
    t_b, V_b, ML_b, u_b = simulate(spec_controller(N), Nominal(P))
    cP_b, cL_b = P.MP / V_b, ML_b / V_b
    plot_charts("Tracking MPC", t_b, cP_b, cL_b, u_b)
    st.info(f"⏱️ Batch time **{batch_time(t_b, cP_b, cL_b):.2f} h**")

    # 1.2. Tracking MPC fixed N
    st.markdown("---"); st.markdown("### 1.2. Tracking MPC (N=5,20,50)")

    data ={}
    Ns = [5, 20, 50]
    for n in Ns:
        t_b, V_b, ML_b, u_b = simulate(spec_controller(n), Nominal(P))
        cP_b, cL_b = P.MP / V_b, ML_b / V_b
        data[f't_n{n}'] = t_b / 3600.0
        data[f'cP_n{n}'] = cP_b
        data[f'cL_n{n}'] = cL_b
        data[f'u_n{n}'] = u_b

    # # Define colors and labels for each N
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    col_cP, col_cL, col_u = st.columns(3, gap="small")

    # Panel 1 – Protein concentration
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, n in enumerate(Ns):
        ax.plot(data[f't_n{n}'], data[f'cP_n{n}'], label=f'N={n}', color=colors[i])
    ax.axhline(P.cP_star, ls="--", color="grey", label="cP target 100")
    ax.set_ylabel("Protein cP  [mol m⁻³]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="best")
    col_cP.pyplot(fig, use_container_width=True)

    # Panel 2 – Lactose concentration
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, n in enumerate(Ns):
        ax.plot(data[f't_n{n}'], data[f'cL_n{n}'], label=f'N={n}', color=colors[i])
    ax.axhline(P.cL_star, ls="--", color="grey", label="cL target 15")
    ax.axhline(P.cL_max, ls=":", color="r", label="cL max 570")
    ax.set_ylabel("Lactose cL  [mol m⁻³]")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="best")
    col_cL.pyplot(fig, use_container_width=True)

    # Panel 3 – Control trajectory
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, n in enumerate(Ns):
        t = data[f't_n{n}']
        u = data[f'u_n{n}']
        min_len = min(len(t), len(u))
        ax.step(t[:min_len], u[:min_len], label=f'N={n}', where='post', color=colors[i])
    ax.set_ylabel("Control u")
    ax.set_xlabel("Time [h]")
    ax.legend(loc="best")
    col_u.pyplot(fig, use_container_width=True)

    for n in Ns:
        st.info(f"⏱️ Batch time of Tracking MPC with N={n}: **{batch_time(data[f't_n{n}']*3600.0, data[f'cP_n{n}'], data[f'cL_n{n}']):.2f} h**")

    # 2. Threshold policy comparison
    st.markdown("---"); st.markdown("### 2. Threshold policy ($u=0.86$ if $c_P\\ge55$)")
    t_th, V_th, ML_th, u_th = simulate(threshold_policy(), Nominal(P))
    cP_th, cL_th = P.MP / V_th, ML_th / V_th
    for lbl, t, cP, cL, u in [("MPC", t_b, cP_b, cL_b, u_b), ("Threshold", t_th, cP_th, cL_th, u_th)]:
        plot_charts(lbl, t, cP, cL, u)

    # 3. Time-optimal MPC
    st.markdown("---"); st.markdown("### 3. Time-optimal MPC")
    N_opt = st.slider("Horizon (time-opt.)", 5, 50, 20, key="topth")
    t_to, V_to, ML_to, u_to = simulate(mpc_time_opt(N_opt), Nominal(P))
    cP_to, cL_to = P.MP / V_to, ML_to / V_to
    plot_charts("Time-optimal MPC", t_to, cP_to, cL_to, u_to)
    st.success(f"🏁 Time-opt batch **{batch_time(t_to, cP_to, cL_to):.2f} h**  "
               f"vs Tracking-MPC **{batch_time(t_b,  cP_b,  cL_b ):.2f} h** ")

    # 4. Economic MPC with tariff cost analysis
    st.markdown("---"); st.markdown("### 4. Economic MPC")
    N_econ = st.slider("Horizon (economic)", 5, 50, 20, key="econ")
    t_e, V_e, ML_e, u_e = simulate(econ_controller(N_econ), Nominal(P))
    cP_e, cL_e = P.MP / V_e, ML_e / V_e
    plot_charts("Economic MPC", t_e, cP_e, cL_e, u_e)

    # ─────────────────── energy & cost summary ──────────────────────
    euro, kwh = energy_cost(t_e, u_e)        # new realistic model
    avg_ct = euro / kwh if kwh else 0.0      # € / kWh actually paid

    # Short, information-dense status line
    st.info(
        f"💡 { kwh:.2f} kWh → **€{ euro:.2f}**  "
        f"(spot-price avg ≈ {avg_ct:.2f} €/kWh)"
    )

# ─────────────────────────────── Test Page ─────────────────────────────────

def show_tests() -> None:
    """Simulate MPC robustness across faulty or perturbed plants."""
    st.markdown("## Test scenarios on best MPC")

    # 1. Filter-cake tear scenario
    st.subheader("1. Filter-cake tear disturbance")
    t, V, ML, u = simulate(econ_controller(20), Tear(P))
    plot_charts("Tear disturbance", t, P.MP / V, ML / V, u, highlight_tear=True)

    # 2. Plant-model mismatch scenario
    st.subheader("2. Plant-model mismatch")
    tol = 1e-3
    summary = []

    for factor in [0.75, 0.5, 0.25]:
        scen = KmMismatch(factor, P)
        t, V, ML, u = simulate(econ_controller(20), scen)
        plot_charts(f"Mismatch factor {factor}", t, P.MP / V, ML / V, u)

        cP = P.MP / V
        cL = ML / V

        # 1️⃣ Check if spec constraints ever met
        idx = np.where((cP >= P.cP_star - tol) & (cL <= P.cL_star + tol))[0]
        spec_met = idx.size > 0

        # 2️⃣ Check if path constraints were ever violated
        path_violated = np.any(cP > P.cP_star + tol) or np.any(cL > P.cL_max + tol)

        # ✅ Final decision: both spec must be met and no violation
        ok = spec_met and not path_violated

        # Use first spec reach time if spec_met, else t_final
        t_b = t[idx[0]] / 3600 if spec_met else P.t_final / 3600
        peak = float(np.max(cL))
        summary.append((factor, ok, t_b, peak))

    st.markdown("##### Batch-time summary")
    cols = st.columns(3, gap="small")
    for col, (factor, ok, t_b, peak) in zip(cols, summary):
        if ok:
            msg = f"✅ {t_b:.2f} h, peak $c_L≈{peak:.0f}$"
        else:
            msg = f"❌ Spec or constraints failed, peak $c_L≈{peak:.0f}$"
        col.info(f"factor {factor}: {msg}")

    # 3. Protein leakage scenario
    st.subheader("3. Protein leakage")
    scen = ProteinLeakage()
    t, V, ML, u = simulate(econ_controller(20), scen)
    plot_charts("Protein leakage", t, P.MP / V, ML / V, u)

    # 4. Monte-Carlo robustness test
    st.subheader("4. Monte-Carlo robustness")
    draws = st.slider("Number of random plants", 20, 300, 100, 20)
    if st.button("Run Monte-Carlo"):
        ctrl = mpc_robust(20)
        times, peaks, ok = mc_run(draws, ctrl)

        col_ok, col_t, col_p = st.columns(3)
        col_ok.metric("Pass-rate", f"{100*sum(ok)/len(ok):.1f} %")
        col_t.metric("Median time [h]", f"{np.median(times):.2f}")
        col_p.metric("90-perc peak $c_L$", f"{np.percentile(peaks, 90):.0f}")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(times, bins=15, alpha=0.75)
        ax.set_xlabel("Batch time [h]"); ax.set_ylabel("# runs")
        st.pyplot(fig, use_container_width=True)