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

def energy_cost(t: np.ndarray, u: np.ndarray, *, pump_kw: float = 2.0) -> tuple[float, float]:
    """
    Estimate energy usage and cost:
    - Assumes linear pump power draw with control signal `u`.
    - Applies time-of-use tariff for electricity.
    """
    dt = np.diff(t, prepend=t[0])
    n = min(len(u), len(dt))  # handle possible length mismatch
    lam = np.vectorize(lambda_tou)(t[:n])
    kwh = np.sum(u[:n] * pump_kw * dt[:n]) / 3600.0
    euro = float(np.sum(lam * u[:n] * pump_kw * dt[:n]) / 3600.0)
    return euro, kwh

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
    ax_V.set_ylabel("V [m³]"); ax_V.set_xlabel("Time [h]"); ax_V.legend()

    cols[0].pyplot(fig_cP, use_container_width=True)
    cols[1].pyplot(fig_cL, use_container_width=True)
    cols[2].pyplot(fig_V , use_container_width=True)

    st.markdown("✅ Constant-u policies and their effect.")

# ──────────────────────────── Generic Chart Plotting ───────────────────────

def plot_charts(title: str, t, cP, cL, u) -> None:
    """Plot cP, cL, and u over time in 3 columns."""
    tol = 1e-3
    idx = np.where((cP >= P.cP_star - tol) & (cL <= P.cL_star + tol))[0]
    idx_end = idx[0] if idx.size else len(t) - 1

    # Trim signals up to constraint satisfaction or end
    t_p, cP_p, cL_p = t[:idx_end+1], cP[:idx_end+1], cL[:idx_end+1]
    u_p = u[: max(1, min(len(u), idx_end))]

    st.markdown(f"**{title}**")
    c1, c2, c3 = st.columns(3, gap="small")

    # cP plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(t_p/3600, cP_p); ax.axhline(P.cP_star, ls="--", color="k")
    ax.set_ylabel("$c_P$")
    c1.pyplot(fig, use_container_width=True)

    # cL plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(t_p/3600, cL_p)
    ax.axhline(P.cL_star, ls="--", color="k")
    ax.axhline(P.cL_max , ls=":", color="r")
    ax.set_ylabel("$c_L$")
    c2.pyplot(fig, use_container_width=True)

    # u plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.step(t_p[:len(u_p)]/3600, u_p, where="post")
    ax.set_ylabel("$u$"); ax.set_xlabel("Time [h]")
    c3.pyplot(fig, use_container_width=True)

# ──────────────────────────────── MPC Page ─────────────────────────────────

def show_mpc() -> None:
    """Main visualization page for different MPC strategies."""
    st.markdown("## Closed-loop MPC simulation")

    # 1. Baseline MPC
    st.markdown("---"); st.markdown("### 1. Baseline MPC")
    N = st.slider("Prediction horizon N", 5, 50, 20)
    t_b, V_b, ML_b, u_b = simulate(spec_controller(N), Nominal(P))
    cP_b, cL_b = P.MP / V_b, ML_b / V_b
    plot_charts("Baseline MPC", t_b, cP_b, cL_b, u_b)
    st.info(f"⏱️ Batch time **{batch_time(t_b, cP_b, cL_b):.2f} h**")

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
               f"vs Threshold **{batch_time(t_th, cP_th, cL_th):.2f} h**")

    # 4. Economic MPC with tariff cost analysis
    st.markdown("---"); st.markdown("### 4. Economic MPC (time-of-use tariff)")
    N_econ = st.slider("Horizon (economic)", 5, 50, 20, key="econ")
    t_e, V_e, ML_e, u_e = simulate(econ_controller(N_econ), Nominal(P))
    cP_e, cL_e = P.MP / V_e, ML_e / V_e
    plot_charts("Economic MPC", t_e, cP_e, cL_e, u_e)

    euro, kwh = energy_cost(t_e, u_e)
    st.info(f"💡 Total energy {kwh:.3f} kWh  •  cost **€{euro:.4f}**  "
            "(0–2 h & 4–6 h = €0.10/kWh, 2–4 h = €0.35/kWh)")

# ─────────────────────────────── Test Page ─────────────────────────────────

def show_tests() -> None:
    """Simulate MPC robustness across faulty or perturbed plants."""
    st.markdown("## Test scenarios")

    # 1. Filter-cake tear scenario
    st.subheader("1. Filter-cake tear disturbance")
    t, V, ML, u = simulate(spec_controller(20), Tear(P))
    plot_charts("Tear disturbance", t, P.MP / V, ML / V, u)

    # 2. Parameter mismatch with robust MPC
    st.subheader("2. Plant-model mismatch (robust MPC)")
    tol = 1e-3
    summary = []  # summary list of results

    for factor in [0.75, 0.5, 0.25]:
        scen = KmMismatch(factor, P)
        t, V, ML, u = simulate(mpc_robust(20), scen)
        plot_charts(f"Mismatch factor {factor}", t, P.MP / V, ML / V, u)

        cP, cL = P.MP / V, ML / V
        ok = (cP[-1] >= P.cP_star - tol) and (cL[-1] <= P.cL_star + tol)
        t_b = batch_time(t, cP, cL) if ok else P.t_final/3600
        peak = float(np.max(cL))
        summary.append((factor, ok, t_b, peak))

    st.markdown("##### Batch-time summary")
    cols = st.columns(3, gap="small")
    for col, (factor, ok, t_b, peak) in zip(cols, summary):
        msg = f"✅ {t_b:.2f} h, peak $c_L≈{peak:.0f}$" if ok else f"❌ > {t_b:.2f} h, peak $c_L≈{peak:.0f}$"
        col.info(f"factor {factor}: {msg}")

    # 3. Protein leakage scenario
    st.subheader("3. Protein leakage (β = 1.3)")
    scen = ProteinLeakage()
    t, V, ML, u = simulate(spec_controller(20), scen)
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
