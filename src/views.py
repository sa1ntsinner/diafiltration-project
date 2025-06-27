# src/views.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ── core / control / sim layers ────────────────────────────────────────────
from core.params import default as P
from control.builder import build_mpc
from control       import mpc_robust
from sim import (
    simulate,
    constant_u,
    threshold_policy,
    mpc_time_opt,
    Nominal,
    Tear,
    KmMismatch,
)

# ───────────────────────── helper ─────────────────────────
def batch_time(t, cP, cL, tol=1e-3):
    idx = np.where((cP >= P.cP_star - tol) & (cL <= P.cL_star + tol))[0]
    done = idx[0] if idx.size else len(t) - 1
    return t[done] / 3600


def spec_controller(N: int, *, params=P, rho_time: float = 0.10):
    """
    Quadratic spec-tracking MPC with a fixed per-step time penalty (ρ_time).
    Returns a callable u = ctrl(state).
    """
    solver, meta, LBG, UBG = build_mpc(
        N,                          # horizon
        weights=dict(rho_time=rho_time),
        params=params,
    )

    def _ctrl(state: np.ndarray) -> float:
        x0 = np.hstack([np.tile(state, meta["N"] + 1), meta["u_init"]])
        sol = solver(x0=x0, p=state, lbg=LBG, ubg=UBG)
        return float(sol["x"].full().ravel()[meta["Uslice"]][0])

    return _ctrl


# ──────────────────────────────────────────────────────────
def show_open_loop():
    st.markdown("## Open-loop Simulation")

    num_u = st.slider("How many constant-u values?", 1, 5, 1)
    u_values = [
        st.slider(f"u value {i+1}", 0.0, 1.0, 0.5, 0.01, key=f"u_{i}")
        for i in range(num_u)
    ]
    if not u_values:
        st.warning("Please select at least one u value.")
        return

    col1, col2, col3 = st.columns(3, gap="small")
    fig_cP, ax_cP = plt.subplots(figsize=(4, 3))
    fig_cL, ax_cL = plt.subplots(figsize=(4, 3))
    fig_V,  ax_V  = plt.subplots(figsize=(4, 3))

    for u in u_values:
        t, V, ML, _ = simulate(constant_u(u), Nominal(P))
        cP, cL = P.MP / V, ML / V
        lab = f"u = {u:.2f}"
        ax_cP.plot(t/3600, cP, label=lab)
        ax_cL.plot(t/3600, cL, label=lab)
        ax_V .plot(t/3600, V,  label=lab)

    ax_cP.axhline(P.cP_star, ls="--", color="k");  ax_cP.set_ylabel("$c_P$");  ax_cP.legend()
    ax_cL.axhline(P.cL_star, ls="--", color="k");  ax_cL.axhline(P.cL_max, ls=":", color="r")
    ax_cL.set_ylabel("$c_L$");  ax_cL.legend()
    ax_V .set_ylabel("V [m³]"); ax_V .set_xlabel("Time [h]"); ax_V .legend()

    col1.pyplot(fig_cP, use_container_width=True)
    col2.pyplot(fig_cL, use_container_width=True)
    col3.pyplot(fig_V , use_container_width=True)

    st.markdown("✅ Different constant-u strategies and their effect.")


# ──────────────────────────────────────────────────────────
def show_mpc():
    st.markdown("## Closed-loop MPC Simulation")

    # 1 ── Baseline MPC -----------------------------------------------------
    st.markdown("---");  st.markdown("### 1. Baseline MPC")
    N = st.slider("Prediction horizon N", 5, 50, 20)

    t_m, V_m, ML_m, u_m = simulate(spec_controller(N), Nominal(P))
    cP_m, cL_m = P.MP / V_m, ML_m / V_m
    plot_charts("Baseline MPC", t_m, cP_m, cL_m, u_m)
    st.info(f"⏱️ Batch time: **{batch_time(t_m, cP_m, cL_m):.2f} h**")

    # 2 ── Threshold policy -------------------------------------------------
    st.markdown("---");  st.markdown("### 2. Threshold Policy ($u=0.86$ if $c_P≥55$)")
    t_th, V_th, ML_th, u_th = simulate(threshold_policy(), Nominal(P))
    cP_th, cL_th = P.MP / V_th, ML_th / V_th
    for lab, t, cP, cL, u in [("MPC", t_m, cP_m, cL_m, u_m),
                              ("Threshold", t_th, cP_th, cL_th, u_th)]:
        plot_charts(lab, t, cP, cL, u)

    # 3 ── Time-optimal MPC -------------------------------------------------
    st.markdown("---");  st.markdown("### 3. Time-optimal MPC")
    N_opt = st.slider("Horizon (time-opt.)", 5, 50, 20, key="opt")
    t_o, V_o, ML_o, u_o = simulate(mpc_time_opt(N_opt), Nominal(P))
    cP_o, cL_o = P.MP / V_o, ML_o / V_o
    plot_charts("Time-optimal MPC", t_o, cP_o, cL_o, u_o)

    st.success(
        f"🏁 Time-optimal batch **{batch_time(t_o, cP_o, cL_o):.2f} h**  "
        f"vs Threshold **{batch_time(t_th, cP_th, cL_th):.2f} h**"
    )


# ──────────────────────────────────────────────────────────
def plot_charts(title, t, cP, cL, u):
    tol = 1e-3
    idx_end = np.where((cP >= P.cP_star - tol) & (cL <= P.cL_star + tol))[0]
    idx_end = idx_end[0] if idx_end.size else len(t) - 1

    t_p, cP_p, cL_p = t[:idx_end+1], cP[:idx_end+1], cL[:idx_end+1]
    u_p = u[: max(1, min(len(u), idx_end))]

    st.markdown(f"**{title}**")
    c1, c2, c3 = st.columns(3, gap="small")

    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(t_p/3600, cP_p); ax.axhline(P.cP_star, ls="--", color="k")
    ax.set_ylabel("$c_P$"); c1.pyplot(fig, use_container_width=True)

    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(t_p/3600, cL_p)
    ax.axhline(P.cL_star, ls="--", color="k"); ax.axhline(P.cL_max, ls=":", color="r")
    ax.set_ylabel("$c_L$"); c2.pyplot(fig, use_container_width=True)

    fig, ax = plt.subplots(figsize=(4,3))
    ax.step(t_p[:len(u_p)]/3600, u_p, where="post")
    ax.set_ylabel("$u$"); ax.set_xlabel("Time [h]")
    c3.pyplot(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────
def show_tests():
    st.markdown("## Test Scenarios")

    # 1 -- Tear disturbance -------------------------------------------------
    st.subheader("1. Filter-cake tear disturbance")
    t, V, ML, u = simulate(spec_controller(20), Tear(P))
    plot_charts("Tear Disturbance", t, P.MP / V, ML / V, u)

    # 2 -- Plant-model mismatch (robust) ------------------------------------
    st.subheader("2. Plant–Model Mismatch (robust MPC)")
    for factor in [0.75, 0.5, 0.25]:
        scen = KmMismatch(factor, P)
        t, V, ML, u = simulate(mpc_robust(20), scen)
        plot_charts(f"Mismatch factor = {factor}", t, P.MP / V, ML / V, u)

    # 3 -- Summary table ----------------------------------------------------
    st.subheader("3. Batch Time vs Mismatch")
    cols = st.columns(3)
    for i, factor in enumerate([0.75, 0.5, 0.25]):
        scen = KmMismatch(factor, P)
        t, V, ML, _ = simulate(mpc_robust(20), scen)
        cP, cL = P.MP / V, ML / V
        t_b = batch_time(t, cP, cL)
        peak = np.max(cL)
        ok = (cP[-1] >= P.cP_star) and (cL[-1] <= P.cL_star)
        msg = f"✅ {t_b:.2f} h  |  peak $c_L$ = {peak:.0f}" if ok else \
              f"❌ >6 h  |  peak $c_L$ = {peak:.0f}"
        cols[i].info(f"factor {factor}:  {msg}")
