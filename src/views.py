import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from constants import *
from simulator import simulate_open_loop, closed_loop, closed_loop_threshold
from simulator_time_optimal import closed_loop_time_optimal
from tests import disturbance_test, simulate_mismatch, batch_time_mismatch


def show_open_loop():
    st.markdown("## Open-loop Simulation")

    num_u = st.slider("Number of control intervals", 1, 5, 1)
    u_values = [st.slider(f"u value {i+1}", 0.0, 1.0, 0.5, step=0.01, key=f"u_{i}") for i in range(num_u)]

    if not u_values:
        st.warning("Please select at least one u value.")
        return

    col1, col2, col3 = st.columns(3)
    fig_cP, ax_cP = plt.subplots(figsize=(4, 3))
    fig_cL, ax_cL = plt.subplots(figsize=(4, 3))
    fig_V, ax_V = plt.subplots(figsize=(4, 3))

    for u in u_values:
        t, V, ML = simulate_open_loop(u)
        cP = MP / V
        cL = ML / V
        label = f"u = {u:.2f}"
        ax_cP.plot(t / 3600, cP, label=label)
        ax_cL.plot(t / 3600, cL, label=label)
        ax_V.plot(t / 3600, V, label=label)

    ax_cP.axhline(cP_star, ls='--', color='k', label="$c_P^*$")
    ax_cP.set_ylabel("$c_P$")
    ax_cP.legend()

    ax_cL.axhline(cL_star, ls='--', color='k', label="$c_L^*$")
    ax_cL.axhline(cL_max, ls=':', color='r', label="$c_L^{max}$")
    ax_cL.set_ylabel("$c_L$")
    ax_cL.legend()

    ax_V.set_ylabel("V [m³]")
    ax_V.set_xlabel("Time [h]")
    ax_V.legend()

    col1.pyplot(fig_cP, use_container_width=True)
    col2.pyplot(fig_cL, use_container_width=True)
    col3.pyplot(fig_V, use_container_width=True)

    st.markdown("✅ These plots show how different constant control values affect the purification process.")


def show_mpc():
    st.markdown("## Closed-loop MPC Simulation")

    st.markdown("---")
    st.markdown("### 1. Baseline MPC")
    st.markdown("This simulation uses a classic MPC with adjustable prediction horizon $N$ to achieve the target concentrations.")

    N = st.slider("Prediction Horizon (N)", 5, 50, 20)

    t_mpc, V_mpc, ML_mpc, u_mpc = closed_loop(N=N)
    cP_mpc = MP / V_mpc
    cL_mpc = ML_mpc / V_mpc

    plot_charts("Baseline MPC", t_mpc, cP_mpc, cL_mpc, u_mpc)

    st.markdown("---")
    st.markdown("### 2. Comparison with Threshold Policy")
    st.markdown("This policy applies $u=0.86$ if $c_P \\geq 55$, otherwise $u=0$. It is a simpler but less flexible strategy.")

    t_thr, V_thr, ML_thr, u_thr = closed_loop_threshold()
    cP_thr = MP / V_thr
    cL_thr = ML_thr / V_thr

    for label, t, cP, cL, u in [
        ("MPC", t_mpc, cP_mpc, cL_mpc, u_mpc),
        ("Threshold Policy", t_thr, cP_thr, cL_thr, u_thr)
    ]:
        plot_charts(label, t, cP, cL, u)

    st.markdown("""
    ✅ **MPC** dynamically optimizes $u$ to meet specs faster.  
    ❌ **Threshold Policy** is simpler but may be slower or violate constraints.
    """)

    st.markdown("---")
    st.markdown("### 3. Time-optimal MPC")
    st.markdown("""
    This controller explicitly encourages high $u$ values to minimize total batch duration while respecting constraints.
    The objective function is designed to penalize **low $u$**, indirectly pushing the system toward faster completion.
    """)

    N_opt = st.slider("Prediction Horizon for Time-Optimal MPC", 5, 50, 20, key="opt_N")
    t_opt, V_opt, ML_opt, u_opt = closed_loop_time_optimal(N_opt)
    cP_opt = MP / V_opt
    cL_opt = ML_opt / V_opt

    plot_charts("Time-optimal MPC", t_opt, cP_opt, cL_opt, u_opt)

    st.markdown("""
    ✅ The time-optimal MPC increases $u$ when it is safe, reducing total purification time.  
    ⏱️ Try increasing N to see faster convergence (with more aggressive control).
    """)


def plot_charts(title, t, cP, cL, u):
    st.markdown(f"**{title}**")
    col1, col2, col3 = st.columns(3)

    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.plot(t / 3600, cP)
    ax1.axhline(cP_star, ls='--', color='k')
    ax1.set_ylabel("$c_P$")
    col1.pyplot(fig1, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.plot(t / 3600, cL)
    ax2.axhline(cL_star, ls='--', color='k')
    ax2.axhline(cL_max, ls=':', color='r')
    ax2.set_ylabel("$c_L$")
    col2.pyplot(fig2, use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(4, 3))
    ax3.step(t[:len(u)] / 3600, u, where='post')
    ax3.set_ylabel("$u$")
    ax3.set_xlabel("Time [h]")
    col3.pyplot(fig3, use_container_width=True)


def show_tests():
    st.markdown("## Test Scenarios")

    st.subheader("1. Disturbance Test")
    t, V, ML, u = disturbance_test()
    plot_charts("Disturbance", t, MP / V, ML / V, u)

    st.subheader("2. Plant–Model Mismatch")
    for factor in [0.75, 0.5, 0.25]:
        st.markdown(f"**kM mismatch factor = {factor}**")
        t, V, ML, u = simulate_mismatch(factor)
        plot_charts(f"Mismatch factor = {factor}", t, MP / V, ML / V, u)

    st.subheader("3. Batch Time and Peak $c_L$")
    for factor in [0.75, 0.5, 0.25]:
        t_b, cL_pk, ok = batch_time_mismatch(factor)
        if ok:
            st.success(f"factor={factor:.2f}: ✅ batch = {t_b:.2f} h, peak $c_L$ = {cL_pk:.1f}")
        else:
            st.error(f"factor={factor:.2f}: ❌ specs NOT met in {t_b:.2f} h, peak $c_L$ = {cL_pk:.1f}")