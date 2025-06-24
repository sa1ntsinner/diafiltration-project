import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from constants import *
from simulator import simulate_open_loop
from simulator import closed_loop
from tests import disturbance_test, simulate_mismatch, batch_time_mismatch

def show_open_loop():
    st.markdown("## Open-loop Simulation")

    num_u = st.slider("Number of control intervals", 1, 5, 1)
    u_values = [st.slider(f"u value {i+1}", 0.0, 1.0, 0.5, step=0.01, key=f"u_{i}") for i in range(num_u)]

    if not u_values:
        st.warning("Please select at least one u value.")
        return

    container = st.container()
    with container:
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

        with col1:
            st.pyplot(fig_cP, use_container_width=True)
        with col2:
            st.pyplot(fig_cL, use_container_width=True)
        with col3:
            st.pyplot(fig_V, use_container_width=True)

    st.markdown("✅ These plots show how different constant control values affect the purification process.")

def show_mpc():
    st.markdown("## Closed-loop MPC Simulation")

    N = st.slider("Prediction Horizon (N)", 5, 50, 20)
    t, V, ML, u = closed_loop(N=N)
    cP = MP / V
    cL = ML / V

    container = st.container()
    with container:
        col1, col2, col3 = st.columns(3)

        fig_cP, ax_cP = plt.subplots(figsize=(4, 3))
        fig_cL, ax_cL = plt.subplots(figsize=(4, 3))
        fig_u, ax_u = plt.subplots(figsize=(4, 3))

        ax_cP.plot(t / 3600, cP)
        ax_cP.axhline(cP_star, ls='--', color='k')
        ax_cP.set_ylabel("$c_P$")

        ax_cL.plot(t / 3600, cL)
        ax_cL.axhline(cL_star, ls='--', color='k')
        ax_cL.axhline(cL_max, ls=':', color='r')
        ax_cL.set_ylabel("$c_L$")

        ax_u.step(t[:len(u)] / 3600, u, where='post')
        ax_u.set_ylabel("$u$")
        ax_u.set_xlabel("Time [h]")

        with col1:
            st.pyplot(fig_cP, use_container_width=True)
        with col2:
            st.pyplot(fig_cL, use_container_width=True)
        with col3:
            st.pyplot(fig_u, use_container_width=True)

    st.markdown("""
    ✅ MPC dynamically adjusts $u$ to meet the purification specs as efficiently as possible.
    """)

def show_tests():
    st.markdown("## Test Scenarios")

    st.subheader("1. Disturbance Test")
    t, V, ML, u = disturbance_test()
    cP = MP / V
    cL = ML / V

    col1, col2, col3 = st.columns(3)
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.plot(t/3600, cP)
    ax1.axhline(cP_star, ls='--', color='k')
    ax1.set_ylabel("$c_P$")
    with col1:
        st.pyplot(fig1, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.plot(t/3600, cL)
    ax2.axhline(cL_star, ls='--', color='k')
    ax2.axhline(cL_max, ls=':', color='r')
    ax2.set_ylabel("$c_L$")
    with col2:
        st.pyplot(fig2, use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(4, 3))
    ax3.step(t[:len(u)]/3600, u, where='post')
    ax3.set_ylabel("$u$")
    ax3.set_xlabel("Time [h]")
    with col3:
        st.pyplot(fig3, use_container_width=True)

    st.subheader("2. Plant–Model Mismatch")
    factors = [0.75, 0.50, 0.25]
    for factor in factors:
        st.markdown(f"**kM mismatch factor = {factor}**")
        t, V, ML, u = simulate_mismatch(factor)
        cP = MP / V
        cL = ML / V

        col1, col2, col3 = st.columns(3)
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.plot(t/3600, cP)
        ax1.axhline(cP_star, ls='--', color='k')
        ax1.set_ylabel("$c_P$")
        with col1:
            st.pyplot(fig1, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(t/3600, cL)
        ax2.axhline(cL_star, ls='--', color='k')
        ax2.axhline(cL_max, ls=':', color='r')
        ax2.set_ylabel("$c_L$")
        with col2:
            st.pyplot(fig2, use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.step(t[:len(u)]/3600, u, where='post')
        ax3.set_ylabel("$u$")
        ax3.set_xlabel("Time [h]")
        with col3:
            st.pyplot(fig3, use_container_width=True)

    st.subheader("3. Batch Time and Peak $c_L$")
    st.markdown("Comparing how plant-model mismatch affects purification duration and lactose peak.")
    for f in factors:
        t_b, cL_pk, ok = batch_time_mismatch(f)
        if ok:
            st.success(f"factor={f:.2f}: ✅ batch {t_b:.2f} h, max $c_L$ = {cL_pk:.1f}")
        else:
            st.error(f"factor={f:.2f}: ❌ specs NOT met in {t_b:.2f} h, max $c_L$ = {cL_pk:.1f}")
