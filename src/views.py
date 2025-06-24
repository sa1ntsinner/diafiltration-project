import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from constants import *
from simulator import simulate_open_loop
from simulator import closed_loop
from tests import disturbance_test, simulate_mismatch, batch_time_mismatch


def show_open_loop():
    st.markdown("## Open-loop Simulation")

    u_values = st.multiselect("Select constant u values", [0.1, 0.25, 0.5, 0.75, 1.0], default=[0.5])
    if not u_values:
        st.warning("Please select at least one u value.")
        return

    fig_cP, ax_cP = plt.subplots()
    fig_cL, ax_cL = plt.subplots()
    fig_V, ax_V = plt.subplots()

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
    ax_cP.legend(); st.pyplot(fig_cP)

    ax_cL.axhline(cL_star, ls='--', color='k', label="$c_L^*$")
    ax_cL.axhline(cL_max, ls=':', color='r', label="$c_L^{max}$")
    ax_cL.set_ylabel("$c_L$")
    ax_cL.legend(); st.pyplot(fig_cL)

    ax_V.set_ylabel("V [m³]"); ax_V.set_xlabel("Time [h]")
    ax_V.legend(); st.pyplot(fig_V)

    st.markdown("""
    ✅ These plots show how different constant control values affect the purification process.
    """)


def show_mpc():
    st.markdown("## Closed-loop MPC Simulation")

    N = st.slider("Prediction Horizon (N)", 5, 50, 20)
    t, V, ML, u = closed_loop(N=N)
    cP = MP / V
    cL = ML / V

    fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    ax[0].plot(t/3600, cP); ax[0].axhline(cP_star, ls='--', color='k'); ax[0].set_ylabel("$c_P$")
    ax[1].plot(t/3600, cL); ax[1].axhline(cL_star, ls='--', color='k'); ax[1].axhline(cL_max, ls=':', color='r'); ax[1].set_ylabel("$c_L$")
    ax[2].step(t[:len(u)]/3600, u, where='post'); ax[2].set_ylabel("$u$"); ax[2].set_xlabel("Time [h]")
    st.pyplot(fig)

    st.markdown("""
    ✅ MPC dynamically adjusts $u$ to meet the purification specs as efficiently as possible.
    """)


def show_tests():
    st.markdown("## Test Scenarios")

    st.subheader("1. Disturbance Test")
    t, V, ML, u = disturbance_test()
    cP = MP / V
    cL = ML / V

    fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    ax[0].plot(t/3600, cP); ax[0].axhline(cP_star, ls='--', color='k'); ax[0].set_ylabel("$c_P$")
    ax[1].plot(t/3600, cL); ax[1].axhline(cL_star, ls='--', color='k'); ax[1].axhline(cL_max, ls=':', color='r'); ax[1].set_ylabel("$c_L$")
    ax[2].step(t[:len(u)]/3600, u, where='post'); ax[2].set_ylabel("$u$"); ax[2].set_xlabel("Time [h]")
    st.pyplot(fig)

    st.subheader("2. Plant–Model Mismatch")
    factors = [0.75, 0.50, 0.25]
    for factor in factors:
        t, V, ML, u = simulate_mismatch(factor)
        cL = ML / V
        cP = MP / V

        st.markdown(f"**kM mismatch factor = {factor}**")
        fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
        ax[0].plot(t/3600, cP); ax[0].axhline(cP_star, ls='--', color='k'); ax[0].set_ylabel("$c_P$")
        ax[1].plot(t/3600, cL); ax[1].axhline(cL_star, ls='--', color='k'); ax[1].axhline(cL_max, ls=':', color='r'); ax[1].set_ylabel("$c_L$")
        ax[2].step(t[:len(u)]/3600, u, where='post'); ax[2].set_ylabel("$u$"); ax[2].set_xlabel("Time [h]")
        st.pyplot(fig)

    st.subheader("3. Batch Time and Peak $c_L$")
    st.markdown("Comparing how plant-model mismatch affects purification duration and lactose peak.")
    for f in factors:
        t_b, cL_pk, ok = batch_time_mismatch(f)
        if ok:
            st.success(f"factor={f:.2f}: ✅ batch {t_b:.2f} h, max $c_L$ = {cL_pk:.1f}")
        else:
            st.error(f"factor={f:.2f}: ❌ specs NOT met in {t_b:.2f} h, max $c_L$ = {cL_pk:.1f}")
