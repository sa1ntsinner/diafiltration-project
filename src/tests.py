import numpy as np
from constants import *
from mpc import build_mpc
from model import rk4_step, flux_permeate, lactose_permeate_conc

# ---------- 1. Disturbance Test (step disturbance after 2h) ----------
def disturbance_test(N=20):
    solver, meta, LBG, UBG = build_mpc(N)
    state = np.array([V0, ML0])
    t, V, ML, u_hist = [], [], [], []
    step = 0
    disturbance_step = int(2 * 3600 / dt_ctrl)

    while True:
        t.append(step * dt_ctrl)
        V.append(state[0])
        ML.append(state[1])

        cP = MP / state[0]
        cL = state[1] / state[0]

        if cP >= cP_star and cL <= cL_star:
            break

        x0 = np.hstack([np.tile(state, meta["N"] + 1), 0.5 * np.ones(meta["N"])])
        sol = solver(x0=x0, p=state, lbg=LBG, ubg=UBG)
        u = float(np.clip(sol["x"].full().ravel()[meta["Uslice"]][0], 0, 1))

        u_hist.append(u)

        if step == disturbance_step:
            state[1] += 0.2 * ML0  # 20% lactose load

        state = rk4_step(state, u)
        step += 1

    return np.array(t), np.array(V), np.array(ML), np.array(u_hist)


# ---------- 2. Plant–Model Mismatch ----------
def rhs_km_mismatch(state, u, factor):
    V, ML = state
    V = max(V, 1e-6)
    cP = MP / V
    p = flux_permeate(cP)
    d = u * p
    cL = ML / V
    exp_term = np.exp(p / (factor * kM_L * A))
    cL_p = alpha * cL / (1 + (alpha - 1) * exp_term)
    return np.array([d - p, -cL_p * p])

def rk4_mis(state, u, factor, dt=dt_ctrl):
    f = lambda s: rhs_km_mismatch(s, u, factor)
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

def simulate_mismatch(factor, N=20):
    solver, meta, LBG, UBG = build_mpc(N)
    state = np.array([V0, ML0])
    t, V, ML, u_hist = [], [], [], []
    step = 0

    while True:
        t.append(step * dt_ctrl)
        V.append(state[0])
        ML.append(state[1])

        cP = MP / state[0]
        cL = state[1] / state[0]

        if cP >= cP_star and cL <= cL_star:
            break

        x0 = np.hstack([np.tile(state, meta["N"] + 1), 0.5 * np.ones(meta["N"])])
        sol = solver(x0=x0, p=state, lbg=LBG, ubg=UBG)
        u = float(np.clip(sol["x"].full().ravel()[meta["Uslice"]][0], 0, 1))

        u_hist.append(u)
        state = rk4_mis(state, u, factor)
        step += 1

        if step > 400:
            break

    return np.array(t), np.array(V), np.array(ML), np.array(u_hist)


# ---------- 3. Batch Time and Peak cL with Mismatch ----------
def batch_time_mismatch(factor, N=20, max_steps=400):
    solver, meta, LBG, UBG = build_mpc(N)
    state = np.array([V0, ML0])
    peak_cL = 0.0

    for step in range(max_steps):
        cP_now = MP / state[0]
        cL_now = state[1] / state[0]
        peak_cL = max(peak_cL, cL_now)

        if cP_now >= cP_star and cL_now <= cL_star:
            t_batch = step * dt_ctrl / 3600
            return t_batch, peak_cL, True

        x0 = np.hstack([np.tile(state, meta["N"] + 1), 0.5 * np.ones(meta["N"])])
        sol = solver(x0=x0, p=state, lbg=LBG, ubg=UBG)
        u = float(np.clip(sol["x"].full().ravel()[meta["Uslice"]][0], 0, 1))
        state = rk4_mis(state, u, factor)

    t_batch = max_steps * dt_ctrl / 3600
    return t_batch, peak_cL, False
