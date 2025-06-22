"""
Robustness scenarios:
  1. parametric mismatch  (kM_L,true = factor · kM_L)
  2. protein leakage      (β, kM_P)
Used in scripts/run_disturbances.py
"""
import numpy as np
from .constants  import *
from .model      import rk4_step, flux_permeate, lactose_permeate_conc
from .mpc        import build_mpc

# ------------------------------------------------------------------ 1) kM_L mismatch
def rhs_km_mismatch(state, u, factor):
    V, ML = state
    Vsafe = max(V, 1e-6)
    cP = MP / Vsafe
    p  = flux_permeate(cP)
    d  = u * p
    cL = ML / Vsafe
    exp_term = np.exp(p / (factor * kM_L * A))
    cL_p = alpha * cL / (1 + (alpha-1) * exp_term)
    return np.array([d - p, -cL_p * p])

def rk4_mismatch(state, u, factor, dt=dt_ctrl):
    f = lambda s: rhs_km_mismatch(s, u, factor)
    k1 = f(state)
    k2 = f(state+0.5*dt*k1)
    k3 = f(state+0.5*dt*k2)
    k4 = f(state+dt*k3)
    return state + dt/6*(k1+2*k2+2*k3+k4)

def run_mismatch(factor, N=20, max_steps=400):
    solver, meta, LBG, UBG = build_mpc(N)
    state = np.array([V0, ML0])
    step = 0; peak_cL = 0
    while step < max_steps:
        cP = MP / state[0]; cL = state[1]/state[0]
        peak_cL = max(peak_cL, cL)
        if cP >= cP_star and cL <= cL_star:
            return step*dt_ctrl/3600, peak_cL, True
        x0 = np.hstack([np.tile(state, meta["N"]+1),
                        0.5*np.ones(meta["N"])])
        sol = solver(x0=x0, p=state, lbg=LBG, ubg=UBG)
        u_now = float(np.clip(sol["x"].full().ravel()
                              [meta["Uslice"]][0],0,1))
        state = rk4_mismatch(state, u_now, factor)
        step += 1
    return step*dt_ctrl/3600, peak_cL, False

# ------------------------------------------------------------------ 2) protein leakage
beta      = 1.3
kM_P_true = 1.0e-6

def rhs_leak(x, u):
    V, ML, MP_dyn = x
    Vsafe = max(V, 1e-6)
    cP = MP_dyn / Vsafe
    p  = flux_permeate(cP)
    d  = u * p
    cL = ML / Vsafe

    exp_L = np.exp(p / (kM_L * A))
    cL_p  = alpha * cL / (1 + (alpha-1) * exp_L)

    cP_p = beta * cP / (1 + (beta-1) * np.exp(p/(kM_P_true*A)))
    return np.array([d - p, -cL_p*p, -cP_p*p])

def rk4_leak(x, u, dt=dt_ctrl):
    k1 = rhs_leak(x,u)
    k2 = rhs_leak(x+0.5*dt*k1,u)
    k3 = rhs_leak(x+0.5*dt*k2,u)
    k4 = rhs_leak(x+dt*k3,u)
    return x + dt/6*(k1+2*k2+2*k3+k4)

def run_leak(N=20, max_steps=400):
    solver, meta, LBG, UBG = build_mpc(N)
    state = np.array([V0, ML0, MP])      # note: dynamic MP
    step = 0
    while step < max_steps:
        V, ML, MP_dyn = state
        cP = MP_dyn / V; cL = ML / V
        if cP >= cP_star and cL <= cL_star:
            return step*dt_ctrl/3600, cP, cL, True
        x0 = np.hstack([np.tile(state[:2], meta["N"]+1),
                        0.5*np.ones(meta["N"])])
        sol = solver(x0=x0, p=state[:2], lbg=LBG, ubg=UBG)
        u_now = float(np.clip(sol["x"].full().ravel()
                              [meta["Uslice"]][0],0,1))
        state = rk4_leak(state, u_now)
        step += 1
    return step*dt_ctrl/3600, cP, cL, False
