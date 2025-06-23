"""
Optional robustness helpers:
  • run_mismatch()  – parametric kM,L,true < kM,L
  • apply_leakage() – structural mismatch (protein leakage)
"""
import numpy as np
from .constants import *
from .model     import rk4_step, flux_permeate, lactose_permeate_conc

# ----------------------------------------------------------------------
def run_mismatch(factor: float, u=1.0, hrs_limit=24):
    """Open-loop batch with kM_L,true = factor · kM_L."""
    kM_L_true = factor * kM_L
    state = np.array([V0, ML0]); t = 0
    while t < hrs_limit*3600:
        V, ML = state
        cP, p = MP/V, flux_permeate(MP/V)
        arg = np.clip(p/(kM_L_true*A), -50, 50)
        cL_p = alpha*(ML/V)/(1+(alpha-1)*np.exp(arg))
        rhs = np.array([u*p - p, -cL_p*p])
        state += dt_ctrl*rhs
        t += dt_ctrl
        if MP/state[0] >= cP_star and (state[1]/state[0]) <= cL_star:
            return t/3600
    return np.inf

# ----------------------------------------------------------------------
def apply_leakage(state, u):
    """One RK4 step with protein leakage (β=1.3, kM,P=1e-6)."""
    V, ML = state; cP = MP/V
    p = flux_permeate(cP); d = u*p; cL = ML/V
    β, kM_P = 1.3, 1e-6
    argP = np.clip(p/(kM_P*A), -50, 50)
    cP_p = β*cP / (1+(β-1)*np.exp(argP))
    dMP  = cP_p*p                      # lost protein (optional use)
    argL = np.clip(p/(kM_L*A), -50, 50)
    cL_p = alpha*cL/(1+(alpha-1)*np.exp(argL))
    rhs = np.array([d - p, -cL_p*p])
    return rk4_step(state, u), dMP
