"""
Open-loop plant model: NumPy and CasADi versions + RK4.
"""

import numpy as np
import casadi as ca
from .constants import *

# ─── helpers (NumPy) ─────────────────────────────────────────────────────
def flux_permeate(cP: float) -> float:
    return k * A * np.log(cg / cP)

def lactose_permeate_conc(cL: float, p: float) -> float:
    expo = p / (kM_L * A)
    return alpha * cL / (1 + (alpha - 1) * np.exp(expo))

def rhs(state: np.ndarray, u: float) -> np.ndarray:
    V, ML = state
    cP  = MP / V
    p   = flux_permeate(cP)
    d   = u * p
    cL  = ML / V
    cL_p = lactose_permeate_conc(cL, p)
    return np.array([d - p, -cL_p * p])

def rk4_step(state, u, dt=dt_ctrl):
    k1 = rhs(state,u)
    k2 = rhs(state+0.5*dt*k1,u)
    k3 = rhs(state+0.5*dt*k2,u)
    k4 = rhs(state+dt*k3,u)
    return state + dt/6*(k1+2*k2+2*k3+k4)

# ─── CasADi symbolic RHS (for MPC) ───────────────────────────────────────
def casadi_rhs():
    V, ML, u = ca.SX.sym("V"), ca.SX.sym("ML"), ca.SX.sym("u")
    cP  = MP / V
    p   = k * A * ca.log(cg / cP)
    d   = u * p
    cL  = ML / V
    exp = ca.exp(p / (kM_L * A))
    cL_p = alpha * cL / (1 + (alpha - 1) * exp)
    return ca.Function("f", [ca.vertcat(V,ML), u],
                             [ca.vertcat(d - p, -cL_p * p)])
