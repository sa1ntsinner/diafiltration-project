import numpy as np, casadi as ca
from .constants import *

# ---------- NumPy (plant) ------------------------------------------------
def flux_permeate(cP: float) -> float:
    # Permeate flux (m³ s⁻¹).  Guard against log(≤0).
    ratio = max(cg / cP, 1e-9)           # always > 0
    return k * A * np.log(ratio)

def lactose_permeate_conc(cL: float, p: float) -> float:
    """
    Lactose concentration on permeate side (mol m⁻³).
    Clip exponent argument to avoid overflow: exp(±50) ≈ 5e21 ↔ 2e-22
    """
    arg = np.clip(p / (kM_L * A), -50.0, 50.0)
    exp_term = np.exp(arg)
    return alpha * cL / (1 + (alpha - 1) * exp_term)

def rhs(state: np.ndarray, u: float) -> np.ndarray:
    V, ML = state
    V = max(V, 1e-6)         # guard
    cP = MP / V
    p  = flux_permeate(cP)
    d  = u * p
    cL = ML / V
    cL_p = lactose_permeate_conc(cL, p)
    return np.array([d - p, -cL_p * p])

def rk4_step(state, u, dt=dt_ctrl):
    k1 = rhs(state,u)
    k2 = rhs(state+0.5*dt*k1,u)
    k3 = rhs(state+0.5*dt*k2,u)
    k4 = rhs(state+dt*k3,u)
    return state + dt/6*(k1+2*k2+2*k3+k4)

# ---------- CasADi symbolic (for MPC) ------------------------------------
def casadi_rhs():
    V, ML, u = ca.SX.sym("V"), ca.SX.sym("ML"), ca.SX.sym("u")
    cP  = MP / V
    ratio = ca.fmax(cg / cP, 1e-9)
    p   = k * A * ca.log(ratio)
    d   = u * p
    cL  = ML / V
    arg   = ca.fmin(ca.fmax(p/(kM_L*A), -50.0), 50.0)
    cL_p = alpha*cL / (1 + (alpha-1)*ca.exp(arg))
    return ca.Function("f", [ca.vertcat(V,ML), u], [ca.vertcat(d-p, -cL_p*p)])
