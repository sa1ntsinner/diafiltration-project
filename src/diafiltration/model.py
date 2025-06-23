import numpy as np, casadi as ca
from .constants import *

# ───────── NumPy (plant) ────────────────────────────────────────────────
def flux_permeate(cP: float) -> float:
    """Permeate flux p  [m³ s⁻¹]; negative values are clipped to zero."""
    ratio = max(cg / cP, 1e-9)
    p = k * A * np.log(ratio)
    return max(p, 0.0)

def lactose_permeate_conc(cL: float, p: float) -> float:
    """Lactose concentration in permeate  [mol m⁻³] (eq. 2)."""
    arg = np.clip(p / (kM_L * A), -50.0, 50.0)          # avoid overflow
    return alpha * cL / (1 + (alpha - 1) * np.exp(arg))

def rhs(state: np.ndarray, u: float) -> np.ndarray:
    """Continuous ODE right-hand side."""
    V, ML = state
    V  = max(V, 1e-6)
    cP = MP / V
    p  = flux_permeate(cP)
    d  = u * p
    cL = ML / V
    cL_p = lactose_permeate_conc(cL, p)
    return np.array([d - p, -cL_p * p])

def rk4_step(state, u, dt=dt_ctrl):
    """Single RK4 step of length 10 min (matches notebook)."""
    k1 = rhs(state, u)
    k2 = rhs(state + 0.5*dt*k1, u)
    k3 = rhs(state + 0.5*dt*k2, u)
    k4 = rhs(state +     dt*k3, u)
    out = state + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    out[0] = max(out[0], 1e-6)   # V ≥ 0
    return out

# ───────── CasADi symbolic (for MPC) ────────────────────────────────────
def casadi_rhs():
    V, ML, u = ca.SX.sym("V"), ca.SX.sym("ML"), ca.SX.sym("u")

    Vsafe = ca.fmax(V, 1e-6)
    cP = MP / Vsafe
    ratio = ca.fmax(cg / cP, 1e-9)
    p_raw = k * A * ca.log(ratio)
    p = ca.fmax(p_raw, 0.0)

    d  = u * p
    cL = ML / Vsafe
    arg = ca.fmin(ca.fmax(p/(kM_L*A), -50.0), 50.0)
    cL_p = alpha * cL / (1 + (alpha-1) * ca.exp(arg))

    return ca.Function("f",
                       [ca.vertcat(V, ML), u],
                       [ca.vertcat(d - p,
                                   -cL_p * p)])
