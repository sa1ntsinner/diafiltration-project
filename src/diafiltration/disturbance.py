import numpy as np
from .constants  import *
from .model      import flux_permeate, lactose_permeate_conc
from .mpc        import build_mpc
from .simulator  import rk4_step          # reuse NumPy RK4

# ---------- param-mismatch lactose kM_L ---------------------------------
def rhs_km_mismatch(state: np.ndarray, u: float, factor: float) -> np.ndarray:
    V, ML = state
    cP = MP / V
    p  = flux_permeate(cP)
    d  = u * p
    cL = ML / V
    exp = np.exp(p / (factor * kM_L * A))
    cL_p = alpha * cL / (1 + (alpha - 1) * exp)
    return np.array([d - p, -cL_p * p])

def rk4_mismatch(state, u, factor, dt=dt_ctrl):
    k1 = rhs_km_mismatch(state, u, factor)
    k2 = rhs_km_mismatch(state+0.5*dt*k1, u, factor)
    k3 = rhs_km_mismatch(state+0.5*dt*k2, u, factor)
    k4 = rhs_km_mismatch(state+dt*k3, u, factor)
    return state + dt/6*(k1+2*k2+2*k3+k4)

# ---------- structural mismatch – protein leakage -----------------------
beta      = 1.3
kM_P_true = 1.0e-6

def rhs_leak(state: np.ndarray, u: float) -> np.ndarray:
    V, ML, MP_dyn = state
    cP = MP_dyn / V
    p  = flux_permeate(cP)
    d  = u * p
    cL = ML / V
    exp_L = np.exp(p / (kM_L * A))
    cL_p  = alpha * cL / (1 + (alpha - 1) * exp_L)
    cP_p  = beta  * cP / (1 + (beta  - 1) * np.exp(p/(kM_P_true*A)))
    return np.array([d - p,
                     -cL_p * p,
                     -cP_p * p])

def rk4_leak(state, u, dt=dt_ctrl):
    k1 = rhs_leak(state, u)
    k2 = rhs_leak(state+0.5*dt*k1, u)
    k3 = rhs_leak(state+0.5*dt*k2, u)
    k4 = rhs_leak(state+dt*k3, u)
    return state + dt/6*(k1+2*k2+2*k3+k4)
