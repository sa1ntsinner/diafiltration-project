import numpy as np
from constants import *

def flux_permeate(cP):
    return k * A * np.log(cg / cP)

def lactose_permeate_conc(cL, p):
    exp_term = np.exp(p / (kM_L * A))
    return alpha * cL / (1 + (alpha - 1) * exp_term)

def rhs(state, u):
    V, ML = state
    V = max(V, 1e-6)
    cP = MP / V
    p = flux_permeate(cP)
    d = u * p
    cL = ML / V
    cL_p = lactose_permeate_conc(cL, p)
    return np.array([d - p, -cL_p * p])

def rk4_step(state, u, dt=dt_ctrl):
    k1 = rhs(state, u)
    k2 = rhs(state + 0.5 * dt * k1, u)
    k3 = rhs(state + 0.5 * dt * k2, u)
    k4 = rhs(state + dt * k3, u)
    return state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
