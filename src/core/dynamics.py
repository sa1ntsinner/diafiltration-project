"""
Continuous diafiltration model (2 states) + helper flux functions.

State vector
    x[0] = V   (m³)  – liquid volume
    x[1] = ML  (kg)  – mass of lactose in retentate

Input
    u in [0,1] – valve opening fraction for dilution water
"""
from __future__ import annotations
import numpy as np
import casadi as ca
from core.params import default as P

# ─────────────────────────────────── helper fluxes ────────────────────────── #
def flux_permeate(cP: float, P=P) -> float:
    """Permeate mass-flux p(cP)."""
    return P.k * P.A * np.log(P.cg / cP)


def lactose_permeate_conc(cL: float, p: float, P=P) -> float:
    """Lactose concentration in the permeate stream."""
    exp_term = np.exp(p / (P.kM_L * P.A))
    return P.alpha * cL / (1 + (P.alpha - 1) * exp_term)


# ───────────────────────────────  continuous RHS (NumPy)  ─────────────────── #
def rhs(state: np.ndarray, u: float, P=P) -> np.ndarray:
    """
    ẋ = f(x,u) implemented with NumPy – for fast open-loop simulation.
    """
    V, ML = state
    V = max(V, 1e-6)                      # avoid division-by-zero
    cP = P.MP / V
    p  = flux_permeate(cP, P)
    d  = u * p
    cL = ML / V
    cL_p = lactose_permeate_conc(cL, p, P)
    return np.asarray([d - p, -cL_p * p])


# ───────────────────────────────  continuous RHS (CasADi)  ────────────────── #
def casadi_rhs(P=P) -> ca.Function:
    """
    casadi.Function f(x,u) = ẋ  — used by MPC discretisers / collocation.
    """
    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u")

    V, ML = x[0], x[1]
    V_safe = ca.fmax(V, 1e-6)
    cP = P.MP / V_safe
    p  = P.k * P.A * ca.log(P.cg / cP)
    d  = u * p
    cL = ML / V_safe
    exp_term = ca.exp(p / (P.kM_L * P.A))
    cL_p = P.alpha * cL / (1 + (P.alpha - 1) * exp_term)
    dxdt = ca.vertcat(d - p, -cL_p * p)
    return ca.Function("rhs", [x, u], [dxdt])
