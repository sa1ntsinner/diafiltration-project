"""
core/dynamics.py
────────────────
Continuous-time diafiltration model with 2 states:

    x[0] = V   → retentate volume      [m³]
    x[1] = ML  → mass of lactose       [kg]
    u    ∈ [0,1]  → control input: dilution valve opening

Also includes:
- Permeate flux model (based on protein concentration)
- Lactose concentration in the permeate
"""

from __future__ import annotations
import numpy as np
import casadi as ca
from core.params import default as P


# ════════════════════════════════════════════════════════════════════════════
# 1. Auxiliary flux functions (used by the dynamics)                          #
# ════════════════════════════════════════════════════════════════════════════

def flux_permeate(cP: float, P=P) -> float:
    """
    Permeate mass flux [kg/s] as a function of protein concentration.

    Parameters
    ----------
    cP : float
        Protein concentration in the retentate [kg/m³]

    Returns
    -------
    float
        Permeate mass flux
    """
    return P.k * P.A * np.log(P.cg / cP)


def lactose_permeate_conc(cL: float, p: float, P=P) -> float:
    """
    Lactose concentration in the permeate stream [kg/m³].

    Parameters
    ----------
    cL : float
        Lactose concentration in retentate [kg/m³]
    p : float
        Permeate flux [kg/s]

    Returns
    -------
    float
        Lactose concentration in permeate
    """
    exp_term = np.exp(p / (P.kM_L * P.A))
    return P.alpha * cL / (1 + (P.alpha - 1) * exp_term)


# ════════════════════════════════════════════════════════════════════════════
# 2. NumPy implementation of RHS for simulation                               #
# ════════════════════════════════════════════════════════════════════════════

def rhs(state: np.ndarray, u: float, P=P) -> np.ndarray:
    """
    Continuous-time RHS of the system:   ẋ = f(x, u)

    Parameters
    ----------
    state : ndarray
        Current state [V, ML]
    u : float
        Valve opening (0–1)

    Returns
    -------
    ndarray
        Derivative [dV/dt, dML/dt]
    """
    V, ML = state
    V = max(V, 1e-6)                      # avoid divide-by-zero errors

    cP = P.MP / V                         # protein concentration
    p  = flux_permeate(cP, P)            # total permeate flux
    d  = u * p                            # dilution inflow
    cL = ML / V                           # lactose concentration in retentate
    cL_p = lactose_permeate_conc(cL, p, P)  # lactose concentration in permeate

    return np.asarray([
        d - p,                           # dV/dt: net volume change
        -cL_p * p                        # dML/dt: lactose loss through membrane
    ])


# ════════════════════════════════════════════════════════════════════════════
# 3. CasADi version of RHS for use in MPC (symbolic)                          #
# ════════════════════════════════════════════════════════════════════════════

def casadi_rhs(P=P) -> ca.Function:
    """
    Create a CasADi function: f(x,u) = ẋ

    This is used by MPC and numeric integrators for symbolic modelling.

    Returns
    -------
    ca.Function
        CasADi function implementing ẋ = f(x,u)
    """
    x = ca.SX.sym("x", 2)     # [V, ML]
    u = ca.SX.sym("u")        # valve input

    V, ML = x[0], x[1]
    V_safe = ca.fmax(V, 1e-6)       # ensure stability

    cP = P.MP / V_safe              # protein concentration
    p  = P.k * P.A * ca.log(P.cg / cP)  # flux expression
    d  = u * p                      # dilution inflow

    cL = ML / V_safe
    exp_term = ca.exp(p / (P.kM_L * P.A))
    cL_p = P.alpha * cL / (1 + (P.alpha - 1) * exp_term)

    dxdt = ca.vertcat(
        d - p,                      # dV/dt
        -cL_p * p                   # dML/dt
    )

    return ca.Function("rhs", [x, u], [dxdt])
