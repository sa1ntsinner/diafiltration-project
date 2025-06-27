"""
Shared time-discretisation helpers (RK4 for NumPy and CasADi).
"""
from typing import Callable
import numpy as np
import casadi as ca

# ───────────────────────── CasADi RK4 maps (for MPC) ──────────────────────── #
def rk4_disc(rhs_fun: ca.Function, dt: float) -> ca.Function:
    """
    Build a CasADi RK4 integrator: x⁺ = F(x,u)  with step dt.
    """
    x = ca.SX.sym("x", rhs_fun.size1_in(0))
    u = ca.SX.sym("u", rhs_fun.size1_in(1))
    k1 = rhs_fun(x, u)
    k2 = rhs_fun(x + 0.5 * dt * k1, u)
    k3 = rhs_fun(x + 0.5 * dt * k2, u)
    k4 = rhs_fun(x +       dt * k3, u)
    x_next = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    return ca.Function("F", [x, u], [x_next])


# ─────────────────────────  NumPy RK4 step (for simulation) ───────────────── #
def rk4_step(state: np.ndarray,
             u: float,
             dt: float,
             rhs: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
    """
    One explicit RK4 step for pure-NumPy simulation loops.
    """
    k1 = rhs(state,                 u)
    k2 = rhs(state + 0.5 * dt * k1, u)
    k3 = rhs(state + 0.5 * dt * k2, u)
    k4 = rhs(state +       dt * k3, u)
    return state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
