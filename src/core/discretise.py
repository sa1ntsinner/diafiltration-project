"""
core/discretise.py
──────────────────
Utility functions that turn a *continuous* RHS  ẋ = f(x,u)
into explicit Runge–Kutta-4 (RK4) *discrete-time* maps.

Two flavours are provided

1. rk4_disc  – returns a **CasADi** Function  F(x,u) = x⁺  (used inside MPC)
2. rk4_step  – performs one **NumPy** RK4 step          (used for simulation)
"""

from __future__ import annotations
from typing import Callable

import casadi as ca
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# 1.  CasADi RK4 map  (symbolic – for MPC optimiser)                          #
# ════════════════════════════════════════════════════════════════════════════
def rk4_disc(rhs_fun: ca.Function, dt: float) -> ca.Function:
    """
    Create a *symbolic* single-step RK4 integrator

        x⁺ = F(x,u)      with sample time  dt  [s]

    Parameters
    ----------
    rhs_fun : casadi.Function
        Symbolic function f(x,u) that returns the continuous-time state
        derivative  ẋ.
    dt : float
        Integration step size (sampling period).

    Returns
    -------
    F : casadi.Function
        Discrete-time map suitable for use inside an MPC.
    """
    # symbolic state and input
    x = ca.SX.sym("x", rhs_fun.size1_in(0))
    u = ca.SX.sym("u", rhs_fun.size1_in(1))

    # classical four RK stages
    k1 = rhs_fun(x,                     u)
    k2 = rhs_fun(x + 0.5 * dt * k1,     u)
    k3 = rhs_fun(x + 0.5 * dt * k2,     u)
    k4 = rhs_fun(x +       dt * k3,     u)

    # weighted average → next state
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return ca.Function("F", [x, u], [x_next])


# ════════════════════════════════════════════════════════════════════════════
# 2.  NumPy RK4 step  (numeric – for open-loop simulation)                    #
# ════════════════════════════════════════════════════════════════════════════
def rk4_step(
    state: np.ndarray,
    u: float,
    dt: float,
    rhs: Callable[[np.ndarray, float], np.ndarray],
) -> np.ndarray:
    """
    Perform **one** explicit RK4 step for a NumPy simulation loop.

    Parameters
    ----------
    state : ndarray
        Current state  x  (shape (n,)).
    u : float
        Current control input  u  (scalar here, but could be ndarray).
    dt : float
        Integration step size.
    rhs : callable
        Python function implementing  ẋ = f(x,u).

    Returns
    -------
    ndarray
        Next state  x⁺  after a single RK4 step.
    """
    k1 = rhs(state,                 u)
    k2 = rhs(state + 0.5 * dt * k1, u)
    k3 = rhs(state + 0.5 * dt * k2, u)
    k4 = rhs(state +       dt * k3, u)

    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
