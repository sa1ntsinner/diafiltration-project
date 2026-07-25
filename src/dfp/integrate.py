"""
dfp.integrate
=============
Discretisation of the continuous model.  Every integrator has the *same*
signature ::

    F(x, u, theta, h) -> x_next

so the simulator, the MPC and the estimators are interchangeable.

Available schemes
-----------------
``rk4``
    Explicit Runge–Kutta 4 with ``M`` equidistant sub-steps inside one
    interval.  Fourth-order accurate, fully differentiable, cheap – the
    default for optimisation.
``irk``
    Implicit Gauss–Legendre / Radau collocation of arbitrary degree,
    formulated as *additional NLP variables and equations* (see
    :func:`collocation_stage`).  Stiff-stable and of order ``2d`` (Legendre);
    used to cross-check the RK4 grid.
``cvodes``
    CasADi's adaptive BDF/Adams wrapper – the accuracy reference used by the
    unit tests and by the high-fidelity plant.

Why this matters
----------------
The control interval is Δt = 600 s while the fastest time constant of the
batch is of the order of 10³ s.  Integrating the *plant* with a single RK4
step of 600 s produces batch-time errors of several minutes and lets the
protein concentration overshoot its specification by >100 %.  All plant
simulations therefore use ``n_sub_plant`` sub-steps (5 s by default).
"""

from __future__ import annotations

from typing import Tuple

import casadi as ca
import numpy as np

from .model import NTHETA, NU, NX

__all__ = ["rk4_integrator", "cvodes_integrator", "collocation_coefficients",
           "make_integrator"]


# ─────────────────────────────────────────────────────────────────────────────
def rk4_integrator(f: ca.Function, n_sub: int = 1) -> ca.Function:
    """Return ``F(x, u, theta, h)`` – RK4 with ``n_sub`` sub-steps per ``h``."""
    if n_sub < 1:
        raise ValueError("n_sub must be >= 1")
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    th = ca.SX.sym("theta", NTHETA)
    h = ca.SX.sym("h")
    dt = h / n_sub

    xk = x
    for _ in range(n_sub):
        k1 = f(xk, u, th)
        k2 = f(xk + 0.5 * dt * k1, u, th)
        k3 = f(xk + 0.5 * dt * k2, u, th)
        k4 = f(xk + dt * k3, u, th)
        xk = xk + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return ca.Function("F_rk4", [x, u, th, h], [xk],
                       ["x", "u", "theta", "h"], ["x_next"])


def cvodes_integrator(f: ca.Function, h: float, *, abstol: float = 1e-12,
                      reltol: float = 1e-12) -> ca.Function:
    """Adaptive CVODES step of *fixed* length ``h`` (accuracy reference)."""
    x = ca.MX.sym("x", NX)
    u = ca.MX.sym("u", NU)
    th = ca.MX.sym("theta", NTHETA)
    dae = {"x": x, "p": ca.vertcat(u, th), "ode": f(x, u, th)}
    opts = {"abstol": abstol, "reltol": reltol}
    try:                                    # CasADi >= 3.6 signature
        I = ca.integrator("I", "cvodes", dae, 0.0, h, opts)
    except (NotImplementedError, RuntimeError, TypeError):  # pragma: no cover
        opts["tf"] = h
        I = ca.integrator("I", "cvodes", dae, opts)
    xf = I(x0=x, p=ca.vertcat(u, th))["xf"]
    hs = ca.MX.sym("h_dummy")
    return ca.Function("F_cvodes", [x, u, th, hs], [xf],
                       ["x", "u", "theta", "h"], ["x_next"])


# ─────────────────────────────────────────────────────────────────────────────
def collocation_coefficients(d: int = 3, scheme: str = "radau"
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Butcher-style coefficients ``(C, D, tau)`` for IRK collocation.

    Returns
    -------
    C : (d+1, d) array
        Derivative of the Lagrange basis at the collocation points.
    D : (d+1,) array
        Basis evaluated at the end of the interval (continuity weights).
    tau : (d,) array
        Normalised collocation points.
    """
    tau = np.append(0.0, ca.collocation_points(d, scheme))
    C = np.zeros((d + 1, d + 1))
    D = np.zeros(d + 1)
    for j in range(d + 1):
        coeff = np.array([1.0])
        for r in range(d + 1):
            if r != j:
                coeff = np.convolve(coeff, np.array([1.0, -tau[r]]))
                coeff /= tau[j] - tau[r]
        D[j] = np.polyval(coeff, 1.0)
        dcoeff = np.polyder(coeff)
        for r in range(d + 1):
            C[j, r] = np.polyval(dcoeff, tau[r])
    return C, D, tau[1:]


def make_integrator(f: ca.Function, kind: str = "rk4", *, n_sub: int = 1,
                    h: float | None = None) -> ca.Function:
    """Dispatch helper: ``kind in {"rk4", "cvodes"}``."""
    kind = kind.lower()
    if kind == "rk4":
        return rk4_integrator(f, n_sub)
    if kind == "cvodes":
        if h is None:
            raise ValueError("cvodes needs a fixed step length h")
        return cvodes_integrator(f, h)
    raise ValueError(f"unknown integrator '{kind}'")
