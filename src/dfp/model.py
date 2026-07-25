"""
dfp.model
=========
The diafiltration process model – **one** symbolic implementation that is
shared by the simulator, every MPC and the state/parameter estimators.

Derivation
----------
Total mass balances over the well-mixed feed tank (the recirculation loop is
fast compared with the batch, hence lumped):

.. math::

   \\dot V   &= d(t) - p(t) = \\bigl(u(t) - 1\\bigr)\\,p(t) \\\\
   \\dot M_L &= -c_{L,p}\\,p = -r_L\\,c_L\\,p \\\\
   \\dot M_P &= -c_{P,p}\\,p = -r_P\\,c_P\\,p

with the algebraic relations of the task sheet

.. math::

   c_P = M_P/V,\\qquad c_L = M_L/V,\\qquad
   p   = k A \\ln\\!\\bigl(c_g/c_P\\bigr),

.. math::

   r_L = \\frac{c_{L,p}}{c_L}
       = \\frac{\\alpha}{1+(\\alpha-1)\\exp\\!\\bigl(p/(k_{M,L}A)\\bigr)},
   \\qquad
   r_P = \\frac{\\beta}{1+(\\beta-1)\\exp\\!\\bigl(p/(k_{M,P}A)\\bigr)} .

Choice of states
----------------
``x = [V, M_L, M_P]``.  Using *hold-ups* instead of concentrations keeps the
balances in conservation form (no division inside the derivative), so the
integrator conserves mass to machine precision.  Concentrations are recovered
as outputs.  For the nominal plant the membrane retains protein completely,
``\\dot M_P = 0``, and the model collapses to the two states ``[V, M_L]``;
``M_P`` is kept in the state vector so that the *structural* mismatch of
Eq. (6) needs no separate model.

Non-linearity
-------------
The model is non-linear in both state and input: ``ln(c_g V/M_P)`` and
``exp(p/(k_M A))`` are transcendental, and the inflow ``d = u·p(V)`` is a
*product* of input and state (bilinear).  No change of coordinates removes
this, hence non-linear MPC is required.

Exact reachability property
---------------------------
Because ``u ≤ 1`` we have ``\\dot V ≤ 0``: the volume is monotonically
non-increasing and therefore ``c_P`` is monotonically non-decreasing (nominal
plant).  Since the terminal specification is the *equality* ``c_P = c_{P,f}``,
the protein concentration can never overshoot – the reachable set is a
one-parameter family in ``c_P``.  :mod:`dfp.controllers.ocp` exploits this to
derive the analytic time-optimal solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import casadi as ca
import numpy as np

from .config import NOMINAL, ProcessParams

__all__ = ["DiafiltrationModel", "build_model", "NX", "NU", "NTHETA", "THETA_NAMES"]

NX: int = 3       #: number of states  [V, ML, MP]
NU: int = 1       #: number of inputs  [u]
NTHETA: int = 7   #: number of symbolic parameters
THETA_NAMES: Tuple[str, ...] = ("k", "A", "cg", "kM_L", "alpha", "beta", "kM_P")

_V_FLOOR = 1e-9   # numerical guard, never active on physically valid runs


# ─────────────────────────────────────────────────────────────────────────────
def _flux(cP: ca.SX, k: ca.SX, A: ca.SX, cg: ca.SX) -> ca.SX:
    """Permeate volumetric flow, Eq. (1)."""
    return k * A * ca.log(cg / cP)


def _partition(gamma: ca.SX, p: ca.SX, kM: ca.SX, A: ca.SX) -> ca.SX:
    """Permeate/retentate concentration ratio, Eqs. (2) and (6).

    For ``gamma >= 1`` and ``p >= 0`` the result is confined to ``(0, 1]``,
    i.e. the membrane can never enrich the permeate above the retentate.
    """
    return gamma / (1.0 + (gamma - 1.0) * ca.exp(p / (kM * A)))


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DiafiltrationModel:
    """Bundle of CasADi functions describing one *variant* of the plant.

    Attributes
    ----------
    f
        ``f(x, u, theta) -> dx/dt`` – continuous-time right-hand side.
    out
        ``out(x, theta) -> [cP, cL, p, rL]`` – algebraic outputs.
    protein_leakage
        Whether Eq. (6) is active (structural mismatch scenario).
    tear
        ``(cP_lo, cP_hi, factor)`` of an active filter-cake tear, or ``None``.
    """

    f: ca.Function
    out: ca.Function
    protein_leakage: bool = False
    tear: Optional[Tuple[float, float, float]] = None
    params: ProcessParams = NOMINAL

    # ── convenience numeric wrappers ────────────────────────────────────────
    def rhs(self, x, u, theta=None) -> np.ndarray:
        theta = self.params.theta if theta is None else theta
        return np.asarray(self.f(x, u, theta)).ravel()

    def outputs(self, x, theta=None) -> np.ndarray:
        """Return ``[cP, cL, p, rL]`` for a single state or a state *trajectory*."""
        theta = self.params.theta if theta is None else theta
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[0] != NX:
            x = x.T
        return np.asarray(self.out(x, theta))

    def cP(self, x, theta=None) -> np.ndarray:
        return self.outputs(x, theta)[0]

    def cL(self, x, theta=None) -> np.ndarray:
        return self.outputs(x, theta)[1]


# ─────────────────────────────────────────────────────────────────────────────
def build_model(
    *,
    protein_leakage: bool = False,
    tear: Optional[Tuple[float, float, float]] = None,
    params: ProcessParams = NOMINAL,
) -> DiafiltrationModel:
    """Assemble the symbolic model.

    Parameters
    ----------
    protein_leakage
        If ``True`` the protein balance uses Eq. (6) (β, ``kM_P`` taken from
        ``theta``); otherwise protein is retained completely (``dMP/dt = 0``),
        which is the nominal assumption of the task sheet.
    tear
        ``(cP_lo, cP_hi, factor)``.  Inside the concentration window the
        permeate flow is multiplied by ``factor`` – the filter-cake tear of
        Eq. (5).  The switch is *exact* (not smoothed); the plant integrator
        resolves it with fine sub-steps.  Never use a torn model inside an
        optimiser.
    params
        Only used to fill :attr:`DiafiltrationModel.params` (the default
        numeric ``theta``); the returned functions stay fully parametric.
    """
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    th = ca.SX.sym("theta", NTHETA)
    k, A, cg, kM_L, alpha, beta, kM_P = (th[i] for i in range(NTHETA))

    V = ca.fmax(x[0], _V_FLOOR)
    ML, MP = x[1], x[2]

    cP = MP / V
    cL = ML / V

    p_nom = _flux(cP, k, A, cg)
    if tear is not None:
        lo, hi, fac = tear
        inside = ca.logic_and(cP >= lo, cP <= hi)
        p = p_nom * (1.0 + (fac - 1.0) * inside)
    else:
        p = p_nom

    rL = _partition(alpha, p, kM_L, A)
    dV = (u - 1.0) * p
    dML = -rL * cL * p

    if protein_leakage:
        rP = _partition(beta, p, kM_P, A)
        dMP = -rP * cP * p
    else:
        dMP = ca.SX.zeros(1)

    f = ca.Function("f", [x, u, th], [ca.vertcat(dV, dML, dMP)],
                    ["x", "u", "theta"], ["dx"])
    out = ca.Function("out", [x, th], [ca.vertcat(cP, cL, p, rL)],
                      ["x", "theta"], ["y"])
    return DiafiltrationModel(f=f, out=out, protein_leakage=protein_leakage,
                              tear=tear, params=params)
