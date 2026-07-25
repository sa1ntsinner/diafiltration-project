r"""
dfp.controllers.ocp
===================
The *exact* time-optimal solution of the benchmark – analytically and
numerically.

Analytic derivation
-------------------
Because :math:`u\le 1` the volume never increases, so :math:`c_P=M_P/V` is
monotonically non-decreasing and can be used as the independent variable.
With :math:`\sigma := 1/(1-u)\in[1,\infty)` the dynamics become **linear in
the control** :math:`\sigma`:

.. math::

   \frac{\mathrm dt}{\mathrm dc_P} = \frac{M_P}{p\,c_P^{2}}\,\sigma ,
   \qquad
   \frac{\mathrm d\ln c_L}{\mathrm dc_P}
        = \frac{1-r_L(c_P)\,\sigma}{c_P} .

Integrating the second equation from :math:`c_{P,0}` to :math:`c_{P,f}` and
imposing the lactose specification turns the problem into a *linear program in
a function*:

.. math::

   \min_{\sigma(\cdot)\ge 1}\ \int_{c_{P,0}}^{c_{P,f}}
        \underbrace{\frac{M_P}{p\,c_P^{2}}}_{a(c_P)}\,\sigma\,\mathrm dc_P
   \quad\text{s.t.}\quad
   \int_{c_{P,0}}^{c_{P,f}}\frac{r_L}{c_P}\,\sigma\,\mathrm dc_P
        = \ln\frac{c_{L,0}}{c_{L,f}} + \ln\frac{c_{P,f}}{c_{P,0}} .

The optimal solution of such a program puts the minimum admissible effort
:math:`\sigma=1` everywhere and concentrates all remaining "washing" where the
price per unit of washing,

.. math:: \frac{a(c_P)\,c_P}{r_L(c_P)} = \frac{M_P}{p(c_P)\,c_P\,r_L(c_P)},

is smallest.  For the given data :math:`p\,c_P\,r_L` increases strictly on
:math:`[10,100]\,\mathrm{mol\,m^{-3}}` (its maximum lies at
:math:`c_P\approx127`, outside the reachable set), hence the minimiser is the
*right end point*:

.. admonition:: Optimal operation

   1. **Pre-concentration**  :math:`u=0` until :math:`c_P=c_{P,f}`;
   2. **Constant-volume diafiltration**  :math:`u=1` until :math:`c_L=c_{L,f}`.

   No singular arc occurs, and the crystallisation limit stays inactive
   (:math:`c_L^{\max,\text{traj}}\approx232 < 570`).

Closed-form batch time
----------------------
.. math::

   t_1=\int_{c_{P,0}}^{c_{P,f}}\frac{M_P\,\mathrm dc_P}{p(c_P)c_P^{2}},\quad
   c_L(t_1)=c_{L,0}\exp\!\int_{c_{P,0}}^{c_{P,f}}\frac{1-r_L}{c_P}\mathrm dc_P,

.. math::

   t_2=\frac{M_P}{p(c_{P,f})\,c_{P,f}\,r_L(c_{P,f})}
        \ln\frac{c_L(t_1)}{c_{L,f}},\qquad T^\star=t_1+t_2 .

For the nominal data :math:`T^\star = 3.5435\;\mathrm h`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import casadi as ca
import numpy as np

from ..config import NOMINAL, ProcessParams
from ..integrate import rk4_integrator
from ..model import NX, build_model

__all__ = ["AnalyticOptimum", "analytic_optimum", "solve_min_time_ocp",
           "switching_price"]


# ─────────────────────────────────────────────────────────────────────────────
def _p(cP, P: ProcessParams):
    return P.k * P.A * np.log(P.cg / cP)


def _rL(cP, P: ProcessParams):
    return P.alpha / (1.0 + (P.alpha - 1.0) * np.exp(_p(cP, P) / (P.kM_L * P.A)))


def switching_price(P: ProcessParams = NOMINAL, n: int = 2001
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Price of washing ``MP/(p·cP·rL)`` over the reachable ``cP`` range."""
    cP = np.linspace(P.cP0, P.cP_f, n)
    return cP, P.MP0 / (_p(cP, P) * cP * _rL(cP, P))


@dataclass
class AnalyticOptimum:
    """Result of :func:`analytic_optimum`."""

    T: float                  #: optimal batch time [s]
    t_switch: float           #: end of the pre-concentration phase [s]
    cL_switch: float          #: lactose concentration at the switch [mol m⁻³]
    cL_peak: float            #: maximum lactose concentration along the batch
    t: np.ndarray             #: time grid of the optimal trajectory [s]
    V: np.ndarray
    cP: np.ndarray
    cL: np.ndarray
    u: np.ndarray
    bang_bang: bool           #: whether the "concentrate-then-wash" structure is optimal

    @property
    def T_h(self) -> float:
        return self.T / 3600.0

    def summary(self) -> Dict[str, float]:
        return {"T_h": round(self.T_h, 5),
                "t_switch_h": round(self.t_switch / 3600.0, 5),
                "cL_switch": round(self.cL_switch, 3),
                "cL_peak": round(self.cL_peak, 3),
                "bang_bang_optimal": self.bang_bang}


def analytic_optimum(P: ProcessParams = NOMINAL, n: int = 200_001
                     ) -> AnalyticOptimum:
    """Closed-form time-optimal solution (quadrature only, no optimiser)."""
    cP = np.linspace(P.cP0, P.cP_f, n)

    # structure test: is the washing price minimal at the right end point?
    _, price = switching_price(P, n=4001)
    bang_bang = bool(np.argmin(price) == price.size - 1)

    # phase 1 – pre-concentration at u = 0
    a = P.MP0 / (_p(cP, P) * cP ** 2)
    t1_traj = np.concatenate([[0.0], np.cumsum(np.diff(cP) * 0.5 * (a[1:] + a[:-1]))])
    b = (1.0 - _rL(cP, P)) / cP
    ln_traj = np.concatenate([[0.0], np.cumsum(np.diff(cP) * 0.5 * (b[1:] + b[:-1]))])
    cL1_traj = P.cL0 * np.exp(ln_traj)
    t1, cL1 = float(t1_traj[-1]), float(cL1_traj[-1])

    # phase 2 – constant-volume diafiltration at u = 1
    xi = _p(P.cP_f, P) * P.cP_f / P.MP0
    rate = xi * _rL(P.cP_f, P)
    t2 = float(np.log(cL1 / P.cL_f) / rate)
    t2_traj = np.linspace(0.0, t2, 2001)
    cL2_traj = cL1 * np.exp(-rate * t2_traj)

    t = np.concatenate([t1_traj, t1 + t2_traj[1:]])
    cP_traj = np.concatenate([cP, np.full(t2_traj.size - 1, P.cP_f)])
    cL_traj = np.concatenate([cL1_traj, cL2_traj[1:]])
    u = np.concatenate([np.zeros(cP.size), np.ones(t2_traj.size - 1)])

    return AnalyticOptimum(
        T=t1 + t2, t_switch=t1, cL_switch=cL1, cL_peak=float(cL_traj.max()),
        t=t, V=P.MP0 / cP_traj, cP=cP_traj, cL=cL_traj, u=u, bang_bang=bang_bang)


# ─────────────────────────────────────────────────────────────────────────────
def solve_min_time_ocp(
    P: ProcessParams = NOMINAL,
    N: int = 200,
    *,
    theta: Optional[np.ndarray] = None,
    protein_leakage: bool = False,
    n_sub: int = 2,
    T_guess: float = 4 * 3600.0,
    verbose: bool = False,
) -> Dict[str, np.ndarray | float]:
    """Direct free-final-time OCP – the numerical reference solution.

    Uses the same RK4 discretisation as the MPC, ``N`` intervals of free but
    equal length, and enforces the terminal specification as a hard equality
    on ``cP`` and an inequality on ``cL``.
    """
    theta = P.theta if theta is None else np.asarray(theta, float)
    F = rk4_integrator(build_model(protein_leakage=protein_leakage, params=P).f, n_sub)

    opti = ca.Opti()
    X = opti.variable(NX, N + 1)
    U = opti.variable(N)
    T = opti.variable()
    h = T / N

    for i in range(N):
        opti.subject_to(X[:, i + 1] == F(X[:, i], U[i], theta, h))
    opti.subject_to(X[:, 0] == ca.DM(P.x0))
    opti.subject_to(opti.bounded(P.u_min, U, P.u_max))
    opti.subject_to(opti.bounded(P.MP0 / (0.98 * P.cg), X[0, :].T, 2.0 * P.V0))
    opti.subject_to(X[1, :].T / X[0, :].T <= P.cL_max)
    opti.subject_to(X[2, N] / X[0, N] == P.cP_f)
    opti.subject_to(X[1, N] / X[0, N] <= P.cL_f)
    opti.subject_to(opti.bounded(60.0, T, P.t_max))
    opti.minimize(T / 3600.0)

    opti.set_initial(T, T_guess)
    opti.set_initial(U, 0.3)
    opti.set_initial(X, np.tile(P.x0.reshape(NX, 1), (1, N + 1)))
    opti.solver("ipopt", {"print_time": False},
                {"print_level": 5 if verbose else 0, "sb": "yes",
                 "max_iter": 3000, "tol": 1e-10})
    sol = opti.solve()

    Xv = np.asarray(sol.value(X))
    Tv = float(sol.value(T))
    return {"T": Tv, "T_h": Tv / 3600.0, "X": Xv, "u": np.asarray(sol.value(U)).ravel(),
            "t": np.linspace(0.0, Tv, N + 1),
            "cP": Xv[2] / Xv[0], "cL": Xv[1] / Xv[0], "V": Xv[0], "N": N}
