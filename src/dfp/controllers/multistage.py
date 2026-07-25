r"""
dfp.controllers.multistage
==========================
Multi-stage (scenario-tree) robust NMPC – the *non-conservative* answer to the
parametric plant–model mismatch of the additional tasks.

Idea
----
Instead of tightening constraints with a worst-case tube (which needs a
linearisation and is conservative for a strongly non-linear batch process), the
uncertainty is represented by a **tree of discrete realisations**
:math:`\theta^{(j)}`.  Up to the *robust horizon* :math:`N_r` the branches
share their control input (non-anticipativity: the controller cannot know
which plant it is driving); afterwards every branch may recover with its own
input sequence, which models the fact that future measurements *will* reveal
the plant.  The result is a control law that is feasible for **all**
realisations without paying the price of an open-loop worst case.

.. math::

   \min_{u,\,h}\ \max_j\ N h^{(j)}
   \quad\text{s.t.}\quad
   \begin{cases}
   x^{(j)}_{k+1}=F\!\left(x^{(j)}_k,u^{(j)}_k,\theta^{(j)},h^{(j)}\right)\\
   u^{(j)}_k = u^{(0)}_k, & k < N_r \quad(\text{non-anticipativity})\\
   c_L^{(j)}\le c_L^{\max},\quad c_P^{(j)}\le c_{P,f}\\
   c_P^{(j)}(N)\ge c_{P,f},\quad c_L^{(j)}(N)\le c_{L,f}
   \end{cases}

The min–max is written in epigraph form so the NLP stays smooth.  Setting
``worst_case=False`` minimises the probability-weighted expected batch time
instead (stochastic instead of robust flavour).

With the four realisations
:math:`k_{M,L}\in\{0.25,0.5,0.75,1\}\,k_{M,L}^{\text{nom}}` requested by the
task sheet and :math:`N_r=1` the problem has four branches – roughly four
times the cost of the nominal NMPC, still far below one second per sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import casadi as ca
import numpy as np

from ..config import KM_L_UNCERTAINTY, NOMINAL, ProcessParams, UncertaintySet
from ..integrate import rk4_integrator
from ..model import NTHETA, NX, build_model
from .base import ConstraintStack, VarStack
from .nmpc import _IPOPT_SILENT

__all__ = ["MultiStageNMPC", "build_multistage_nmpc"]


@dataclass
class MultiStageNMPC:
    solver: ca.Function
    vs: VarStack
    lbg: np.ndarray
    ubg: np.ndarray
    params: ProcessParams
    N: int
    n_scen: int
    robust_horizon: int
    thetas: np.ndarray
    label: str = "multi-stage NMPC"
    _w0: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._w0 = self.vs.x0.copy()
        self._lam_g = None
        self.n_fail = 0
        self.T_pred = float(self.N * self.params.dt_ctrl)
        self.last: Dict[str, np.ndarray] = {}

    def __call__(self, x: np.ndarray, t: float = 0.0) -> float:
        x = np.asarray(x, float).ravel()
        kw = dict(x0=self._w0, p=x, lbx=self.vs.lb, ubx=self.vs.ub,
                  lbg=self.lbg, ubg=self.ubg)
        if self._lam_g is not None:
            kw["lam_g0"] = self._lam_g
        sol = self.solver(**kw)
        if not self.solver.stats().get("success", False):
            self.n_fail += 1
        else:
            self._lam_g = np.asarray(sol["lam_g"]).ravel()
        w = np.asarray(sol["x"]).ravel()
        U0 = self.vs.extract(w, "U0")
        hs = self.vs.extract(w, "h")
        self.T_pred = float(self.params.dt_ctrl + (self.N - 1) * np.max(hs))
        self.last = {"U0": U0, "h": hs,
                     **{f"X{j}": self.vs.extract(w, f"X{j}") for j in range(self.n_scen)}}
        self._w0 = w.copy()
        return float(np.clip(U0[0], self.params.u_min, self.params.u_max))

    def suggested_dt(self, x: np.ndarray) -> float:
        return self.params.dt_ctrl


# ─────────────────────────────────────────────────────────────────────────────
def build_multistage_nmpc(
    N: int = 20,
    *,
    params: ProcessParams = NOMINAL,
    uncertainty: UncertaintySet = KM_L_UNCERTAINTY,
    robust_horizon: int = 1,
    worst_case: bool = True,
    rho_term: float = 1e4,
    rho_path: float = 1e4,
    n_sub: Optional[int] = None,
    label: Optional[str] = None,
) -> MultiStageNMPC:
    """Build the scenario-tree robust NMPC (free final time, min–max cost)."""
    P = params
    reals = uncertainty.realisations(P)
    thetas = np.array([r.theta for r in reals])
    S = len(reals)
    probs = uncertainty.probabilities(S)

    h_lo, h_hi = 1.0, 2.0 * P.t_max / (N - 1)
    if N < 2:
        raise ValueError("the multi-stage formulation needs N >= 2")
    if n_sub is None:
        n_sub = int(max(P.n_sub_mpc, np.ceil(max(h_hi, P.dt_ctrl) / 300.0)))
    F = rk4_integrator(build_model(params=P).f, n_sub)

    vs = VarStack()
    V_lo = P.MP0 / (0.98 * P.cg)
    x_lb = np.tile(np.array([V_lo, 0.0, 0.0]), N + 1)
    x_ub = np.tile(np.array([2.0 * P.V0, 10.0 * P.ML0, 10.0 * P.MP0]), N + 1)

    X = [vs.add(f"X{j}", (NX, N + 1), lb=x_lb, ub=x_ub, x0=np.tile(P.x0, N + 1))
         for j in range(S)]
    U = [vs.add(f"U{j}", N, lb=P.u_min, ub=P.u_max, x0=0.3) for j in range(S)]
    h = vs.add("h", S, lb=h_lo, ub=h_hi, x0=min(max(P.dt_ctrl, h_lo), h_hi))
    s = vs.add("s", (4, S), lb=0.0, ub=ca.inf, x0=0.0)
    tau = vs.add("tau", 1, lb=0.0, ub=ca.inf, x0=4.0)   # epigraph of the max

    x_init = ca.SX.sym("x_init", NX)
    cs = ConstraintStack()
    J = ca.SX.zeros(1)

    for j in range(S):
        th_j = ca.DM(thetas[j])
        cs.eq(X[j][:, 0] - x_init)
        for k in range(N):
            h_k = P.dt_ctrl if k == 0 else h[j]   # interval 0 is applied to the plant
            cs.eq(X[j][:, k + 1] - F(X[j][:, k], U[j][k], th_j, h_k))
            if k < robust_horizon and j > 0:                # non-anticipativity
                cs.eq(U[j][k] - U[0][k])
        for k in range(N + 1):
            cP_k = X[j][2, k] / X[j][0, k]
            cL_k = X[j][1, k] / X[j][0, k]
            cs.leq(cL_k - P.cL_max - s[1, j])
            if k > 0:
                cs.leq(cP_k - P.cP_f - s[0, j])
        cP_N = X[j][2, N] / X[j][0, N]
        cL_N = X[j][1, N] / X[j][0, N]
        cs.leq(P.cP_f - cP_N - s[2, j])
        cs.leq(cL_N - P.cL_f - s[3, j])

        T_j = (P.dt_ctrl + (N - 1) * h[j]) / 3600.0
        if worst_case:
            cs.leq(T_j - tau[0])
        else:
            J += probs[j] * T_j
        J += rho_term * (s[2, j] / P.cP_f + s[3, j] / P.cL_f)
        J += rho_path * (s[0, j] / P.cP_f + s[1, j] / P.cL_max)

    if worst_case:
        J += tau[0]

    g, lbg, ubg = cs.build()
    solver = ca.nlpsol("ms_nmpc", "ipopt",
                       {"f": J, "x": vs.vector, "p": x_init, "g": g}, _IPOPT_SILENT)

    tag = "min-max" if worst_case else "expected"
    lbl = label or f"multi-stage NMPC ({S} scen., N_r={robust_horizon}, {tag})"
    return MultiStageNMPC(solver=solver, vs=vs, lbg=lbg, ubg=ubg, params=P, N=N,
                          n_scen=S, robust_horizon=robust_horizon, thetas=thetas,
                          label=lbl)
