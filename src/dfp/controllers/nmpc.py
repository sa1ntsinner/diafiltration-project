"""
dfp.controllers.nmpc
====================
One non-linear MPC implementation, five objective functions.

Why a single builder?
---------------------
All formulations of the task sheet share the same dynamics, the same
crystallisation path constraint and the same terminal specification; only the
*cost* changes.  Sharing the constraint code means the comparison of
objectives is a genuine comparison of objectives and not of implementations.

Objectives
----------
``tracking`` (Eq. 3, literally)
    :math:`J=\\sum_{k=0}^{N}(c_{L,k}-c_{L,f})^2+(c_{P,k}-c_{P,f})^2`
    on the fixed grid Δt = 10 min.  Kept *exactly* as written in the task –
    including its poor scaling – because the project asks to demonstrate why
    it is unsuitable.

``tracking_scaled``
    Same, with each term normalised by its own span.  Isolates "bad scaling"
    from "bad objective".

``exact_penalty``
    :math:`J=\\sum_k \\Delta t\\,[\\,1+\\rho_L(c_{L,k}-c_{L,f})_+
    +\\rho_P(c_{P,f}-c_{P,k})_+]` – an :math:`\\ell_1` (non-quadratic) cost on
    the *remaining* specification violation, integrated over time.  Being an
    exact penalty it makes "finish early" strictly cheaper than "finish late",
    which the quadratic tracking cost does not.

``min_time`` *(recommended)*
    True free-final-time formulation.  The **first** interval is pinned to the
    sampling time Δt (it is the one that is actually applied to the plant),
    while the remaining ``N-1`` intervals share a *free* length ``h``, so that

    .. math:: J = T = \Delta t + (N-1)\,h .

    This is the direct discretisation of the original time-optimal OCP, hence
    the closed loop reproduces the analytic optimum
    (:mod:`dfp.controllers.ocp`) to within the sampling time.  Pinning the
    first interval matters: with a *uniform* free grid the optimiser plans in
    intervals of length ``h`` but the plant receives ``u₀`` for Δt, and near
    the end of the batch (``h ≪ Δt``) the protein concentration overshoots its
    specification by more than 10 %.

``economic``
    :math:`J=c_t T+\\int \\lambda(t)\\,P_\\text{pump}(u)\\,dt` – trades batch
    time against a time-of-use electricity tariff.

Feasibility
-----------
Every terminal and path constraint carries an :math:`\\ell_\\infty` slack with
a large exact-penalty weight, so the NLP is feasible for **any** initial state
and **any** horizon.  This removes the silent failure of the original code,
where ``N = 5`` made the terminal equality unreachable, IPOPT returned an
infeasible point and the first element of that point was used as the control
action.  Slack activity is logged instead and reported in the studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import casadi as ca
import numpy as np

from ..config import NOMINAL, ProcessParams
from ..integrate import rk4_integrator
from ..model import NTHETA, NX, build_model
from ..tariff import lambda_tou_casadi
from .base import ConstraintStack, VarStack

__all__ = ["NMPC", "build_nmpc", "OBJECTIVES"]

OBJECTIVES = ("tracking", "tracking_scaled", "l1_time", "min_time", "economic")

_IPOPT_SILENT = {
    "ipopt.print_level": 0,
    "print_time": False,
    "ipopt.sb": "yes",
    "ipopt.max_iter": 500,
    "ipopt.tol": 1e-8,
    "ipopt.acceptable_tol": 1e-6,
    "ipopt.warm_start_init_point": "yes",
    "ipopt.mu_strategy": "adaptive",
}


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class NMPC:
    """Callable MPC controller ``u = mpc(x, t=...)``."""

    solver: ca.Function
    vs: VarStack
    lbg: np.ndarray
    ubg: np.ndarray
    params: ProcessParams
    N: int
    objective: str
    free_time: bool
    theta: np.ndarray
    label: str = "NMPC"
    verbose_fail: bool = False

    # ── bookkeeping ────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        self._theta0 = np.asarray(self.theta, dtype=float).copy()
        self.reset()

    def reset(self) -> None:
        """Full reset, *including* the model parameters.

        Restoring ``theta`` matters when one solver object is reused for several
        plants (studies, Monte-Carlo): otherwise the controller would start the
        next run with the parameter identified for the previous one.
        """
        self.theta = self._theta0.copy()
        self._w0 = self.vs.x0.copy()
        self._lam_g = None
        self._lam_x = None
        self.n_fail = 0
        self.slack_active = 0
        self.last: Dict[str, np.ndarray] = {}
        self.T_pred = float(self.N * self.params.dt_ctrl)

    def set_theta(self, theta: np.ndarray) -> None:
        """Update the model parameters used inside the optimiser (no rebuild)."""
        self.theta = np.asarray(theta, dtype=float)

    # ── the control law ────────────────────────────────────────────────────
    def __call__(self, x: np.ndarray, t: float = 0.0) -> float:
        x = np.asarray(x, dtype=float).ravel()
        p = np.concatenate([x, self.theta, [t]])
        self._seed(x)
        kw = dict(x0=self._w0, p=p, lbx=self.vs.lb, ubx=self.vs.ub,
                  lbg=self.lbg, ubg=self.ubg)
        if self._lam_g is not None:
            kw.update(lam_g0=self._lam_g, lam_x0=self._lam_x)
        sol = self.solver(**kw)
        ok = self.solver.stats().get("success", False)
        w = np.asarray(sol["x"]).ravel()
        if not ok:
            self.n_fail += 1
            if self.verbose_fail:  # pragma: no cover
                print(f"[{self.label}] IPOPT: {self.solver.stats().get('return_status')}")
        else:
            self._lam_g = np.asarray(sol["lam_g"]).ravel()
            self._lam_x = np.asarray(sol["lam_x"]).ravel()

        U = self.vs.extract(w, "U")
        X = self.vs.extract(w, "X")
        s = self.vs.extract(w, "s")
        h = float(self.vs.extract(w, "h")[0]) if self.free_time else self.params.dt_ctrl
        self.T_pred = (self.params.dt_ctrl + (self.N - 1) * h if self.free_time
                       else self.N * h)
        self.slack_active += int(np.max(s) > 1e-6)
        grid = np.concatenate([[0.0], np.cumsum(
            [self.params.dt_ctrl if (self.free_time and k == 0) else h
             for k in range(self.N)])])
        self.last = {"X": X, "U": U, "h": h, "s": s, "t_grid": t + grid}
        self._shift(w, h)
        return float(np.clip(U[0], self.params.u_min, self.params.u_max))

    def suggested_dt(self, x: np.ndarray) -> float:
        """Always one full sampling period.

        The optimiser already plans its first interval with exactly this length
        (see :func:`build_nmpc`), and :func:`dfp.simulate.closed_loop` truncates
        the *last* interval at the terminal event, so no shortening is needed
        here.  Returning the predicted remaining time instead would make the
        driver take arbitrarily small steps whenever the terminal slack is
        active - i.e. exactly when the batch is hardest.
        """
        return self.params.dt_ctrl

    # ── warm start helpers ─────────────────────────────────────────────────
    def _seed(self, x: np.ndarray) -> None:
        if not self.last:
            X = np.tile(x.reshape(NX, 1), (1, self.N + 1))
            self.vs.set_x0(self._w0, "X", X)

    def _shift(self, w: np.ndarray, h: float) -> None:
        """Shift the previous solution one interval ahead (standard warm start)."""
        X = self.vs.extract(w, "X")
        U = self.vs.extract(w, "U")
        Xs = np.concatenate([X[:, 1:], X[:, -1:]], axis=1)
        Us = np.concatenate([U[1:], U[-1:]])
        self._w0 = w.copy()
        self.vs.set_x0(self._w0, "X", Xs)
        self.vs.set_x0(self._w0, "U", Us)
        if self.free_time:
            self.vs.set_x0(self._w0, "h", max(h, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
def build_nmpc(
    objective: str = "min_time",
    N: int = 20,
    *,
    params: ProcessParams = NOMINAL,
    theta: Optional[np.ndarray] = None,
    n_sub: Optional[int] = None,
    protein_leakage: bool = False,
    rho_term: float = 1e4,
    rho_path: float = 1e4,
    rho_du: float = 0.0,
    rho_L: float = 10.0,
    rho_P: float = 10.0,
    value_of_time: float = 30.0,
    price_solvent: float = 0.0,
    terminal_constraint: bool = True,
    back_off_cL: float = 0.0,
    label: Optional[str] = None,
    h_bounds: Optional[tuple] = None,
) -> NMPC:
    """Assemble one of the five MPC formulations.

    Parameters
    ----------
    objective
        One of :data:`OBJECTIVES`.
    N
        Prediction horizon (number of shooting intervals).
    protein_leakage
        Whether the controller's own model includes protein passage (Eq. 6).
        Comparing ``False`` (the nominal assumption) with ``True`` isolates the
        cost of the *structural* mismatch of additional task 2.
    theta
        Model parameters *believed by the controller*.  Defaults to
        ``params.theta``.  Can be updated online via :meth:`NMPC.set_theta`
        (used by the adaptive/estimator-based controller) – the NLP is built
        once and ``theta`` enters as a parameter.
    rho_term, rho_path
        Exact-penalty weights of the terminal / path slacks.  Normalised, so
        the same numbers work for every objective.
    rho_L, rho_P
        Weights of the two normalised distance terms of ``l1_time``.
    value_of_time
        €/h charged for occupying the plant (``economic`` objective).
    price_solvent
        €/m³ of diafiltration buffer (``economic`` objective).  Section 5 of
        the report shows analytically that the *solvent*-optimal policy is the
        same bang-bang law as the *time*-optimal one, so this term shifts the
        optimum only through the tariff.
    terminal_constraint
        If ``False`` the terminal specification is dropped entirely.  Used to
        reproduce Eq. (3) *exactly as written in the task sheet* and to show
        that pure quadratic tracking then never terminates the batch.
    back_off_cL
        Constraint tightening [mol m⁻³] subtracted from ``cL_max`` – the
        "back-off" robustification benchmark.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")

    P = params
    theta0 = P.theta if theta is None else np.asarray(theta, float)
    free_time = objective in ("min_time", "economic")

    # ── grid / integrator ──────────────────────────────────────────────────
    #  interval 0 is the one that is applied to the plant → pin it to dt_ctrl
    if free_time:
        if N < 2:
            raise ValueError("free-final-time formulations need N >= 2")
        #  allow the *planned* batch to exceed t_max: a controller that is
        #  artificially forbidden to plan a long batch behaves like the
        #  myopic tracking MPC as soon as the plant is slower than nominal.
        h_lo, h_hi = h_bounds or (1.0, 2.0 * P.t_max / (N - 1))
    else:
        h_lo = h_hi = P.dt_ctrl
    if n_sub is None:
        n_sub = int(max(P.n_sub_mpc, np.ceil(max(h_hi, P.dt_ctrl) / 300.0)))
    F = rk4_integrator(build_model(params=P, protein_leakage=protein_leakage).f,
                      n_sub)

    # ── decision variables ────────────────────────────────────────────────
    vs = VarStack()
    V_lo = P.MP0 / (0.98 * P.cg)                      # keeps ln(cg/cP) > 0
    X = vs.add("X", (NX, N + 1),
               lb=np.tile(np.array([V_lo, 0.0, 0.0]), N + 1),
               ub=np.tile(np.array([2.0 * P.V0, 10.0 * P.ML0, 10.0 * P.MP0]), N + 1),
               x0=np.tile(P.x0, N + 1))
    U = vs.add("U", N, lb=P.u_min, ub=P.u_max, x0=0.5)
    h = vs.add("h", 1, lb=h_lo, ub=h_hi, x0=min(max(P.dt_ctrl, h_lo), h_hi))
    #  s = [max cP path violation, max cL path violation, cP terminal, cL terminal].
    #  With the terminal constraint switched off the last two are pinned to zero
    #  so the NLP keeps no dead degrees of freedom.
    _inf_term = ca.inf if terminal_constraint else 0.0
    s = vs.add("s", 4, lb=0.0, ub=[ca.inf, ca.inf, _inf_term, _inf_term], x0=0.0)
    #  epigraph variables for the l1 objective (keeps the NLP smooth: a plain
    #  fmax(...) in the cost makes IPOPT stall or fail on ~20 % of the samples)
    if objective == "l1_time":
        eL = vs.add("eL", N, lb=0.0, ub=ca.inf, x0=0.0)
        eP = vs.add("eP", N, lb=0.0, ub=ca.inf, x0=0.0)

    # ── parameters ────────────────────────────────────────────────────────
    x_init = ca.SX.sym("x_init", NX)
    th = ca.SX.sym("theta", NTHETA)
    t_now = ca.SX.sym("t_now")
    par = ca.vertcat(x_init, th, t_now)

    cs = ConstraintStack()
    cs.eq(X[:, 0] - x_init)

    cL_bound = P.cL_max - back_off_cL
    J = ca.SX.zeros(1)

    def conc(col):
        Vk, MLk, MPk = X[0, col], X[1, col], X[2, col]
        return MPk / Vk, MLk / Vk

    def _p_of(xcol, th_):
        """Permeate flow of a symbolic state column (used by the buffer cost)."""
        return th_[0] * th_[1] * ca.log(th_[2] * xcol[0] / xcol[2])

    h_k = [P.dt_ctrl if (free_time and k == 0) else h[0] for k in range(N)]
    T_pred = sum(h_k)

    for k in range(N):
        cs.eq(X[:, k + 1] - F(X[:, k], U[k], th, h_k[k]))
        if rho_du > 0.0 and k > 0:
            J += rho_du * (U[k] - U[k - 1]) ** 2

    for k in range(N + 1):
        cP_k, cL_k = conc(k)
        cs.leq(cL_k - cL_bound - s[1])            # crystallisation limit
        if k > 0:
            cs.leq(cP_k - P.cP_f - s[0])          # never over-concentrate

    # ── objective ─────────────────────────────────────────────────────────
    if objective in ("tracking", "tracking_scaled"):
        wL = 1.0 if objective == "tracking" else 1.0 / (P.cL0 - P.cL_f) ** 2
        wP = 1.0 if objective == "tracking" else 1.0 / (P.cP_f - P.cP0) ** 2
        for k in range(N + 1):
            cP_k, cL_k = conc(k)
            J += wL * (cL_k - P.cL_f) ** 2 + wP * (cP_k - P.cP_f) ** 2

    elif objective == "l1_time":
        gap_L = P.cL0 - P.cL_f
        gap_P = P.cP_f - P.cP0
        for k in range(N):
            cP_k, cL_k = conc(k)
            cs.leq(cL_k - P.cL_f - eL[k])          # eL[k] >= (cL - cL_f)+
            cs.leq(P.cP_f - cP_k - eP[k])          # eP[k] >= (cP_f - cP)+
            J += (1.0 + rho_L * eL[k] / gap_L
                      + rho_P * eP[k] / gap_P) * h_k[k] / 3600.0

    elif objective == "min_time":
        J += T_pred / 3600.0

    elif objective == "economic":
        J += value_of_time * T_pred / 3600.0
        t_k = 0.0
        for k in range(N):
            lam = lambda_tou_casadi(t_now + t_k)
            kw = P.pump_idle_kW + P.pump_dyn_kW * U[k]
            J += lam * kw * h_k[k] / 3600.0                    # electricity  [EUR]
            J += price_solvent * U[k] * _p_of(X[:, k], th) * h_k[k]  # buffer  [EUR]
            t_k = t_k + h_k[k]

    # ── terminal specification (soft, exact penalty) ──────────────────────
    cP_N, cL_N = conc(N)
    if terminal_constraint:
        cs.leq(P.cP_f - cP_N - s[2])
        cs.leq(cL_N - P.cL_f - s[3])
        J += rho_term * (s[2] / P.cP_f + s[3] / P.cL_f)
    J += rho_path * (s[0] / P.cP_f + s[1] / P.cL_max)

    g, lbg, ubg = cs.build()
    nlp = {"f": J, "x": vs.vector, "p": par, "g": g}
    solver = ca.nlpsol("nmpc", "ipopt", nlp, _IPOPT_SILENT)

    lbl = label or f"{objective} MPC (N={N})"
    return NMPC(solver=solver, vs=vs, lbg=lbg, ubg=ubg, params=P, N=N,
                objective=objective, free_time=free_time, theta=theta0, label=lbl)
