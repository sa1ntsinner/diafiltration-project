r"""
dfp.estimation
==============
Online identification of the lactose mass-transfer coefficient – the
*adaptive* answer to the parametric mismatch of the additional tasks.

Rationale
---------
Multi-stage NMPC (:mod:`dfp.controllers.multistage`) makes the controller
*immune* to an unknown :math:`k_{M,L}`, but it always pays for the worst
branch.  If the parameter can be **identified from the measurements that are
available anyway** (tank level and an inline lactose assay), the controller can
instead be *corrected*, recovering the performance of a perfectly-modelled
plant.

Two estimators are provided and share one interface, ``update(u, y) -> theta``:

:class:`EKF`
    Extended Kalman filter on the augmented state
    :math:`z=[V,\;M_L,\;M_P,\;\ln\kappa]`, where
    :math:`k_{M,L}=\kappa\,k_{M,L}^{\text{nom}}`.  Estimating the *logarithm*
    keeps the factor positive without constraints and makes the filter
    scale-invariant.  Jacobians are generated symbolically by CasADi, so they
    are exact.

:class:`MHE`
    Moving-horizon estimation: a least-squares NLP over the last ``M`` samples
    with box constraints on :math:`\kappa` and a regularising arrival cost.
    Slower but constrained and far less sensitive to a poor initial guess –
    the standard choice for non-linear, constrained processes.

Measurement model
-----------------
:math:`y=[V,\;c_L]` with independent Gaussian noise.  The volume is a level
measurement; the lactose concentration is an inline refractometer/HPLC reading.
The protein concentration is *not* measured – it follows from ``V`` because
protein is retained (and is estimated when leakage is active).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import casadi as ca
import numpy as np

from .config import NOMINAL, ProcessParams
from .integrate import rk4_integrator
from .model import NTHETA, NX, build_model

__all__ = ["EKF", "MHE", "AdaptiveNMPC", "measurement"]

_KM_L_INDEX = 3   # position of kM_L inside theta


def measurement(x: np.ndarray, rng: Optional[np.random.Generator] = None,
                sigma: Tuple[float, float] = (2e-4, 2.0)) -> np.ndarray:
    """Noisy measurement ``y = [V, cL]``."""
    x = np.asarray(x, float).ravel()
    y = np.array([x[0], x[1] / x[0]])
    if rng is not None:
        y = y + rng.normal(0.0, np.asarray(sigma))
    return y


def _theta_of(kappa: float, params: ProcessParams) -> np.ndarray:
    th = params.theta.copy()
    th[_KM_L_INDEX] = params.kM_L * kappa
    return th


# ─────────────────────────────────────────────────────────────────────────────
#  Extended Kalman filter
# ─────────────────────────────────────────────────────────────────────────────
class EKF:
    """Augmented-state EKF estimating ``[V, ML, MP, ln kappa]``."""

    def __init__(self, params: ProcessParams = NOMINAL, *, dt: Optional[float] = None,
                 kappa0: float = 1.0, n_sub: int = 12,
                 sigma_y: Tuple[float, float] = (2e-4, 2.0),
                 q_state: Tuple[float, float, float] = (1e-10, 1e-6, 1e-12),
                 q_kappa: float = 1e-4, P0_kappa: float = 0.05,
                 kappa_bounds: Tuple[float, float] = (0.1, 2.0)):
        self.P = params
        self.dt = params.dt_ctrl if dt is None else dt
        self._kappa0 = float(kappa0)
        self.z = np.concatenate([params.x0, [np.log(kappa0)]])
        self._P0 = np.diag(list(q_state) + [P0_kappa])
        self.Pk = self._P0.copy()
        self.Q = np.diag(list(q_state) + [q_kappa])
        self.R = np.diag(np.asarray(sigma_y) ** 2)
        self.history: List[float] = [kappa0]
        self._kb = kappa_bounds

        # symbolic augmented step and its Jacobians
        F = rk4_integrator(build_model(params=params).f, n_sub)
        z = ca.SX.sym("z", NX + 1)
        u = ca.SX.sym("u")
        th = ca.SX(params.theta)
        th[_KM_L_INDEX] = params.kM_L * ca.exp(z[NX])
        z_next = ca.vertcat(F(z[:NX], u, th, self.dt), z[NX])
        y = ca.vertcat(z[0], z[1] / z[0])
        self._f = ca.Function("f_aug", [z, u], [z_next])
        self._A = ca.Function("A", [z, u], [ca.jacobian(z_next, z)])
        self._h = ca.Function("h", [z], [y])
        self._C = ca.Function("C", [z], [ca.jacobian(y, z)])

    # ── interface ──────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Forget all data (used when the same controller drives a new plant)."""
        self.z = np.concatenate([self.P.x0, [np.log(self._kappa0)]])
        self.Pk = self._P0.copy()
        self.history = [self._kappa0]

    @property
    def kappa(self) -> float:
        return float(np.exp(self.z[NX]))

    @property
    def theta(self) -> np.ndarray:
        return _theta_of(self.kappa, self.P)

    @property
    def x(self) -> np.ndarray:
        return self.z[:NX].copy()

    def update(self, u: float, y: np.ndarray) -> np.ndarray:
        """One predict/correct cycle; returns the current ``theta`` estimate."""
        # predict
        z_p = np.asarray(self._f(self.z, u)).ravel()
        A = np.asarray(self._A(self.z, u))
        P_p = A @ self.Pk @ A.T + self.Q
        # correct
        C = np.asarray(self._C(z_p))
        S = C @ P_p @ C.T + self.R
        K = np.linalg.solve(S.T, (P_p @ C.T).T).T
        innov = np.asarray(y, float).ravel() - np.asarray(self._h(z_p)).ravel()
        self.z = z_p + K @ innov
        self.z[NX] = float(np.clip(self.z[NX], np.log(self._kb[0]), np.log(self._kb[1])))
        self.Pk = (np.eye(NX + 1) - K @ C) @ P_p
        self.history.append(self.kappa)
        return self.theta


# ─────────────────────────────────────────────────────────────────────────────
#  Moving-horizon estimation
# ─────────────────────────────────────────────────────────────────────────────
class MHE:
    """Least-squares moving-horizon estimator for ``kappa`` and the state."""

    def __init__(self, params: ProcessParams = NOMINAL, *, window: int = 8,
                 dt: Optional[float] = None, kappa0: float = 1.0, n_sub: int = 12,
                 sigma_y: Tuple[float, float] = (2e-4, 2.0),
                 w_arrival: float = 1e-2, w_kappa: float = 1e-2,
                 kappa_bounds: Tuple[float, float] = (0.1, 2.0)):
        self.P = params
        self.M = window
        self.dt = params.dt_ctrl if dt is None else dt
        self.kappa = float(kappa0)
        self.x_hat = params.x0.copy()
        self.history: List[float] = [kappa0]
        self._u: List[float] = []
        self._y: List[np.ndarray] = []
        self._sigma = np.asarray(sigma_y, float)
        self._wa, self._wk = w_arrival, w_kappa
        self._kb = kappa_bounds
        self._kappa0 = float(kappa0)
        self._x_start = params.x0.copy()      # estimate of the oldest window node
        self._F = rk4_integrator(build_model(params=params).f, n_sub)

    def reset(self) -> None:
        """Forget the estimation window (used when the plant is replaced)."""
        self.kappa = self._kappa0
        self.x_hat = self.P.x0.copy()
        self._x_start = self.P.x0.copy()
        self.history = [self._kappa0]
        self._u, self._y = [], []

    @property
    def theta(self) -> np.ndarray:
        return _theta_of(self.kappa, self.P)

    @property
    def x(self) -> np.ndarray:
        return self.x_hat.copy()

    def update(self, u_prev: float, y: np.ndarray) -> np.ndarray:
        """Add the sample ``y`` (taken *after* ``u_prev`` was applied) and re-estimate.

        The window stores ``y_0 … y_m`` and ``u_0 … u_{m-1}`` with ``u_i``
        driving ``x_i → x_{i+1}``; keeping that offset right is essential – an
        off-by-one input/measurement pairing biases the estimate of ``kM_L`` by
        tens of percent even with noise-free data.
        """
        self._y.append(np.asarray(y, float).ravel())
        if len(self._y) > 1:
            self._u.append(float(u_prev))
        if len(self._y) > self.M + 1:
            self._y = self._y[-(self.M + 1):]
            self._u = self._u[-self.M:]
        m = len(self._u)
        if m < 1:
            return self.theta

        opti = ca.Opti()
        X = opti.variable(NX, m + 1)
        lk = opti.variable()
        th = ca.vertcat(*[self.P.kM_L * ca.exp(lk) if i == _KM_L_INDEX
                          else ca.MX(float(self.P.theta[i]))
                          for i in range(NTHETA)])

        J = 0
        for i in range(m):
            opti.subject_to(X[:, i + 1] == self._F(X[:, i], self._u[i], th, self.dt))
        for i in range(m + 1):
            y_pred = ca.vertcat(X[0, i], X[1, i] / X[0, i])
            r = (y_pred - ca.DM(self._y[i])) / ca.DM(self._sigma)
            J += ca.sumsqr(r)
        # arrival cost: keep the *oldest* node of the window close to the estimate
        # that was made for that instant (not to the newest one - anchoring the
        # window start to the current state would drag the whole window forward)
        J += self._wa * ca.sumsqr((X[:, 0] - ca.DM(self.x_hat_prior())) / ca.DM(self.P.x0))
        J += self._wk * (lk - np.log(self.kappa)) ** 2

        opti.subject_to(opti.bounded(self.P.MP0 / (0.98 * self.P.cg), X[0, :].T,
                                     2.0 * self.P.V0))
        opti.subject_to(opti.bounded(0.0, X[1, :].T, 5.0 * self.P.ML0))
        opti.subject_to(opti.bounded(0.5 * self.P.MP0, X[2, :].T, 1.5 * self.P.MP0))
        opti.subject_to(opti.bounded(np.log(self._kb[0]), lk, np.log(self._kb[1])))
        opti.minimize(J)
        opti.set_initial(lk, np.log(self.kappa))
        opti.set_initial(X, np.tile(self.x_hat.reshape(NX, 1), (1, m + 1)))
        opti.solver("ipopt", {"print_time": False},
                    {"print_level": 0, "sb": "yes", "max_iter": 200, "tol": 1e-8})
        try:
            sol = opti.solve()
            self.kappa = float(np.exp(sol.value(lk)))
            Xs = np.asarray(sol.value(X))
            self.x_hat = Xs[:, -1]
            #  the node that will be the window start at the *next* call
            self._x_start = Xs[:, 1] if Xs.shape[1] > 1 else Xs[:, 0]
        except RuntimeError:                              # pragma: no cover
            pass
        self.history.append(self.kappa)
        return self.theta

    def x_hat_prior(self) -> np.ndarray:
        """Estimate of the oldest state currently inside the window."""
        return self._x_start


# ─────────────────────────────────────────────────────────────────────────────
#  Estimator + NMPC = adaptive (self-tuning) controller
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AdaptiveNMPC:
    """Couple an estimator to an NMPC: identify ``kM_L``, then re-optimise.

    The NLP is built **once**; the parameter estimate enters through the NLP
    parameter vector, so adaptation costs nothing beyond the estimator itself.
    """

    mpc: "object"
    estimator: object
    params: ProcessParams = NOMINAL
    rng: Optional[np.random.Generator] = None
    warmup: int = 2
    label: str = "adaptive NMPC (MHE + min-time)"
    kappa_log: List[float] = field(default_factory=list)

    def reset(self) -> None:
        self.mpc.reset()
        if hasattr(self.estimator, "reset"):
            self.estimator.reset()
        self.kappa_log = []
        self._u_prev = 0.0
        self._k = 0

    @property
    def n_fail(self) -> int:
        return getattr(self.mpc, "n_fail", 0)

    def suggested_dt(self, x: np.ndarray) -> float:
        return self.mpc.suggested_dt(x)

    def __call__(self, x: np.ndarray, t: float = 0.0) -> float:
        y = measurement(x, self.rng)
        theta = self.estimator.update(self._u_prev, y)
        self._k += 1
        if self._k > self.warmup:
            self.mpc.set_theta(theta)
        self.kappa_log.append(float(theta[_KM_L_INDEX] / self.params.kM_L))
        u = float(self.mpc(x, t=t))
        self._u_prev = u
        return u

    def __post_init__(self) -> None:
        self.reset()
