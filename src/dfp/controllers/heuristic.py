"""
dfp.controllers.heuristic
=========================
Reference control laws that need no optimiser.

* :class:`ConstantU` – open-loop study of task 2.
* :class:`ThresholdPolicy` – the benchmark policy of Eq. (4).
* :class:`BangBang` – the *analytic* time-optimal feedback law derived in
  :mod:`dfp.controllers.ocp`: concentrate at ``u = 0`` until the protein
  specification is reached, then wash at ``u = 1``.  It costs one comparison
  per sample and is the yardstick every MPC is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import NOMINAL, ProcessParams

__all__ = ["ConstantU", "ThresholdPolicy", "BangBang"]


@dataclass
class ConstantU:
    u: float
    params: ProcessParams = NOMINAL

    @property
    def label(self) -> str:
        return f"u = {self.u:.2f}"

    def __call__(self, x: np.ndarray, t: float = 0.0) -> float:
        return self.u


@dataclass
class ThresholdPolicy:
    """Eq. (4): ``u = 0`` while ``cP < threshold``, ``u = u_high`` afterwards."""

    threshold: float = 55.0
    u_high: float = 0.86
    params: ProcessParams = NOMINAL

    @property
    def label(self) -> str:
        return f"threshold policy (cP*={self.threshold:g}, u={self.u_high:g})"

    def __call__(self, x: np.ndarray, t: float = 0.0) -> float:
        cP = x[2] / x[0]
        return self.u_high if cP >= self.threshold else 0.0


@dataclass
class BangBang:
    """Analytic time-optimal feedback: ``u = 0`` then ``u = 1``.

    ``margin`` switches marginally *before* the specification so that the
    numerical integrator does not step past ``cP_f``; with the event detection
    of :func:`dfp.simulate.closed_loop` a margin of a few 10⁻³ mol m⁻³ is
    enough.
    """

    params: ProcessParams = NOMINAL
    margin: float = 1e-3

    @property
    def label(self) -> str:
        return "analytic optimum (bang-bang)"

    def __call__(self, x: np.ndarray, t: float = 0.0) -> float:
        cP = x[2] / x[0]
        return 1.0 if cP >= self.params.cP_f - self.margin else 0.0

    def suggested_dt(self, x: np.ndarray) -> float:
        """Time left until ``cP`` hits its specification at ``u = 0``.

        Truncating the control interval at that instant keeps the protein
        concentration from stepping past ``cP_f`` (the specification is an
        *equality*), so the bang-bang law realises the analytic optimum
        instead of over-concentrating by up to one control interval.
        """
        P = self.params
        cP = x[2] / x[0]
        if cP >= P.cP_f - self.margin:
            return P.dt_ctrl
        c = np.linspace(cP, P.cP_f - self.margin, 512)
        integrand = x[2] / (P.k * P.A * np.log(P.cg / c) * c ** 2)
        area = float(np.sum(np.diff(c) * 0.5 * (integrand[1:] + integrand[:-1])))
        return float(min(P.dt_ctrl, max(area, 1.0)))
