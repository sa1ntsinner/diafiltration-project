"""
dfp.plant
=========
"Reality" for the closed-loop studies.

A :class:`Plant` couples

* a *model variant* (nominal, filter-cake tear, protein leakage …),
* the **true** parameter vector ``theta_true`` (which the controller may not
  know), and
* a high-accuracy integrator.

Design note
-----------
The previous implementation kept the protein hold-up as a *Python attribute*
that was mutated inside the right-hand side.  Because RK4 evaluates the RHS
four times per step, the hold-up was advanced four times per step with the
wrong step length, and the "plant" was no longer a well-defined dynamic
system (results depended on the integrator).  Protein is now a genuine state,
so every scenario is a plain ODE and the simulation is reproducible and
integrator-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import casadi as ca
import numpy as np

from .config import NOMINAL, ProcessParams
from .integrate import rk4_integrator
from .model import DiafiltrationModel, build_model

__all__ = ["Plant", "nominal_plant", "tear_plant", "mismatch_plant",
           "leakage_plant", "TEAR_WINDOW"]

#: Filter-cake tear of Eq. (5): flux doubles while 30 ≤ cP ≤ 60 mol m⁻³.
TEAR_WINDOW: Tuple[float, float, float] = (30.0, 60.0, 2.0)


@dataclass
class Plant:
    """A concrete plant realisation used as the simulated truth."""

    name: str = "nominal"
    params: ProcessParams = NOMINAL
    protein_leakage: bool = False
    tear: Optional[Tuple[float, float, float]] = None
    theta_true: Optional[np.ndarray] = None
    n_sub: Optional[int] = None
    model: DiafiltrationModel = field(init=False, repr=False)
    _F: ca.Function = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model = build_model(protein_leakage=self.protein_leakage,
                                 tear=self.tear, params=self.params)
        if self.theta_true is None:
            self.theta_true = self.params.theta
        self.theta_true = np.asarray(self.theta_true, dtype=float)
        self._F = rk4_integrator(self.model.f, 1)

    # ── dynamics ───────────────────────────────────────────────────────────
    @property
    def sub_steps(self) -> int:
        return self.n_sub if self.n_sub is not None else self.params.n_sub_plant

    def step(self, x: np.ndarray, u: float, dt: float,
             n_sub: Optional[int] = None) -> np.ndarray:
        """Advance the true plant by ``dt`` with a piecewise-constant input."""
        n = n_sub if n_sub is not None else self.sub_steps
        h = dt / n
        xk = np.asarray(x, dtype=float)
        for _ in range(n):
            xk = np.asarray(self._F(xk, u, self.theta_true, h)).ravel()
        return xk

    def outputs(self, x) -> np.ndarray:
        """``[cP, cL, p, rL]`` evaluated with the *true* parameters."""
        return self.model.outputs(x, self.theta_true)

    def concentrations(self, x) -> Tuple[np.ndarray, np.ndarray]:
        y = self.outputs(x)
        return y[0], y[1]

    # ── specification test ─────────────────────────────────────────────────
    def specs_met(self, x: np.ndarray) -> bool:
        return self.spec_residual(x) <= 0.0

    def spec_residual(self, x: np.ndarray) -> float:
        """``≤ 0`` ⟺ both terminal specifications are satisfied.

        Both requirements are scaled by their own set point and share the
        relative tolerance ``params.spec_tol``, so the test is dimensionless
        and a controller that lands on the specification from below (e.g. the
        bang-bang law, which holds ``cP`` at ``cP_f`` while washing) is not
        rejected by round-off.
        """
        cP, cL = self.concentrations(x)
        P = self.params
        tol = P.spec_tol
        return float(max((P.cP_f - cP[0]) / P.cP_f - tol,
                         (cL[0] - P.cL_f) / P.cL_f - tol))


# ─────────────────────────────────────────────────────────────────────────────
#  Ready-made scenarios
# ─────────────────────────────────────────────────────────────────────────────
def nominal_plant(params: ProcessParams = NOMINAL) -> Plant:
    """Perfect model match."""
    return Plant(name="nominal", params=params)


def tear_plant(params: ProcessParams = NOMINAL,
               window: Tuple[float, float, float] = TEAR_WINDOW) -> Plant:
    """Filter-cake tear – permeate flow doubles inside a ``cP`` window (Eq. 5)."""
    lo, hi, fac = window
    return Plant(name=f"tear x{fac:g} ({lo:g}-{hi:g})", params=params, tear=window)


def mismatch_plant(factor: float, params: ProcessParams = NOMINAL) -> Plant:
    """Parametric mismatch ``kM_L,true = factor · kM_L`` (additional task 1)."""
    true = params.scaled(kM_L=factor)
    return Plant(name=f"kM_L x{factor:g}", params=params, theta_true=true.theta)


def leakage_plant(params: ProcessParams = NOMINAL, *, beta: float | None = None,
                  kM_P: float | None = None) -> Plant:
    """Structural mismatch – protein partially permeates (Eq. 6)."""
    p = params
    if beta is not None:
        p = p.with_(beta=beta)
    if kM_P is not None:
        p = p.with_(kM_P=kM_P)
    return Plant(name=f"protein leakage (beta={p.beta:g})", params=params,
                 protein_leakage=True, theta_true=p.theta)
