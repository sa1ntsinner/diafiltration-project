"""
sim/scenarios.py

Scenario classes – each supplies a continuous-time RHS f(x, u, t).

The base class `Scenario` wraps around a process model (`ProcessParams`) and
its continuous dynamics. Subclasses override `rhs()` to introduce specific
plant deviations such as damage, parameter mismatch, or leakage.

Used for robustness testing, Monte-Carlo simulation, and controller validation.
"""

from __future__ import annotations
from dataclasses import replace
import numpy as np

from core.params import ProcessParams, default as P_default
from core.dynamics import (
    rhs as rhs_nom,                    # nominal dynamics function (NumPy version)
    flux_permeate,                    # helper: protein flux through membrane
    lactose_permeate_conc,            # helper: lactose concentration in permeate
)

# ───────────────────────────── Base class ────────────────────────────────────
class Scenario:
    """
    Base scenario class representing a plant model.

    Attributes
    ----------
    self.P : ProcessParams
        The process parameters used in this scenario.
    """

    def __init__(self, params: ProcessParams = P_default):
        self.P = params  # use default process parameters unless specified

    def rhs(self, x: np.ndarray, u: float, t: float) -> np.ndarray:
        """
        Continuous-time right-hand side (dynamics function) of the plant.

        Parameters
        ----------
        x : ndarray
            Current state vector [V, ML]
        u : float
            Control input (valve opening)
        t : float
            Current time (unused in nominal model)

        Returns
        -------
        np.ndarray : derivative [dV/dt, dML/dt]
        """
        return rhs_nom(x, u, self.P)

    def specs_met(self, x: np.ndarray) -> bool:
        """
        Check whether the final product meets purity and lactose specs.

        Returns True if:
        - protein concentration ≥ target (cP_star)
        - lactose concentration ≤ target (cL_star)
        """
        V, ML = x
        cP = self.P.MP / V
        cL = ML / V
        return (cP >= self.P.cP_star) and (cL <= self.P.cL_star)

# ───────────────────────────── Variants ──────────────────────────────────────

class Nominal(Scenario):
    """Perfectly modeled plant (matches the controller’s internal model)."""
    pass


class Tear(Scenario):
    """
    Simulates a membrane “filter-cake tear”:

    • Whenever 30 ≤ c_P ≤ 60 mol m⁻³, the permeate flux is doubled  
      (identical to the legacy disturbance you liked).

    Notes
    -----
    • Protein mass stays constant – only flux is affected.
    • The implementation is numerically safe (avoids V = 0).
    """

    def rhs(self, x: np.ndarray, u: float, t: float) -> np.ndarray:
        # ---------------- unpack & guard against V→0 --------------------
        V, ML = x
        V_safe = max(V, 1e-9)                 # prevent division-by-zero
        cP = self.P.MP / V_safe               # current protein conc.

        # ---------------- tear-modified permeate flux -------------------
        p_nom   = flux_permeate(cP, self.P)   # nominal flux
        factor  = 2.0 if 30.0 <= cP <= 60.0 else 1.0
        p       = p_nom * factor              # boosted flux in band

        # ---------------- mass balances ---------------------------------
        d  = u * p                            # solvent inflow
        cL = ML / V_safe
        cL_p = lactose_permeate_conc(cL, p, self.P)

        dV_dt  = d - p                       # volume change
        dML_dt = -cL_p * p                   # lactose mass change

        return np.array([dV_dt, dML_dt])


class KmMismatch(Scenario):
    """
    Models uncertainty in the mass-transfer coefficient.

    The true `kM_L` is scaled by a factor (e.g., plant is slower/faster).
    This simulates incorrect assumptions in controller modeling.
    """

    def __init__(self, factor: float, params: ProcessParams = P_default):
        adjusted_params = replace(params, kM_L=params.kM_L * factor)
        super().__init__(adjusted_params)

# ─────────────── Structural mismatch: Protein leakage through membrane ──────

class ProteinLeakage(Scenario):
    """
    Simulates protein loss through a defective membrane.

    The permeate now contains protein, modeled using the leakage formula:

        cP,p / cP = β / (1 + (β - 1) · exp(p / (kM_P · A)))

    Attributes
    ----------
    beta : float
        Partition coefficient (default 1.3)
    kM_P : float
        Protein mass-transfer coefficient (default 1e-6 m/s)
    MP_cur : float
        Internally tracked total protein mass (decreases over time)
    """

    def __init__(
        self,
        *,
        beta: float = 1.3,
        kM_P: float = 1e-6,
        params: ProcessParams = P_default,
    ):
        super().__init__(params)
        self.beta = beta
        self.kM_P = kM_P
        self.MP_cur = params.MP  # current (dynamic) protein mass in tank

    def specs_met(self, x: np.ndarray) -> bool:
        """
        Override spec check to use current (possibly reduced) protein mass.
        """
        V, ML = x
        cP = self.MP_cur / V
        cL = ML / V
        return (cP >= self.P.cP_star) and (cL <= self.P.cL_star)

    def rhs(self, x, u, t):
        """
        Plant dynamics with protein leakage modeled.
        """
        V, ML = x
        V_safe = max(V, 1e-6)  # avoid division by zero

        # compute concentrations
        cP = self.MP_cur / V_safe
        cL = ML / V_safe

        # compute permeate flux and water inflow
        p = flux_permeate(cP, self.P)
        d = u * p

        # compute lactose permeate concentration
        cL_p = lactose_permeate_conc(cL, p, self.P)

        # compute protein permeate concentration using leakage formula
        exp_term = np.exp(p / (self.kM_P * self.P.A))
        cP_p = (self.beta * cP) / (1 + (self.beta - 1) * exp_term)

        # compute rate of change
        dV_dt = d - p                      # net volume change
        dML_dt = -cL_p * p                 # lactose loss
        dMP_dt = -cP_p * p                 # protein loss (leaked out)

        # update internal protein tracker
        self.MP_cur += dMP_dt * self.P.dt_ctrl

        return np.array([dV_dt, dML_dt])
