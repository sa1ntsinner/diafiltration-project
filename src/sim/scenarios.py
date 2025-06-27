"""
Scenario classes – each supplies a continuous-time RHS f(x,u,t).

Nominal dynamics come from core.dynamics.rhs; subclasses override rhs()
to inject disturbances, parameter mismatch, or structural changes.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np

from core.params    import ProcessParams
from core.dynamics  import (
    rhs                as rhs_nom,
    flux_permeate,
    lactose_permeate_conc,
)

# ───────────────────────────────── base ──────────────────────────────────────
class Scenario:
    """Base class holding a ProcessParams instance + generic spec check."""
    def __init__(self, params: ProcessParams):
        self.P = params            # immutable dataclass (see core/params.py)

    # continuous dynamics -----------------------------------------------------
    def rhs(self, x: np.ndarray, u: float, t: float) -> np.ndarray:
        """ẋ = f(x,u,t).  Overwrite in subclasses if needed."""
        return rhs_nom(x, u, self.P)

    # batch finished? ---------------------------------------------------------
    def specs_met(self, x: np.ndarray) -> bool:
        V, ML = x
        cP = self.P.MP / V
        cL = ML / V
        return (cP >= self.P.cP_star) and (cL <= self.P.cL_star)


# ─────────────────────────── existing scenarios ─────────────────────────────
class Nominal(Scenario):
    """Exactly the model assumed by the controller – no mismatch."""
    pass


class Tear(Scenario):
    """
    Filter-cake tear: permeate flux doubles while 30 ≤ cP ≤ 60 kg m⁻³.
    """
    def rhs(self, x, u, t):
        V, ML = x
        cP = self.P.MP / max(V, 1e-6)
        scale = 2.0 if (30.0 <= cP <= 60.0) else 1.0

        p  = scale * flux_permeate(cP, self.P)
        d  = u * p
        cL = ML / V
        cL_p = lactose_permeate_conc(cL, p, self.P)
        return np.array([d - p, -cL_p * p])


class KmMismatch(Scenario):
    """
    Plant–model mismatch: kM_L is lower by *factor* (< 1 ⇒ slower lactose
    transfer into permeate).
    """
    def __init__(self, factor: float, params: ProcessParams):
        super().__init__(replace(params, kM_L=params.kM_L * factor))


# ─────────────────────────── new scenario  (Step 4) ─────────────────────────
class ProteinLeakage(Scenario):
    """
    Structural mismatch: a fraction β of protein passes the membrane.
    The retentate protein mass MP becomes time-varying.

    State vector x = [V, ML] (unchanged); we update an *internal* MP tracker
    so that cP = MP_current / V is correct for flux + specs.
    """
    def __init__(self, beta: float, params: ProcessParams):
        super().__init__(params)
        self.beta = beta
        self.MP_current = params.MP      # start with nominal total protein mass

    # override spec check because MP is no longer constant --------------------
    def specs_met(self, x: np.ndarray) -> bool:
        V, ML = x
        cP = self.MP_current / V
        cL = ML / V
        return (cP >= self.P.cP_star) and (cL <= self.P.cL_star)

    # plant dynamics with leakage -------------------------------------------
    def rhs(self, x, u, t):
        V, ML = x
        V_safe = max(V, 1e-6)
        cP = self.MP_current / V_safe

        # permeate flux & dilution
        p  = flux_permeate(cP, self.P)
        d  = u * p

        # lactose transport
        cL = ML / V_safe
        cL_p = lactose_permeate_conc(cL, p, self.P)

        # protein leakage: d(MP_current)/dt = –β * cP * p
        dMP_dt = -self.beta * cP * p
        self.MP_current += dMP_dt * self.P.dt_ctrl   # euler update over Δt

        return np.array([d - p, -cL_p * p])
