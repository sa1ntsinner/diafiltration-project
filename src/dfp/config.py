"""
dfp.config
==========
Single source of truth for every physical constant, specification and
numerical setting of the batch-diafiltration benchmark
(*P2: Time-Optimal Control of a Diafiltration Process*, APC SS25, TU Dortmund).

Units
-----
Strictly SI, with **molar** concentrations exactly as in the task sheet:

===========  ==========================  ============
symbol       meaning                     unit
===========  ==========================  ============
``V``        retentate volume            m³
``ML``       lactose hold-up             mol
``MP``       protein hold-up             mol
``cP, cL``   concentrations              mol m⁻³
``p``        permeate *volumetric* flow  m³ s⁻¹
``u``        d/p ratio (manipulated)     –
===========  ==========================  ============

.. note::
   The task sheet prints the permeation coefficient as ``4.79e-6 m s^-2``.
   That is a typo: for :math:`p = kA\\ln(c_g/c_P)` to be a volumetric flow
   (m³ s⁻¹) with ``A`` in m², ``k`` must carry ``m s⁻¹``.  The numerical
   value is used unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np

__all__ = ["ProcessParams", "NOMINAL", "UncertaintySet", "KM_L_UNCERTAINTY"]


# ─────────────────────────────────────────────────────────────────────────────
#  Process parameters
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ProcessParams:
    """Immutable container of every constant of the diafiltration benchmark."""

    # ── membrane / transport ────────────────────────────────────────────────
    k: float = 4.79e-6      #: permeation coefficient                 [m s⁻¹]
    A: float = 1.0          #: membrane area                          [m²]
    cg: float = 319.0       #: gel-layer protein concentration        [mol m⁻³]
    kM_L: float = 1.6e-5    #: lactose mass-transfer coefficient      [m s⁻¹]
    alpha: float = 1.3      #: lactose partition function  (α ≥ 1)    [–]

    # ── structural extension: partial protein passage (Eq. 6) ──────────────
    #   β = protein partition function, kM_P its mass-transfer coefficient.
    #   The *nominal* plant retains protein completely; that case is selected
    #   by the ``protein_leakage`` build flag of :func:`dfp.model.build_model`,
    #   not by β, because β = 0 is outside the validity range of Eq. (2)/(6).
    beta: float = 1.3       #: protein partition function  (β ≥ 1)    [–]
    kM_P: float = 1.0e-6    #: protein mass-transfer coefficient      [m s⁻¹]

    # ── initial charge ─────────────────────────────────────────────────────
    V0: float = 0.10        #: initial volume  (100 L)                [m³]
    cP0: float = 10.0       #: initial protein concentration          [mol m⁻³]
    cL0: float = 150.0      #: initial lactose concentration          [mol m⁻³]

    # ── product specification ──────────────────────────────────────────────
    cP_f: float = 100.0     #: required final protein concentration   [mol m⁻³]
    cL_f: float = 15.0      #: maximum final lactose concentration    [mol m⁻³]
    cL_max: float = 570.0   #: crystallisation limit (path constraint)[mol m⁻³]

    # ── operation / numerics ───────────────────────────────────────────────
    dt_ctrl: float = 600.0  #: control interval Δt = 10 min           [s]
    t_max: float = 6 * 3600  #: simulation horizon                    [s]
    u_min: float = 0.0      #: valve lower bound
    u_max: float = 1.0      #: valve upper bound
    n_sub_plant: int = 120  #: plant sub-steps per control interval (→ 5 s)
    n_sub_mpc: int = 4      #: RK4 sub-steps per MPC shooting interval
    spec_tol: float = 1e-4  #: relative tolerance used to declare "batch done"

    # ── economics (used only by the economic-MPC extension) ────────────────
    pump_idle_kW: float = 0.5   #: stand-by pump power
    pump_dyn_kW: float = 2.5    #: additional power at u = 1

    # ── derived quantities ─────────────────────────────────────────────────
    @property
    def MP0(self) -> float:
        """Initial protein hold-up ``cP0·V0`` [mol]."""
        return self.cP0 * self.V0

    @property
    def ML0(self) -> float:
        """Initial lactose hold-up ``cL0·V0`` [mol]."""
        return self.cL0 * self.V0

    @property
    def x0(self) -> np.ndarray:
        """Initial state ``[V, ML, MP]``."""
        return np.array([self.V0, self.ML0, self.MP0], dtype=float)

    @property
    def V_f(self) -> float:
        """Volume that corresponds to the protein specification [m³]."""
        return self.MP0 / self.cP_f

    @property
    def V_gel(self) -> float:
        """Volume at which ``cP = cg`` and the flux model degenerates [m³]."""
        return self.MP0 / self.cg

    @property
    def theta(self) -> np.ndarray:
        """Uncertain-parameter vector consumed by the symbolic model."""
        return np.array(
            [self.k, self.A, self.cg, self.kM_L, self.alpha, self.beta, self.kM_P],
            dtype=float,
        )

    # ── helpers ────────────────────────────────────────────────────────────
    def with_(self, **kw) -> "ProcessParams":
        """Return a copy with selected fields replaced."""
        return replace(self, **kw)

    def scaled(self, **factors: float) -> "ProcessParams":
        """Return a copy with selected fields *multiplied* by a factor."""
        return replace(self, **{k: getattr(self, k) * f for k, f in factors.items()})

    def __post_init__(self) -> None:  # pragma: no cover - cheap validation
        if self.alpha < 1.0:
            raise ValueError("alpha must be >= 1 for Eq. (2) to stay physical")
        if self.beta < 1.0:
            raise ValueError("beta must be >= 1 for Eq. (6) to stay physical")
        if not 0.0 <= self.u_min < self.u_max <= 1.0:
            raise ValueError("require 0 <= u_min < u_max <= 1")
        if self.cP_f >= self.cg:
            raise ValueError("cP_f must stay below the gel concentration")


#: The nominal plant / controller-model parameter set.
NOMINAL = ProcessParams()


# ─────────────────────────────────────────────────────────────────────────────
#  Uncertainty description (used by multi-stage NMPC and Monte-Carlo)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class UncertaintySet:
    """A finite set of *multiplicative* realisations of model parameters.

    Parameters
    ----------
    factors
        ``{"kM_L": (0.25, 0.5, 1.0), ...}`` – every combination of the listed
        factors becomes one branch of the scenario tree.
    weights
        Optional probability weights, one per realisation.  Uniform if ``None``.
    """

    factors: Dict[str, Tuple[float, ...]]
    weights: Tuple[float, ...] | None = None

    def realisations(self, base: ProcessParams = NOMINAL) -> list[ProcessParams]:
        """Cartesian product of all factor combinations."""
        import itertools

        keys = list(self.factors)
        out: list[ProcessParams] = []
        for combo in itertools.product(*(self.factors[k] for k in keys)):
            out.append(base.scaled(**dict(zip(keys, combo))))
        return out

    def probabilities(self, n: int) -> np.ndarray:
        if self.weights is None:
            return np.full(n, 1.0 / n)
        w = np.asarray(self.weights, dtype=float)
        if w.size != n:
            raise ValueError("weights length does not match number of branches")
        return w / w.sum()


#: Uncertainty in the lactose mass-transfer coefficient, exactly the range
#: requested in the *additional tasks* section of the task sheet.
KM_L_UNCERTAINTY = UncertaintySet(factors={"kM_L": (0.25, 0.5, 0.75, 1.0)})
