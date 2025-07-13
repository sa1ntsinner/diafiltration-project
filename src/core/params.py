"""
core/params.py
──────────────────
Defines the process parameters and initial conditions for the diafiltration model.

These are used throughout the simulation and MPC codebase via a single
immutable instance: `default = ProcessParams()`.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class ProcessParams:
    # ── Initial state of the system ─────────────────────────────────────
    V0:     float = 0.10      # Initial volume of liquid in tank [m³]
    cP0:    float = 10.0      # Initial concentration of protein [kg/m³]
    cL0:    float = 150.0     # Initial concentration of lactose [kg/m³]

    # ── Transport and geometry parameters ──────────────────────────────
    k:      float = 4.79e-6   # Permeability constant [m/s]
    A:      float = 1.0       # Membrane surface area [m²]
    cg:     float = 319.0     # Protein concentration at gel layer [kg/m³]

    # ── Mass-transfer and retention characteristics ────────────────────
    kM_L:   float = 1.6e-5    # Lactose mass-transfer coefficient [m/s]
    alpha:  float = 1.3       # Sieving coefficient (0 < α ≤ 1); higher = more leakage

    # ── Target specs and constraints ───────────────────────────────────
    cP_star: float = 100.0    # Target protein concentration [kg/m³]
    cL_star: float = 15.0     # Target lactose concentration [kg/m³]
    cL_max:  float = 570.0    # Max allowed lactose concentration [kg/m³]

    # ── Controller configuration ───────────────────────────────────────
    dt_ctrl: float = 600.0    # Control interval (sampling time) [s]
    t_final: float = 6 * 3600 # Total simulation time = 6 hours [s]

    # ── Derived parameters (automatically computed) ────────────────────
    @property
    def MP(self) -> float:
        """
        Total mass of protein in the tank [kg], assumed constant.
        MP = cP0 × V0
        """
        return self.cP0 * self.V0

    @property
    def ML0(self) -> float:
        """
        Initial total mass of lactose [kg].
        ML0 = cL0 × V0
        """
        return self.cL0 * self.V0


# A default instance shared across all modules unless explicitly overridden.
default = ProcessParams()
