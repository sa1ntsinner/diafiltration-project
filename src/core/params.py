from dataclasses import dataclass

@dataclass(frozen=True)
class ProcessParams:
    # ── Initial state ────────────────────────────────────────────────────
    V0:     float = 0.10      # [m³]
    cP0:    float = 10.0      # [kg m⁻³]
    cL0:    float = 150.0     # [kg m⁻³]

    # ── Transport & geometry ────────────────────────────────────────────
    k:      float = 4.79e-6   # [m s⁻¹]
    A:      float = 1.0       # [m²]
    cg:     float = 319.0     # [kg m⁻³]

    # ── Mass-transfer & retention ───────────────────────────────────────
    kM_L:   float = 1.6e-5    # [m s⁻¹]
    alpha:  float = 1.3       # [–] (relative sieving)

    # ── Specs / constraints ────────────────────────────────────────────
    cP_star: float = 100.0    # [kg m⁻³]
    cL_star: float = 15.0     # [kg m⁻³]
    cL_max:  float = 570.0    # [kg m⁻³]

    # ── Controller grid ────────────────────────────────────────────────
    dt_ctrl: float = 600.0             # s
    t_final: float = 6 * 3600          # s (6 h)

    # ── Derived (read-only) ────────────────────────────────────────────
    @property
    def MP(self) -> float:             # total protein mass in tank [kg]
        return self.cP0 * self.V0

    @property
    def ML0(self) -> float:            # initial lactose mass [kg]
        return self.cL0 * self.V0


# A single immutable default instance that the whole codebase can reuse
default = ProcessParams()
