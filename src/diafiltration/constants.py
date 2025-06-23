# ---------- physical data & specs ---------- #
V0      = 0.10      # m³
cP0     = 10.0      # mol/m³
cL0     = 150.0     # mol/m³

k       = 4.79e-3   # m s⁻¹
A       = 1.0       # m²
cg      = 319.0     # mol/m³
kM_L    = 1.6e-5    # m s⁻¹
alpha   = 1.3       # –

cP_star = 100.0
cL_star = 15.0
cL_max  = 570.0     # path constraint

MP      = cP0 * V0  # mol      (conserved)
ML0     = cL0 * V0  # mol

dt_ctrl = 600.0     # s (10 min)
