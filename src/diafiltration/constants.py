"""
Physical data & target specs (exactly as in diafiltration_project.ipynb)
"""

# ─── geometry / transport ───────────────────────────────────────────────
A       = 1.0            # membrane area               [m²]
k       = 4.79e-6        # hydraulic coefficient       [m s⁻¹]   <-- notebook value
kM_L    = 2.0e-6         # mass-transfer coeff. lactose[m s⁻¹]

# ─── concentrations / masses ────────────────────────────────────────────
cg      = 10.0           # gel concentration protein   [mol m⁻³]
MP      = 100.0          # protein mass (constant)     [mol]

V0      = 0.10           # initial volume              [m³]
cL0     = 270.0          # initial lactose concentration [mol m⁻³]
ML0     = V0 * cL0       # initial lactose mass        [mol]

alpha   = 0.8            # partition factor lactose

# ─── controller sampling ────────────────────────────────────────────────
dt_ctrl = 600.0          # 10 min sample time          [s]

# ─── specifications ─────────────────────────────────────────────────────
cP_star = 100.0          # target protein conc.        [mol m⁻³]
cL_star =  15.0          # target lactose conc.        [mol m⁻³]
cL_max  = 570.0          # path constraint lactose     [mol m⁻³]
