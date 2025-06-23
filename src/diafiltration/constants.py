"""
Physical data & target specs – identical to diafiltration_project.ipynb
"""

# geometry / transport
A     = 1.0          # m²
k     = 4.79e-6      # m s⁻¹   ← NOTE the 10⁻⁶ !
cg    = 319.0        # mol m⁻³
kM_L  = 1.6e-5       # m s⁻¹
alpha = 1.3

# initial inventory
V0   = 0.10          # m³
cP0  = 10.0          # mol m⁻³
cL0  = 150.0         # mol m⁻³
MP   = cP0 * V0      # mol (constant)
ML0  = cL0 * V0      # mol

# controller
dt_ctrl = 600.0      # s (10 min)

# specs / constraints
cP_star = 100.0      # mol m⁻³
cL_star =  15.0      # mol m⁻³
cL_max  = 570.0      # mol m⁻³
