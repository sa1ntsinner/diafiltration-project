# Initial Conditions
V0      = 0.10      # Initial volume [m^3]
cP0     = 10.0      # Initial protein concentration [mol/m^3]
cL0     = 150.0     # Initial lactose concentration [mol/m^3]

# Derived initial masses
MP      = cP0 * V0  # Initial protein mass [mol]
ML0     = cL0 * V0  # Initial lactose mass [mol]

# Membrane and process constants
k       = 4.79e-6   # Permeate rate constant [m s^-2]
A       = 1.0       # Membrane area [m^2]
cg      = 319.0     # Gel concentration [mol/m^3]
kM_L    = 1.6e-5    # Mass transfer coefficient for lactose [m s^-1]
alpha   = 1.3       # Rejection coefficient

# Targets and constraints
cP_star = 100.0     # Target protein concentration [mol/m^3]
cL_star = 15.0      # Target lactose concentration [mol/m^3]
cL_max  = 570.0     # Max allowed lactose concentration [mol/m^3]

# Control settings
dt_ctrl = 600.0     # Control interval [s] (10 min)
t_final = 6 * 3600  # Total simulation time [s] (6 hours)
