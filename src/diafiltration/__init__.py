"""
Diafiltration – Time-Optimal MPC  (core package export)

Typical use:
    import diafiltration as df
    t, V, ML, u = df.closed_loop()
"""
from .constants  import *                               # noqa: F401, F403
from .model      import (flux_permeate,
                         lactose_permeate_conc,
                         rk4_step, rhs, casadi_rhs)     # noqa: F401
from .mpc        import build_mpc                       # noqa: F401
from .simulator  import closed_loop                     # noqa: F401
