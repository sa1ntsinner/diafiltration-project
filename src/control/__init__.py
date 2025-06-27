from .robust import build_robust_mpc  # <- new public symbol
from core.params import default as P_default

def mpc_robust(N=20, *, params=P_default):
    """Tube-based robust MPC (spec mode)."""
    solver, meta, LBG, UBG = build_robust_mpc(horizon=N, params=params)
    # wrap solver into a callable identical to mpc_spec/mpc_time_opt
    def _ctrl(state):
        import numpy as np
        x0 = np.hstack([np.tile(state, meta["N"] + 1), meta["u_init"]])
        sol = solver(x0=x0, p=state, lbg=LBG, ubg=UBG)
        return float(sol["x"].full().ravel()[meta["Uslice"]][0])
    return _ctrl
