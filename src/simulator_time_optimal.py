import numpy as np
from model import rk4_step
from constants import V0, ML0, MP, cP_star, cL_star, dt_ctrl
from mpc_time_optimal import build_time_optimal_mpc


def closed_loop_time_optimal(N: int = 20, tf: float = 6 * 3600):
    """Run time-optimal MPC until BOTH targets are met.

    Stops immediately after the first control interval in which  
    cP ≥ cP* **and** cL ≤ cL* hold (no extra ‘flat-line’ tail).
    """
    solver, meta, LBG, UBG = build_time_optimal_mpc(N)
    steps = int(tf / dt_ctrl) + 1

    # pre-allocate
    t  = np.empty(steps)
    V  = np.empty(steps)
    ML = np.empty(steps)
    u_hist = []

    state = np.array([V0, ML0])

    for k in range(steps):
        t[k] = k * dt_ctrl
        V[k], ML[k] = state

        # ❶ If the **current** state already meets the spec — exit
        if MP / V[k] >= cP_star and ML[k] / V[k] <= cL_star:
            return t[:k + 1], V[:k + 1], ML[:k + 1], np.array(u_hist)

        # ---- MPC optimisation -------------------------------------------------
        x_init  = np.tile(state, meta["N"] + 1)
        u_init  = np.ones(meta["N"])          # start from u = 1
        var_init = np.hstack([x_init, u_init])

        sol  = solver(x0=var_init, p=state, lbg=LBG, ubg=UBG)
        opt  = sol["x"].full().ravel()
        u_now = float(np.clip(opt[meta["Uslice"]][0], 0, 1))

        # apply control for one interval
        u_hist.append(u_now)
        state = rk4_step(state, u_now, dt_ctrl)

        # ❷ Check **after** applying the move; if met –– append *this* point & quit
        if MP / state[0] >= cP_star and state[1] / state[0] <= cL_star:
            # log the final point
            k_next = k + 1
            if k_next < steps:               # in practise всегда true
                t[k_next]  = (k + 1) * dt_ctrl
                V[k_next]  = state[0]
                ML[k_next] = state[1]
            return t[:k_next + 1], V[:k_next + 1], ML[:k_next + 1], np.array(u_hist)

    # fallback – specs not reached within tf
    return t, V, ML, np.array(u_hist)
