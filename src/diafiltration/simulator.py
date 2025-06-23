import numpy as np
from .constants   import *
from .model       import rk4_step
from .mpc         import build_mpc

def closed_loop(N=20, u_init=0.5, max_steps=300):
    solver, meta, LBG, UBG = build_mpc(N)
    state = np.array([V0, ML0])

    t=[]; V=[]; ML=[]; u_hist=[]
    for step in range(max_steps):
        t.append(step*dt_ctrl)
        V.append(state[0]); ML.append(state[1])

        if MP/state[0] >= cP_star and ML[-1]/V[-1] <= cL_star:
            break

        x0 = np.hstack([np.tile(state, meta["N"]+1),
                        u_init*np.ones(meta["N"])])
        sol = solver(x0=x0, p=state, lbg=LBG, ubg=UBG)
        try:
            u_now = float(np.clip(sol["x"].full()
                                  [meta["Uslice"]][0], 0, 1))
        except Exception:
            u_now = 1.0
        u_hist.append(u_now)
        state = rk4_step(state, u_now)
    else:
        raise RuntimeError("closed_loop() hit max_steps")

    return np.array(t), np.array(V), np.array(ML), np.array(u_hist)
