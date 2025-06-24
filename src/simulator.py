import numpy as np
from model import rk4_step
from constants import V0, ML0, MP, cP_star, cL_star, dt_ctrl
from mpc import build_mpc
from constants import V0, ML0, t_final

def closed_loop(N=20, tf=6*3600):
    solver, meta, LBG, UBG = build_mpc(N)
    steps = int(tf/dt_ctrl)+1
    t = np.empty(steps); V = np.empty(steps); ML = np.empty(steps); u_hist = []
    state = np.array([V0, ML0])
    for k in range(steps):
        t[k] = k * dt_ctrl; V[k], ML[k] = state
        if MP/state[0] >= cP_star and ML[k]/V[k] <= cL_star:
            return t[:k+1], V[:k+1], ML[:k+1], np.array(u_hist)
        x_init = np.tile(state, meta['N']+1)
        u_init = 0.5 * np.ones(meta['N'])
        var_init = np.hstack([x_init, u_init])
        sol = solver(x0=var_init, p=state, lbg=LBG, ubg=UBG)
        opt = sol['x'].full().ravel()
        u_now = np.clip(opt[meta['Uslice']][0], 0, 1)
        u_hist.append(u_now)
        state = rk4_step(state, u_now, dt_ctrl)
    return t, V, ML, np.array(u_hist)

def simulate_open_loop(u: float, T: int = t_final):
    steps = int(T / dt_ctrl) + 1
    t = np.empty(steps)
    V = np.empty(steps)
    ML = np.empty(steps)
    state = np.array([V0, ML0])

    for k in range(steps):
        t[k] = k * dt_ctrl
        V[k], ML[k] = state
        state = rk4_step(state, u, dt_ctrl)

    return t, V, ML