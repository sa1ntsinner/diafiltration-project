import numpy as np
from model import rk4_step
from constants import V0, ML0, MP, cP_star, cL_star, cL_max, dt_ctrl, t_final
from mpc import build_mpc
from policies import threshold_policy


def closed_loop(N=20, tf=t_final):
    """
    Run closed-loop MPC simulation with given prediction horizon N.
    Stops early if cP ≥ cP* and cL ≤ cL*.
    """
    solver, meta, LBG, UBG = build_mpc(N)
    steps = int(tf / dt_ctrl) + 1
    t = np.empty(steps)
    V = np.empty(steps)
    ML = np.empty(steps)
    u_hist = []
    state = np.array([V0, ML0])

    for k in range(steps):
        t[k] = k * dt_ctrl
        V[k], ML[k] = state

        # Stop early if both targets are met
        cP = MP / V[k]
        cL = ML[k] / V[k]
        if cP >= cP_star and cL <= cL_star:
            return t[:k + 1], V[:k + 1], ML[:k + 1], np.array(u_hist)

        # MPC optimization
        x_init = np.tile(state, meta['N'] + 1)
        u_init = 0.5 * np.ones(meta['N'])
        var_init = np.hstack([x_init, u_init])
        sol = solver(x0=var_init, p=state, lbg=LBG, ubg=UBG)
        opt = sol['x'].full().ravel()
        u_now = np.clip(opt[meta['Uslice']][0], 0, 1)
        u_hist.append(u_now)
        state = rk4_step(state, u_now, dt_ctrl)

    return t, V, ML, np.array(u_hist)


def closed_loop_threshold(tf=t_final):
    """
    Simulate closed-loop using a simple threshold-based policy.
    Stops early if cP ≥ cP* and cL ≤ cL*.
    """
    steps = int(tf / dt_ctrl) + 1
    t = np.empty(steps)
    V = np.empty(steps)
    ML = np.empty(steps)
    u_hist = []
    state = np.array([V0, ML0])

    for k in range(steps):
        t[k] = k * dt_ctrl
        V[k], ML[k] = state

        # Stop early if both targets are met
        cP = MP / V[k]
        cL = ML[k] / V[k]
        if cP >= cP_star and cL <= cL_star:
            return t[:k + 1], V[:k + 1], ML[:k + 1], np.array(u_hist)

        u = threshold_policy(state)
        u_hist.append(u)
        state = rk4_step(state, u, dt_ctrl)

    return t, V, ML, np.array(u_hist)


def simulate_open_loop(u: float, T: int = t_final):
    """
    Simulate open-loop diafiltration process with constant control u.
    """
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
