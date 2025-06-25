import numpy as np
from model import rk4_step
from constants import V0, ML0, MP, cP_star, cL_star, dt_ctrl
from mpc_time_optimal import build_time_optimal_mpc

def closed_loop_time_optimal(N=20, tf=6 * 3600):
    solver, meta, LBG, UBG = build_time_optimal_mpc(N)
    steps = int(tf / dt_ctrl) + 1

    # Выделяем память
    t = np.empty(steps)
    V = np.empty(steps)
    ML = np.empty(steps)
    u_hist = []

    state = np.array([V0, ML0])

    for k in range(steps):
        t[k] = k * dt_ctrl
        V[k], ML[k] = state

        # Прерывание при достижении цели
        if MP / V[k] >= cP_star and ML[k] / V[k] <= cL_star:
            return t[:k + 1], V[:k + 1], ML[:k + 1], np.array(u_hist)

        # Инициализация переменных
        x_init = np.tile(state, meta['N'] + 1)
        u_init = np.ones(meta['N'])
        var_init = np.hstack([x_init, u_init])

        # Решение задачи оптимизации
        sol = solver(x0=var_init, p=state, lbg=LBG, ubg=UBG)
        opt = sol['x'].full().ravel()
        u_now = np.clip(opt[meta['Uslice']][0], 0, 1)

        # Сохраняем и применяем управление
        u_hist.append(u_now)
        state = rk4_step(state, u_now, dt_ctrl)

    return t, V, ML, np.array(u_hist)
