# ───────── src/mpc_time_optimal.py ─────────
import casadi as ca
from constants import (
    dt_ctrl, cP_star, cL_star, cL_max,
    MP                                   # остальные константы берёт rk4_disc
)
# переиспользуем дискретизацию, уже объявленную в mpc.py
from mpc import rk4_disc                 # возвращает F(x,u) за dt_ctrl

# ───────────────── time-optimal MPC ─────────────────
def build_time_optimal_mpc(N: int = 20, rho: float = 2e4):
    """
    N   – prediction horizon (steps of dt_ctrl = 10 min)
    rho – вес штрафа за недостижение cP* / cL*
    """
    F = rk4_disc(dt_ctrl)

    # переменные оптимизации
    X  = ca.SX.sym("X", 2, N + 1)        # [V, ML]
    U  = ca.SX.sym("U",      N)          # 0 ≤ u ≤ 1
    X0 = ca.SX.sym("X0",     2)          # текущие [V, ML] – параметр

    g, lbg, ubg = [], [], []
    J = 0.0                              # критерий: минимизация «времени»

    # начальное состояние
    g += [X[:, 0] - X0]; lbg += [0, 0]; ubg += [0, 0]

    # дискретные шаги
    for k in range(N):
        # динамика
        g += [X[:, k+1] - F(X[:, k], U[k])]
        lbg += [0, 0]; ubg += [0, 0]

        # 0 ≤ u ≤ 1
        g += [U[k]];        lbg += [0.0]; ubg += [1.0]

        # текущая лактоза ≤ cL_max
        cL_k = X[1, k] / X[0, k]
        g += [cL_k];        lbg += [-ca.inf]; ubg += [cL_max]

        # штраф за недостижения (чем раньше – тем меньше J)
        cP_k = MP / X[0, k]
        slackP = ca.fmax(cP_star - cP_k, 0)
        slackL = ca.fmax(cL_k     - cL_star, 0)

        J += 1 + rho * (slackP**2 + slackL**2)

    # терминальное жёсткое требование
    cP_f = MP / X[0, N]
    cL_f = X[1, N] / X[0, N]
    g += [cP_f,          cL_f,          cL_f]
    lbg += [cP_star,        0.0,           0.0]
    ubg += [ca.inf,      cL_star,      cL_max]

    nlp = {
        "f": J,
        "x": ca.vertcat(ca.reshape(X, -1, 1), U),
        "p": X0,
        "g": ca.vertcat(*g)
    }

    solver = ca.nlpsol("solver", "ipopt", nlp,
                       {"ipopt.print_level": 0,
                        "ipopt.max_iter":   3000,
                        "print_time":       False})

    meta = {
        "N": N,
        "Uslice": slice(2*(N+1), None)   # где начинается вектор U в sol['x']
    }
    return solver, meta, lbg, ubg
# ───────────────────────────────────────────
