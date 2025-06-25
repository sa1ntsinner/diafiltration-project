import casadi as ca
from constants import (
    dt_ctrl, cP_star, cL_star, cL_max, MP
)
from mpc import rk4_disc                       # дискретизация из baseline MPC

def build_time_optimal_mpc(N: int = 40,
                           rho_under: float = 2e4,
                           rho_over : float = 5e3,
                           rho_L    : float = 2e4,
                           w_u      : float = 1e-3):
    """
    N          – горизонт прогнозирования (шаги по 10 мин)
    rho_under  – штраф за недо-достижение cP*
    rho_over   – штраф за «перетоп» (> cP*)
    rho_L      – штраф за превышение cL*
    w_u        – мягкое притяжение к u = 1
    """
    F = rk4_disc(dt_ctrl)

    X  = ca.SX.sym("X", 2, N+1)       # [V, ML]
    U  = ca.SX.sym("U",      N)       # 0 ≤ u ≤ 1
    X0 = ca.SX.sym("X0",     2)       # параметр – текущее состояние

    g, lbg, ubg = [], [], []
    J = 0.0

    # старт
    g += [X[:,0] - X0]; lbg += [0,0]; ubg += [0,0]

    for k in range(N):
        # динамика
        g += [X[:,k+1] - F(X[:,k], U[k])]
        lbg += [0,0]; ubg += [0,0]

        # 0 ≤ u ≤ 1
        g += [U[k]]; lbg += [0.0]; ubg += [1.0]

        # path-constraint: мгновенный потолок лактозы
        cL_k = X[1,k] / X[0,k]
        g += [cL_k]; lbg += [-ca.inf]; ubg += [cL_max]

        # слэки
        cP_k = MP / X[0,k]
        slackP_under = ca.fmax(cP_star - cP_k, 0)
        slackP_over  = ca.fmax(cP_k     - cP_star, 0)
        slackL       = ca.fmax(cL_k     - cL_star, 0)

        # критерий
        J += (1                                   # «время»
              + rho_under * slackP_under**2
              + rho_over  * slackP_over **2
              + rho_L     * slackL      **2
              + w_u       * (1 - U[k])  **2)

    # терминальные жёсткие условия
    cP_f = MP / X[0,-1]
    cL_f = X[1,-1] / X[0,-1]
    g += [cP_f,          cL_f,          cL_f]
    lbg += [cP_star,        0.0,           0.0]
    ubg += [ca.inf,      cL_star,      cL_max]

    nlp = {"f": J,
           "x": ca.vertcat(ca.reshape(X,-1,1), U),
           "p": X0,
           "g": ca.vertcat(*g)}

    solver = ca.nlpsol("solver","ipopt", nlp,
                       {"ipopt.print_level": 0,
                        "ipopt.sb":          "yes",
                        "ipopt.max_iter": 3000,
                        "print_time":     False})

    meta = {"N": N,
            "Uslice": slice(2*(N+1), None)}
    return solver, meta, lbg, ubg
