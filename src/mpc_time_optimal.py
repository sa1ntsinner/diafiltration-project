import casadi as ca
import numpy as np
from constants import dt_ctrl, cP_star, cL_star, cL_max, MP

def build_time_optimal_mpc(N=20):
    # Состояния и управление
    V, ML = ca.MX.sym("V"), ca.MX.sym("ML")
    x = ca.vertcat(V, ML)
    u = ca.MX.sym("u")
    f = lambda x, u: x + ca.vertcat(-u * dt_ctrl, -u * dt_ctrl * (x[1] / x[0]))

    X = ca.MX.sym("X", 2, N + 1)
    U = ca.MX.sym("U", N)
    X0 = ca.MX.sym("X0", 2)

    obj = 0
    g = []
    lbg = []
    ubg = []

    for k in range(N):
        xk = X[:, k]
        uk = U[k]
        x_next = X[:, k + 1]

        # Динамика
        g.append(x_next - f(xk, uk))
        lbg += [0, 0]
        ubg += [0, 0]

        # Ограничения на u
        g.append(uk)
        lbg.append(0.0)
        ubg.append(1.0)

        # Минимизируем общее время за счёт больших u
        obj += 1 - uk

    # Финальные ограничения: cP ≥ cP_star, cL ≤ cL_star, cL ≤ cL_max
    V_end = X[0, -1]
    ML_end = X[1, -1]
    cP_end = MP / V_end
    cL_end = ML_end / V_end

    g += [cP_end, cL_end, cL_end]
    lbg += [cP_star, 0.0, 0.0]
    ubg += [ca.inf, cL_star, cL_max]

    # NLP solver
    prob = {
        "f": obj,
        "x": ca.vertcat(ca.reshape(X, -1, 1), U),
        "p": X0,
        "g": ca.vertcat(*g)
    }

    opts = {
        "ipopt": {"print_level": 0, "tol": 1e-4, "max_iter": 3000},
        "print_time": False
    }

    solver = ca.nlpsol("solver", "ipopt", prob, opts)
    meta = {"N": N, "Uslice": slice(2 * (N + 1), None)}
    return solver, meta, lbg, ubg
