import casadi as ca
import numpy as np
from constants import dt_ctrl, cP_star, cL_star, cL_max, MP
from mpc import rk4_disc


def build_time_optimal_mpc(N=50,
                           rho_L=8e4,
                           k_exp=40.0,
                           w_u=1e-3,
                           exp_cap=50.0):
    F = rk4_disc(dt_ctrl)

    X  = ca.SX.sym("X", 2, N + 1)
    U  = ca.SX.sym("U",      N)
    X0 = ca.SX.sym("X0",     2)

    g, lbg, ubg = [], [], []
    J, done = 0.0, 0        # `done` – флаг «цели достигнуты»

    g += [X[:, 0] - X0]; lbg += [0, 0]; ubg += [0, 0]

    for k in range(N):
        g += [X[:, k + 1] - F(X[:, k], U[k])]
        lbg += [0, 0]; ubg += [0, 0]

        g += [U[k]]; lbg += [0.0]; ubg += [1.0]

        cL_k = X[1, k] / X[0, k]
        g   += [cL_k]; lbg += [-ca.inf]; ubg += [cL_max]

        cP_k = MP / X[0, k]

        hit  = ca.logic_and(cP_k >= cP_star, cL_k <= cL_star)
        done = ca.logic_or(done, hit)              # один раз взвелись → остаётся 1

        # ❶ «время» только пока done==0
        J += (1 - done) * dt_ctrl/3600             # [ч]

        # ❷ экспонента – интегрально и с огран. аргументом
        slack_L = ca.fmax(cL_k - cL_star, 0)
        expo    = ca.exp(ca.fmin(k_exp * slack_L, exp_cap))
        J      += rho_L * expo * dt_ctrl           # интеграл

        # ❸ мягкое притяжение к u=1, нормированное на N
        J += (w_u / N) * (1 - U[k])**2

    # terminal hard constraints
    cP_f = MP / X[0, -1]
    cL_f = X[1, -1] / X[0, -1]
    g += [cP_f,          cL_f,          cL_f]
    lbg += [cP_star,        0.0,           0.0]
    ubg += [ca.inf,      cL_star,      cL_max]

    nlp = dict(f=J,
               x=ca.vertcat(ca.reshape(X, -1, 1), U),
               p=X0,
               g=ca.vertcat(*g))

    opts = dict(ipopt=dict(print_level=0, max_iter=3000, sb="yes"),
                print_time=False)

    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    meta = dict(N=N,
                Uslice=slice(2 * (N + 1), None),
                u_init=0.5 * np.ones(N))

    return solver, meta, np.array(lbg), np.array(ubg)
