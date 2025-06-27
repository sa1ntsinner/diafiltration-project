"""
Generic MPC builder for the diafiltration benchmark
────────────────────────────────────────────────────
call patterns:
    build_mpc(20)
    build_mpc("time_opt", 25)
    build_mpc(mode="spec", horizon=30)
"""
from __future__ import annotations

import casadi as ca
import numpy as np
from typing import Literal, Optional

from core.params    import default as P_default
from core.dynamics  import casadi_rhs
from core.discretise import rk4_disc

_Mode = Literal["spec", "time_opt"]

# ---------- default weights ---------- #
_W = dict(
    # shared
    rho_u        = 1e-3,
    rho_V        = 1e-2,
    # spec-tracking
    rho_spec     = 1e4,
    rho_time     = 0.10,     # ← increased from 0.001
    rho_u_spec   = 5e-3,
    # time-optimal
    rho_L        = 1e4,
    rho_P        = 1e3,
    k_exp        = 40.0,
    exp_cap      = 50.0,
)




# --------------------------------------------------------------------------- #
def build_mpc(
    *args,
    mode: _Mode = "spec",
    horizon: Optional[int] = None,
    params = P_default,
    weights: dict | None = None,
):
    # -------- back-compat positional parsing ------------------------------ #
    if args:
        if len(args) == 1 and isinstance(args[0], int):
            horizon, = args
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], int):
            mode, horizon = args      # type: ignore[assignment]
        else:
            raise TypeError("build_mpc: use (N) or (mode, N) or keyword args")

    horizon = horizon or 20
    w = {**_W, **(weights or {})}

    # -------- discrete model ---------------------------------------------- #
    F = rk4_disc(casadi_rhs(params), params.dt_ctrl)

    # -------- decision variables ------------------------------------------ #
    X  = ca.SX.sym("X", 2, horizon + 1)   # states
    U  = ca.SX.sym("U", horizon)          # inputs
    X0 = ca.SX.sym("X0", 2)               # parameter = current state

    g, lbg, ubg = [], [], []
    J = 0.0

    # initial equality
    g += [X[:, 0] - X0] ; lbg += [0, 0] ; ubg += [0, 0]

    # -------- horizon loop ------------------------------------------------- #
    for k in range(horizon):

        # model step
        g += [X[:, k + 1] - F(X[:, k], U[k])]
        lbg += [0, 0] ; ubg += [0, 0]

        # input bounds
        g += [U[k]] ; lbg += [0.0] ; ubg += [1.0]

        # concentrations
        V_k, ML_k = X[0, k], X[1, k]
        cP_k = params.MP / V_k
        cL_k = ML_k  / V_k

        # -------- new protein upper-bound along horizon ------------------- #
        g += [cP_k] ; lbg += [0.0] ; ubg += [params.cP_star]

        # lactose upper-bound
        g += [cL_k] ; lbg += [-ca.inf] ; ubg += [params.cL_max]

        # -------- stage-cost ---------------------------------------------- #
        if mode == "spec":
            sL = ca.fmax(cL_k - params.cL_star, 0)
            sP = ca.fmax(params.cP_star - cP_k, 0)
            J += w["rho_spec"] * (sL**2 + sP**2)
            J += w["rho_time"] * params.dt_ctrl            # ↑ much heavier
            J += w["rho_u_spec"] * (1 - U[k])**2
            J += w["rho_V"] * V_k * params.dt_ctrl


        elif mode == "time_opt":
            # ── slack variables ------------------------------------------------- #
            sL         = ca.fmax(cL_k - params.cL_star, 0)          # lactose overshoot
            sP_deficit = ca.fmax(params.cP_star - cP_k, 0)          # protein still low

            # ── indicator: 1 while specs NOT yet met, 0 afterwards -------------- #
            done_k = ca.fmax(ca.sign(sL + sP_deficit), 0)

            # ①  pure clock cost (always active)
            J += done_k * params.dt_ctrl

            # ②  heavy lactose penalty while not done
            J += done_k * w["rho_L"] * ca.exp(
                    ca.fmin(w["k_exp"] * sL, w["exp_cap"])
                ) * params.dt_ctrl

            # ③  protein overshoot (cP too high) only matters until done
            J += done_k * w["rho_P"] * ca.fmax(cP_k - params.cP_star, 0)**2 * params.dt_ctrl

            # ④  encourage high-u while chasing the specs
            J += done_k * w["rho_u"] * (1 - U[k])**2

            # ⑤  discourage excess volume until complete
            J += done_k * w["rho_V"] * V_k * params.dt_ctrl


    # -------- terminal specs ---------------------------------------------- #
    V_N, ML_N = X[0, -1], X[1, -1]
    cP_N = params.MP / V_N
    cL_N = ML_N  / V_N
    eps  = 1e-1
    g += [cP_N, cL_N]
    lbg += [params.cP_star - eps, 0.0]
    ubg += [params.cP_star,       params.cL_star]

    # -------- NLP build ---------------------------------------------------- #
    nlp = dict(
        f = J,
        x = ca.vertcat(ca.reshape(X, -1, 1), U),
        p = X0,
        g = ca.vertcat(*g),
    )
    solver = ca.nlpsol(
        "solver", "ipopt", nlp,
        {"ipopt.print_level": 0, "print_time": False},
    )

    meta = dict(
        N       = horizon,
        Uslice  = slice(2 * (horizon + 1), None),
        u_init  = 0.5 * np.ones(horizon),
    )

    return solver, meta, np.array(lbg), np.array(ubg)
