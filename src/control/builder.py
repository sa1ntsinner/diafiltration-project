"""
control/builder.py
──────────────────
Factory that constructs an MPC solver for the diafiltration benchmark.

Usage patterns
--------------
    build_mpc(20)                 → quadratic spec-tracking MPC
    build_mpc("econ", 25)         → linear / economic MPC
    build_mpc("time_opt", 30)     → *economical* time-optimal MPC
"""

from __future__ import annotations

import casadi as ca
import numpy as np
from typing import Literal, Optional

from core.params     import default as P_default         # nominal constants
from core.dynamics   import casadi_rhs                   # continuous RHS
from core.discretise import rk4_disc                     # RK4 discretiser
from core.tariff     import lambda_tou                   # TOU tariff (€/kWh)

# --------------------------------------------------------------------------- #
# 0.  Type alias for the mode selector                                        #
# --------------------------------------------------------------------------- #
_Mode = Literal["spec", "econ", "time_opt"]

# --------------------------------------------------------------------------- #
# 1.  Default weights for the different objective terms                       #
# --------------------------------------------------------------------------- #
_W = dict(
    # – shared small penalties –
    rho_u      = 1e-3,    # input smoothing
    rho_V      = 1e-2,    # discourage large volumes

    # – spec-tracking (quadratic) –
    rho_spec   = 1e4,
    rho_time   = 0.10,    # tiny “clock” cost
    rho_u_spec = 5e-3,

    # – economic (linear) –
    rho_L_lin  = 50.0,
    rho_P_lin  = 50.0,

    # – legacy weights (kept for backward compatibility; not used here) –
    rho_L      = 1e4,
    rho_P      = 1e3,
    k_exp      = 40.0,
    exp_cap    = 50.0,

    # – optional energy add-on –
    rho_energy = 1.0,     # multiplies λ(t) ⋅ u ⋅ Δt   if lambda_fun supplied
)


# --------------------------------------------------------------------------- #
# 2.  Main factory function                                                   #
# --------------------------------------------------------------------------- #
def build_mpc(
    *args,
    mode: _Mode = "spec",
    horizon: Optional[int] = None,
    params=P_default,
    weights: Optional[dict] = None,
):
    """
    Build a CasADi NLP solver for the chosen MPC flavour.

    Parameters
    ----------
    mode     : {"spec", "econ", "time_opt"}
        Objective formulation (see top-level docstring).
    horizon  : int
        Prediction horizon N (number of control intervals).
        If omitted defaults to 20.
    params   : core.params.ProcessParams
        Physical / process constants (default = nominal set).
    weights  : dict, optional
        Override any entry of the global _W dictionary.  This can also
        include a key ``"lambda_fun"`` mapping *t [s]* → tariff [€/kWh].

    Returns
    -------
    solver : casadi.nlpsol
    meta   : dict   – contains {"N", "Uslice", "u_init"}
    LBG    : np.ndarray – lower bounds for g
    UBG    : np.ndarray – upper bounds for g
    """
    # --------------------------------------------------------------------- #
    # 2.1  Parse legacy positional syntax                                   #
    #       e.g. build_mpc(25) or build_mpc("econ", 30)                     #
    # --------------------------------------------------------------------- #
    if args:
        if len(args) == 1 and isinstance(args[0], int):
            horizon, = args
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], int):
            mode, horizon = args        # type: ignore[assignment]
        else:
            raise TypeError("build_mpc: use (N), (mode, N) or keyword arguments")

    horizon = horizon or 20                     # default horizon
    w = {**_W, **(weights or {})}               # merge weight overrides

    # --------------------------------------------------------------------- #
    # 2.2  Obtain discrete-time model  x_{k+1} = F(x_k, u_k)                #
    # --------------------------------------------------------------------- #
    F = rk4_disc(casadi_rhs(params), params.dt_ctrl)

    # --------------------------------------------------------------------- #
    # 2.3  Declare decision variables                                       #
    #       X : 2 × (N+1)  – [V, ML] trajectory                             #
    #       U : N          – valve openings in [0, 1]                       #
    # --------------------------------------------------------------------- #
    X  = ca.SX.sym("X", 2, horizon + 1)
    U  = ca.SX.sym("U", horizon)
    X0 = ca.SX.sym("X0", 2)                      # current state parameter

    # Containers for constraints and objective
    g, lbg, ubg = [], [], []
    J = 0.0

    # --------------------------------------------------------------------- #
    # 2.4  Initial-state equality                                           #
    # --------------------------------------------------------------------- #
    g   += [X[:, 0] - X0]
    lbg += [0, 0]
    ubg += [0, 0]

    # --------------------------------------------------------------------- #
    # 2.5  Horizon loop                                                     #
    # --------------------------------------------------------------------- #
    for k in range(horizon):
        # ----  dynamics -------------------------------------------------- #
        g   += [X[:, k + 1] - F(X[:, k], U[k])]
        lbg += [0, 0]
        ubg += [0, 0]

        # ----  input bounds  0 ≤ u ≤ 1 ----------------------------------- #
        g   += [U[k]]
        lbg += [0.0]
        ubg += [1.0]

        # ----  extract current concentrations ---------------------------- #
        V_k, ML_k = X[0, k], X[1, k]
        cP_k      = params.MP / V_k
        cL_k      = ML_k / V_k

        # ----  hard path constraints ------------------------------------- #
        g   += [cP_k]                # protein must stay ≤ target
        lbg += [0.0]
        ubg += [params.cP_star]

        g   += [cL_k]                # lactose must stay ≤ max allowed
        lbg += [-ca.inf]
        ubg += [params.cL_max]

        # ----  stage cost per mode --------------------------------------- #
        if mode == "spec":
            # • quadratic slack tracking
            sL = ca.fmax(cL_k - params.cL_star, 0)
            sP = ca.fmax(params.cP_star - cP_k, 0)
            J += (sL**2 + sP**2)
            # J += w["rho_time"] * params.dt_ctrl
            # J += w["rho_u_spec"] * (1 - U[k])**2
            # J += w["rho_V"] * V_k * params.dt_ctrl

        elif mode == "econ":
            # • linear “economic” formulation (no squares)
            sL = ca.fmax(cL_k - params.cL_star, 0)
            sP = ca.fmax(params.cP_star - cP_k, 0)
            J += params.dt_ctrl                        # clock term (1 × Δt)
            J += w["rho_L_lin"] * sL * params.dt_ctrl
            J += w["rho_P_lin"] * sP * params.dt_ctrl
            J += w["rho_u"] * (1 - U[k])**2            # light smoothing

        elif mode == "time_opt":
            # • new “economical” time-optimal objective (suggested by user)
            #   maximise Σ cP   –  minimise Σ Δt   –  penalise lactose overshoot
            J += -cP_k                               # maximise protein yield
            J += 10.0 * params.dt_ctrl              # heavy clock cost
            J += ca.fmax(cL_k - params.cL_max, 0)   # lactose penalty
            J += w["rho_u"] * (1 - U[k])**2         # keep some smoothness

        else:
            raise ValueError(f"Unknown mode '{mode}'")

        # ---- optional TOU-energy term (used by spec/econ) --------------- #
        if "lambda_fun" in w:
            lam_k = w["lambda_fun"](k * params.dt_ctrl)  # €/kWh at this time
            J += w["rho_energy"] * lam_k * U[k] * params.dt_ctrl

    # --------------------------------------------------------------------- #
    # 2.6  Terminal specification (soft equality)                           #
    # --------------------------------------------------------------------- #
    V_N, ML_N = X[0, -1], X[1, -1]
    cP_N      = params.MP / V_N
    cL_N      = ML_N / V_N

    eps = 1e-1                                    # small slack for cP lower bound
    g   += [cP_N, cL_N]
    lbg += [params.cP_star - eps, 0.0]
    ubg += [params.cP_star,        params.cL_star]

    # --------------------------------------------------------------------- #
    # 2.7  Create CasADi NLP solver (IPOPT backend)                          #
    # --------------------------------------------------------------------- #
    nlp = dict(
        f = J,
        x = ca.vertcat(ca.reshape(X, -1, 1), U),  # decision vector
        p = X0,                                   # parameter = current state
        g = ca.vertcat(*g),                       # constraints
    )
    solver = ca.nlpsol(
        "solver", "ipopt", nlp,
        {"ipopt.print_level": 0, "print_time": False},
    )

    # --------------------------------------------------------------------- #
    # 2.8  Helper metadata for the caller                                   #
    # --------------------------------------------------------------------- #
    meta = dict(
        N       = horizon,
        Uslice  = slice(2 * (horizon + 1), None),   # slice that extracts U from x
        u_init  = np.ones(horizon) if mode == "time_opt"
                  else 0.5 * np.ones(horizon),
    )

    # Return everything the higher-level code needs
    return solver, meta, np.array(lbg), np.array(ubg)
