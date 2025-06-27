"""
Generic closed-loop / open-loop simulator.

Controller API
--------------
A controller is ANY callable   u = ctrl(state)   where
    state : np.ndarray([V, ML])
    u     : float ∈ [0,1]

We provide helper factories for
* constant-u (open loop)
* threshold heuristic
* MPC (spec-tracking or time-optimal)
"""
from __future__ import annotations
import numpy as np
from typing import Callable
from core.params     import default as P_default
from core.discretise import rk4_step
from control.builder import build_mpc
from sim.scenarios   import Scenario

# --------------------------------------------------------------------------- #
def simulate(
    controller: Callable[[np.ndarray], float],
    scenario  : Scenario,
    tf        : float | None = None,
    dt        : float | None = None,
):
    P   = scenario.P
    tf  = tf or P.t_final
    dt  = dt or P.dt_ctrl
    steps = int(tf / dt) + 1

    t = np.empty(steps)
    V = np.empty(steps)
    ML = np.empty(steps)
    u_hist = []

    x = np.array([P.V0, P.ML0])

    for k in range(steps):
        t[k]  = k * dt
        V[k], ML[k] = x

        if scenario.specs_met(x):
            return t[:k+1], V[:k+1], ML[:k+1], np.array(u_hist)

        u = float(np.clip(controller(x), 0.0, 1.0))
        u_hist.append(u)
        x = rk4_step(x, u, dt, lambda s, uu: scenario.rhs(s, uu, t[k]))

    return t, V, ML, np.array(u_hist)


# --------------------------------------------------------------------------- #
# Helper controller factories
# --------------------------------------------------------------------------- #
def constant_u(u_val: float) -> Callable[[np.ndarray], float]:
    return lambda _: u_val


def threshold_policy(threshold: float = 55.0, u_high: float = 0.86):
    from core.params import default as P
    def _ctrl(x):
        V, _ = x
        cP = P.MP / V
        return u_high if cP >= threshold else 0.0
    return _ctrl


class _MPCController:
    """Wraps the CasADi solver into a simple callable."""
    def __init__(self, mode: str, N: int, params=P_default):
        self.solver, self.meta, self.LBG, self.UBG = build_mpc(
            mode=mode, horizon=N, params=params
        )

    def __call__(self, state: np.ndarray) -> float:
        import numpy as np
        x_init = np.tile(state, self.meta["N"] + 1)
        var_init = np.hstack([x_init, self.meta["u_init"]])
        sol = self.solver(x0=var_init, p=state, lbg=self.LBG, ubg=self.UBG)
        return float(sol["x"].full().ravel()[self.meta["Uslice"]][0])


def mpc_spec(N=20, params=P_default):
    return _MPCController("spec", N, params)


def mpc_time_opt(N=20, params=P_default):
    return _MPCController("time_opt", N, params)
