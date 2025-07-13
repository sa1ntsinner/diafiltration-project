"""
sim/simulate.py

Generic closed-loop / open-loop simulator.

Controller API
--------------
A controller is ANY callable:      u = ctrl(state)
    where:
        state : np.ndarray([V, ML]) – system state: volume and lactose mass
        u     : float ∈ [0,1]       – valve opening (dilution input)

We provide helper controller factories:
* constant-u       → fixed open-loop control
* threshold_policy → simple heuristic control
* mpc_spec         → quadratic-tracking MPC
* mpc_time_opt     → custom time-optimal MPC
"""
from __future__ import annotations
import numpy as np
from typing import Callable

from core.params     import default as P_default
from core.discretise import rk4_step
from control.builder import build_mpc
from sim.scenarios   import Scenario
from core.tariff     import lambda_tou

# ─────────────────────────────────────────────────────────────────────────────
# Simulation engine
# ─────────────────────────────────────────────────────────────────────────────
def simulate(
    controller: Callable[[np.ndarray], float],
    scenario  : Scenario,
    tf        : float | None = None,     # final time (s)
    dt        : float | None = None,     # integration step (s)
):
    """
    Generic closed-loop simulation with any Scenario and any controller.

    Parameters
    ----------
    controller : Callable[[state], float]
        Controller function returning u ∈ [0,1] based on current state.
    scenario : Scenario
        System dynamics (Nominal, ProteinLeakage, etc.)
    tf : float, optional
        Final simulation time [s]. Defaults to scenario.P.t_final.
    dt : float, optional
        Integration step size [s]. Defaults to scenario.P.dt_ctrl.

    Returns
    -------
    t : np.ndarray
        Time stamps.
    V : np.ndarray
        Volume trajectory.
    ML : np.ndarray
        Lactose mass trajectory.
    u_hist : np.ndarray
        Control inputs over time.
    """
    P   = scenario.P
    tf  = tf or P.t_final
    dt  = dt or P.dt_ctrl
    steps = int(tf / dt) + 1

    # initialize logs
    t = np.empty(steps)
    V = np.empty(steps)
    ML = np.empty(steps)
    u_hist = []

    # initial state
    x = np.array([P.V0, P.ML0])

    for k in range(steps):
        t[k] = k * dt
        V[k], ML[k] = x

        if scenario.specs_met(x):  # stop early if product is already good
            return t[:k+1], V[:k+1], ML[:k+1], np.array(u_hist)

        u = float(np.clip(controller(x), 0.0, 1.0))  # ensure u ∈ [0,1]
        u_hist.append(u)

        # RK4 time integration using current scenario dynamics
        x = rk4_step(x, u, dt, lambda s, uu: scenario.rhs(s, uu, t[k]))

    return t, V, ML, np.array(u_hist)


# ─────────────────────────────────────────────────────────────────────────────
# Controller factories
# ─────────────────────────────────────────────────────────────────────────────

def constant_u(u_val: float) -> Callable[[np.ndarray], float]:
    """Returns a controller that always applies a constant u."""
    return lambda _: u_val


def threshold_policy(threshold: float = 55.0, u_high: float = 0.86):
    """
    Simple heuristic:
    - if protein concentration ≥ threshold → apply u_high
    - otherwise → close valve (u = 0.0)
    """
    from core.params import default as P
    def _ctrl(x):
        V, _ = x
        cP = P.MP / V
        return u_high if cP >= threshold else 0.0
    return _ctrl


# ─────────────────────────────────────────────────────────────────────────────
# Generic MPC controller wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _MPCController:
    """
    Wraps a CasADi NLP solver into a Python controller function.

    After solving the MPC optimization problem, it extracts u₀ (first input).
    """
    def __init__(self, mode: str, N: int, params=P_default):
        self.solver, self.meta, self.LBG, self.UBG = build_mpc(
            mode=mode, horizon=N, params=params
        )

    def __call__(self, state: np.ndarray) -> float:
        import numpy as np
        x_init = np.tile(state, self.meta["N"] + 1)           # initial guess for states
        var_init = np.hstack([x_init, self.meta["u_init"]])   # full init guess vector
        sol = self.solver(x0=var_init, p=state, lbg=self.LBG, ubg=self.UBG)
        # return the first control action from optimized sequence
        return float(sol["x"].full().ravel()[self.meta["Uslice"]][0])


# ─────────────────────────────────────────────────────────────────────────────
# Specific MPC controller presets
# ─────────────────────────────────────────────────────────────────────────────

def mpc_spec(N=20, params=P_default):
    """Quadratic spec-tracking MPC."""
    return _MPCController("spec", N, params)

def mpc_time_opt(N: int = 20, *, params=P_default):
    """Time-optimal MPC (objective: maximize cP, minimize cL and time)."""
    return _MPCController("time_opt", N, params)

def mpc_econ(N=20, params=P_default):
    """Linear-cost economic MPC."""
    return _MPCController("econ", N, params)

def mpc_economic(N=20, *, params=P_default, lam_fun=lambda_tou):
    """
    Economic MPC with a time-of-use (TOU) electricity price penalty.
    Requires lambda_tou() to define cost per time.
    """
    return _MPCController(
        "spec", N, params,
        extra_weights=dict(lambda_fun=lam_fun)  # NOTE: this requires support in build_mpc
    )
