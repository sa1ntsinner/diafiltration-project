"""
dfp.simulate
============
Closed-loop driver with **exact batch-time detection**.

Two problems of a naive fixed-grid simulation are fixed here.

1. *Coarse plant integration.*  One RK4 step of Δt = 600 s is not enough once
   the volume becomes small; the plant is integrated with
   ``params.n_sub_plant`` sub-steps (5 s) instead.

2. *Quantised batch time.*  Declaring "batch finished" only on the control
   grid quantises the objective of the whole project to 10 min and lets the
   protein concentration overshoot its specification (values of
   ``cP ≈ 270 mol m⁻³`` were reported for ``u = 0.5``).  The terminal event

   .. math:: g(x) = \\max\\{c_{P,f}-c_P,\\; c_L-c_{L,f}\\} = 0

   is now located inside the last interval by bisection to ``1 ms``, and the
   trajectory is truncated exactly there.  Reported batch times are therefore
   continuous and directly comparable with the analytic optimum.

Additionally the driver records solver diagnostics, constraint violations and
the electricity bill, and it lets a controller ask for a *shorter* final
interval (used by the free-final-time MPC to land exactly on the target).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .config import NOMINAL, ProcessParams
from .plant import Plant, nominal_plant
from .tariff import lambda_tou

__all__ = ["ClosedLoopResult", "closed_loop"]


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ClosedLoopResult:
    """Everything a study needs to know about one closed-loop run."""

    label: str
    t: np.ndarray                    #: time grid [s] (len n+1)
    x: np.ndarray                    #: states, shape (3, n+1)
    u: np.ndarray                    #: applied inputs, shape (n,)
    cP: np.ndarray
    cL: np.ndarray
    p: np.ndarray                    #: permeate flow [m³/s]
    params: ProcessParams = NOMINAL
    finished: bool = False           #: specifications reached before ``t_max``
    batch_time: float = np.nan       #: [s] – ``t_max`` if never finished
    solve_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    solver_fail: int = 0
    info: Dict[str, Any] = field(default_factory=dict)

    # ── derived metrics ────────────────────────────────────────────────────
    @property
    def t_h(self) -> np.ndarray:
        return self.t / 3600.0

    @property
    def batch_time_h(self) -> float:
        return self.batch_time / 3600.0

    @property
    def V(self) -> np.ndarray:
        return self.x[0]

    @property
    def ML(self) -> np.ndarray:
        return self.x[1]

    @property
    def MP(self) -> np.ndarray:
        return self.x[2]

    @property
    def cL_peak(self) -> float:
        return float(np.max(self.cL))

    @property
    def cP_overshoot(self) -> float:
        """How far the product is over-concentrated past ``cP_f`` [mol m⁻³].

        The specification is the *equality* ``cP = cP_f``; a policy that keeps
        diluting after the protein target is reached over-concentrates the
        product, which is an off-spec batch even though ``cP >= cP_f`` holds.
        """
        return float(max(0.0, np.max(self.cP) - self.params.cP_f))

    @property
    def t_cP_spec(self) -> float:
        """First time the protein specification is reached [s] (``nan`` if never)."""
        idx = np.flatnonzero(self.cP >= self.params.cP_f * (1.0 - self.params.spec_tol))
        return float(self.t[idx[0]]) if idx.size else float("nan")

    @property
    def t_cL_spec(self) -> float:
        """First time the lactose specification is reached [s] (``nan`` if never)."""
        idx = np.flatnonzero(self.cL <= self.params.cL_f * (1.0 + self.params.spec_tol))
        return float(self.t[idx[0]]) if idx.size else float("nan")

    @property
    def spec_ok(self) -> bool:
        """Batch finished, product on spec and the crystallisation limit held."""
        return bool(self.finished
                    and self.cP_overshoot <= 1e-2 * self.params.cP_f
                    and self.cL_violation <= 1e-6 * self.params.cL_max)

    @property
    def protein_recovery(self) -> float:
        """Fraction of the initial protein still in the tank at the end."""
        return float(self.x[2, -1] / self.x[2, 0])

    @property
    def cL_violation(self) -> float:
        """Largest crystallisation-constraint violation [mol m⁻³]."""
        return float(max(0.0, np.max(self.cL) - self.params.cL_max))

    @property
    def diavolume(self) -> float:
        """Solvent consumption in units of the initial volume [–]."""
        dt = np.diff(self.t)
        return float(np.sum(self.u * self.p[:-1] * dt) / self.params.V0)

    def energy(self) -> float:
        """Pump energy [kWh] for ``P = P_idle + P_dyn·u``."""
        dt = np.diff(self.t)
        kw = self.params.pump_idle_kW + self.params.pump_dyn_kW * self.u
        return float(np.sum(kw * dt) / 3600.0)

    def energy_cost(self, lam: Callable[[float], float] = lambda_tou,
                    t_start: float = 0.0) -> float:
        """Electricity bill [€] using the day-ahead tariff."""
        dt = np.diff(self.t)
        kw = self.params.pump_idle_kW + self.params.pump_dyn_kW * self.u
        price = np.array([lam(t_start + 0.5 * (self.t[i] + self.t[i + 1]))
                          for i in range(len(dt))])
        return float(np.sum(kw * dt * price) / 3600.0)

    def summary(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "finished": self.finished,
            "batch_time_h": round(self.batch_time_h, 4),
            "cP_final": round(float(self.cP[-1]), 3),
            "cL_final": round(float(self.cL[-1]), 3),
            "cL_peak": round(self.cL_peak, 2),
            "cL_violation": round(self.cL_violation, 4),
            "cP_overshoot": round(self.cP_overshoot, 3),
            "spec_ok": self.spec_ok,
            "t_cP_spec_h": round(self.t_cP_spec / 3600.0, 4),
            "t_cL_spec_h": round(self.t_cL_spec / 3600.0, 4),
            "V_final_L": round(float(self.V[-1] * 1e3), 3),
            "diavolumes": round(self.diavolume, 3),
            "protein_recovery": round(self.protein_recovery, 5),
            "energy_kWh": round(self.energy(), 3),
            "cost_EUR": round(self.energy_cost(), 3),
            "mean_solve_ms": (round(float(np.mean(self.solve_times)) * 1e3, 1)
                              if self.solve_times.size else None),
            "solver_failures": self.solver_fail,
        }


# ─────────────────────────────────────────────────────────────────────────────
def _locate_event(plant: Plant, x0: np.ndarray, u: float, dt: float,
                  atol: float = 1e-3) -> Optional[float]:
    """Bisect for the first ``tau ∈ (0, dt]`` at which both specs hold.

    Returns ``None`` if the specification is not met anywhere in the interval.
    """
    if plant.spec_residual(plant.step(x0, u, dt)) > 0.0:
        return None
    lo, hi = 0.0, dt
    while hi - lo > atol:
        mid = 0.5 * (lo + hi)
        if plant.spec_residual(plant.step(x0, u, mid)) <= 0.0:
            hi = mid
        else:
            lo = mid
    return hi


def closed_loop(
    controller: Callable[..., float],
    plant: Optional[Plant] = None,
    *,
    label: str | None = None,
    params: Optional[ProcessParams] = None,
    t_max: Optional[float] = None,
    dt_ctrl: Optional[float] = None,
    x0: Optional[np.ndarray] = None,
    stop_on_spec: bool = True,
    event_atol: float = 1e-3,
    t0: float = 0.0,
) -> ClosedLoopResult:
    """Run one closed-loop batch.

    Parameters
    ----------
    controller
        Any callable ``u = controller(x)`` or ``u = controller(x, t=..., )``.
        Objects may expose ``reset()``, ``stats`` (list of solve times),
        ``n_fail`` and ``suggested_dt(x)``; all are optional.
    plant
        Simulated truth.  Defaults to the nominal plant.
    stop_on_spec
        Stop as soon as both terminal specifications hold (batch operation).
        Set to ``False`` to always run until ``t_max`` (used for the open-loop
        constant-``u`` study of task 2).
    t0
        Wall-clock time [s since midnight] at which the batch starts.  Only the
        *controller* sees it (economic MPC needs it to evaluate the tariff); the
        returned time grid always starts at zero.
    """
    plant = plant or nominal_plant(params or NOMINAL)
    P = params or plant.params
    t_max = P.t_max if t_max is None else t_max
    dt_ctrl = P.dt_ctrl if dt_ctrl is None else dt_ctrl
    label = label or getattr(controller, "label", controller.__class__.__name__)

    if hasattr(controller, "reset"):
        controller.reset()

    xk = np.asarray(P.x0 if x0 is None else x0, dtype=float)
    t_list: List[float] = [0.0]
    x_list: List[np.ndarray] = [xk.copy()]
    u_list: List[float] = []
    solve_times: List[float] = []

    finished = plant.specs_met(xk)
    t = 0.0
    while not finished and t < t_max - 1e-9:
        import time as _time

        tic = _time.perf_counter()          # NB: must not shadow the t0 argument
        try:
            u = float(controller(xk, t=t0 + t) if _accepts_time(controller)
                      else controller(xk))
        except TypeError:
            u = float(controller(xk))
        solve_times.append(_time.perf_counter() - tic)
        u = float(np.clip(u, P.u_min, P.u_max))

        dt = min(dt_ctrl, t_max - t)
        if hasattr(controller, "suggested_dt"):
            #  A controller may ask for a *shorter* interval (the analytic
            #  bang-bang law uses it to land exactly on cP_f).  Never go below
            #  dt_ctrl/100, so a misbehaving controller cannot stall the run.
            dt = float(min(dt, max(controller.suggested_dt(xk), dt_ctrl / 100.0)))

        tau = _locate_event(plant, xk, u, dt, event_atol) if stop_on_spec else None
        if tau is not None:
            dt = tau
            finished = True

        xk = plant.step(xk, u, dt)
        t += dt
        u_list.append(u)
        t_list.append(t)
        x_list.append(xk.copy())

    t_arr = np.asarray(t_list)
    x_arr = np.asarray(x_list).T
    y = plant.outputs(x_arr)
    batch_time = t_arr[-1] if finished else (t_max if stop_on_spec else t_arr[-1])

    return ClosedLoopResult(
        label=label, t=t_arr, x=x_arr, u=np.asarray(u_list),
        cP=y[0], cL=y[1], p=y[2], params=P, finished=finished,
        batch_time=float(batch_time),
        solve_times=np.asarray(solve_times),
        solver_fail=int(getattr(controller, "n_fail", 0)),
        info={"plant": plant.name, "t0": t0},
    )


def _accepts_time(fn) -> bool:
    """Whether ``fn`` accepts a keyword argument ``t``."""
    import inspect

    target = fn.__call__ if not inspect.isfunction(fn) and hasattr(fn, "__call__") else fn
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):  # pragma: no cover
        return False
    return "t" in sig.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
