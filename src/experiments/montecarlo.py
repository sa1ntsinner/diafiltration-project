"""
experiments/montecarlo.py

Monte-Carlo sweep for parametric uncertainty
─────────────────────────────────────────────
Provides tools to assess controller robustness by simulating many variations
of the plant with uncertain parameters.

Includes:
    - sample_random_params() : generates perturbed ProcessParams
    - run()                  : simulates many runs and returns batch time,
                               peak lactose concentration, and success flag
"""

from __future__ import annotations
import numpy as np
import random
from dataclasses import replace, asdict
from typing import Callable

from core.params import ProcessParams, default as P_default
from sim         import simulate, Nominal  # Reuse the nominal scenario and simulator


# ────────────────────────────────────────────────
# ❶ Random sampling function for parameters
# ────────────────────────────────────────────────

def _u(low: float, high: float) -> float:
    """Helper for uniform random sampling in [low, high]."""
    return low + (high - low) * random.random()

def sample_random_params(P: ProcessParams = P_default) -> ProcessParams:
    """
    Returns a new ProcessParams object with *random perturbations* in key parameters:
    
    - kM_L : mass-transfer coefficient for lactose, scaled in [0.25 – 1.0] × nominal
             → simulates a slower plant (worse diffusion)
    - k    : membrane permeability, scaled in [0.8 – 1.2] × nominal
    - A    : membrane area, scaled in [0.8 – 1.2] × nominal

    Leakage (β) is not modelled here (implicitly disabled by β = 1).
    """
    return replace(
        P,
        kM_L = P.kM_L * _u(0.25, 1.0),
        k    = P.k    * _u(0.80, 1.20),
        A    = P.A    * _u(0.80, 1.20),
        # all other parameters are unchanged
    )


# ────────────────────────────────────────────────
# ❷ Monte-Carlo simulation driver
# ────────────────────────────────────────────────

def run(
    num_runs : int,
    ctrl_fun : Callable[[np.ndarray], float],
    *,
    sampler  : Callable[[ProcessParams], ProcessParams] = sample_random_params,
    P        : ProcessParams = P_default,
):
    """
    Runs `num_runs` simulations of the plant with randomly sampled parameters,
    using the given control function (MPC, policy, etc.).

    Parameters:
        num_runs : int
            How many random plants to simulate.
        ctrl_fun : callable
            Controller function that returns control input u(t) from state x.
        sampler : callable
            Function that returns a randomly perturbed ProcessParams.
        P : ProcessParams
            The base (unperturbed) process parameters.

    Returns:
        times_h   : list of batch durations [hours]
        peaks_cL  : list of max lactose concentrations [kg/m³]
        success   : list of bools (True if cP ≥ cP* and cL ≤ cL* within 6h)
    """
    times_h, peaks_cL, success = [], [], []
    tol = 1e-3  # Small tolerance to allow for floating point errors

    for _ in range(num_runs):
        # 1. Generate a new plant model
        P_rand = sampler(P)

        # 2. Simulate using the provided controller
        scen = Nominal(P_rand)
        t, V, ML, _ = simulate(ctrl_fun, scen)

        # 3. Convert to concentrations
        cP = P_rand.MP / V
        cL = ML / V

        # 4. Check if final concentrations meet specs
        fin = (cP[-1] >= P_rand.cP_star - tol) and (cL[-1] <= P_rand.cL_star + tol)

        # 5. Record results
        times_h.append(t[-1] / 3600)            # total batch time in hours
        peaks_cL.append(float(np.max(cL)))      # max lactose concentration during batch
        success.append(bool(fin))               # did it meet the spec?

    return times_h, peaks_cL, success
