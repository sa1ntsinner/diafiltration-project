"""
sim/__init__.py

Simulation package init
────────────────────────
This makes key functions and classes from the submodules available at package level.

Exposes:
    - simulate         : main simulation engine
    - constant_u       : open-loop controller with constant valve opening
    - threshold_policy : basic feedback controller using a rule-based threshold
    - mpc_spec         : MPC that tracks specs (cP*, cL*) using quadratic costs
    - mpc_time_opt     : time-optimal MPC formulation
    - mpc_econ         : economic MPC with cost-oriented objective
    - Nominal          : baseline plant scenario
    - Tear             : plant with intentional downstream disturbances
    - KmMismatch       : plant with reduced mass-transfer (slower dynamics)
    - ProteinLeakage   : plant with protein loss (leaky membrane)
"""

# Import controllers and the simulation loop
from .simulate import (
    simulate,
    constant_u,
    threshold_policy,
    mpc_spec,
    mpc_time_opt,
    mpc_econ,
)

# Import predefined plant model variations (used for robustness tests)
from .scenarios import (
    Nominal,
    Tear,
    KmMismatch,
    ProteinLeakage,
)
