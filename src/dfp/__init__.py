"""
dfp – time-optimal control of a batch diafiltration process
===========================================================
*Advanced Process Control (SS25), TU Dortmund — project P2.*

The package is deliberately layered so that every study is a few lines of
code:

=========================  ===================================================
:mod:`dfp.config`          physical constants, specifications, uncertainty sets
:mod:`dfp.model`           the symbolic ODE model (single source of truth)
:mod:`dfp.integrate`       RK4 / collocation / CVODES discretisations
:mod:`dfp.plant`           simulated truth incl. tear, mismatch, leakage
:mod:`dfp.simulate`        closed-loop driver with exact batch-time detection
:mod:`dfp.controllers`     heuristics, NMPC (5 objectives), multi-stage NMPC, OCP
:mod:`dfp.estimation`      EKF and moving-horizon estimation of ``kM_L``
:mod:`dfp.experiments`     the study scripts behind every figure of the report
:mod:`dfp.viz`             Matplotlib house style and reusable panels
=========================  ===================================================

Example
-------
>>> from dfp import NOMINAL, build_nmpc, closed_loop, nominal_plant
>>> mpc = build_nmpc("min_time", N=20)
>>> res = closed_loop(mpc, nominal_plant())
>>> round(res.batch_time_h, 3)                       # doctest: +SKIP
3.545
"""

from .config import KM_L_UNCERTAINTY, NOMINAL, ProcessParams, UncertaintySet
from .controllers import (AnalyticOptimum, BangBang, ConstantU, MultiStageNMPC,
                          NMPC, ThresholdPolicy, analytic_optimum,
                          build_multistage_nmpc, build_nmpc, solve_min_time_ocp,
                          switching_price)
from .model import DiafiltrationModel, build_model
from .plant import (Plant, leakage_plant, mismatch_plant, nominal_plant,
                   tear_plant, TEAR_WINDOW)
from .simulate import ClosedLoopResult, closed_loop
from .tariff import lambda_tou

__version__ = "2.0.0"

__all__ = [
    "NOMINAL", "ProcessParams", "UncertaintySet", "KM_L_UNCERTAINTY",
    "build_model", "DiafiltrationModel",
    "Plant", "nominal_plant", "tear_plant", "mismatch_plant", "leakage_plant",
    "TEAR_WINDOW",
    "closed_loop", "ClosedLoopResult",
    "ConstantU", "ThresholdPolicy", "BangBang",
    "NMPC", "build_nmpc", "MultiStageNMPC", "build_multistage_nmpc",
    "AnalyticOptimum", "analytic_optimum", "solve_min_time_ocp", "switching_price",
    "lambda_tou", "__version__",
]
