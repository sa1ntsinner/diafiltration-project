"""Control laws: heuristics, non-linear MPC, robust multi-stage NMPC, OCP."""

from .heuristic import BangBang, ConstantU, ThresholdPolicy
from .nmpc import NMPC, OBJECTIVES, build_nmpc
from .ocp import AnalyticOptimum, analytic_optimum, solve_min_time_ocp, switching_price
from .multistage import MultiStageNMPC, build_multistage_nmpc

__all__ = [
    "BangBang", "ConstantU", "ThresholdPolicy",
    "NMPC", "build_nmpc", "OBJECTIVES",
    "AnalyticOptimum", "analytic_optimum", "solve_min_time_ocp", "switching_price",
    "MultiStageNMPC", "build_multistage_nmpc",
]
