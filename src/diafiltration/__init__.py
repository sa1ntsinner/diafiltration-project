"""Diafiltration batch-MPC library (APC-2025)."""
from importlib.metadata import version as _v
from .constants   import *
from .model       import flux_permeate, lactose_permeate_conc
from .mpc         import build_mpc
from .simulator   import closed_loop, open_loop

__all__ = (
    "flux_permeate", "lactose_permeate_conc",
    "build_mpc", "closed_loop", "open_loop"
)
__version__ = _v("diafiltration")