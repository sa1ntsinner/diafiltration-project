"""
control/__init__.py
───────────────────
Public interface of the *control* package.

Exports
-------
* build_robust_mpc  – factory that constructs the tightened “tube” MPC
* mpc_robust        – convenience helper that wraps `build_robust_mpc`
                      into a callable controller `u = f(x)`
"""

# --------------------------------------------------------------------------- #
# 1. Public symbols --------------------------------------------------------- #
# --------------------------------------------------------------------------- #
from .robust import build_robust_mpc          # main factory for robust MPC
from core.params import default as P_default  # nominal process constants

from typing import Callable, Any
import numpy as np


# --------------------------------------------------------------------------- #
# 2. Convenience wrapper ---------------------------------------------------- #
# --------------------------------------------------------------------------- #
def mpc_robust(
    N: int = 20,
    *,
    params: Any = P_default,
) -> Callable[[np.ndarray], float]:
    """
    Tube-based robust MPC controller (spec-tracking mode).

    This is just a *thin* wrapper around ``build_robust_mpc``.  It hides all
    the low-level solver plumbing and returns a clean Python function that
    maps the current state ``x`` → first control input ``u``.

    Parameters
    ----------
    N : int, optional
        Prediction horizon (number of control intervals).  Default: 20.
    params : core.params.ProcessParams, optional
        Process constants.  The default is the nominal parameter set
        ``core.params.default`` (imported above as *P_default*).

    Returns
    -------
    ctrl : Callable[[np.ndarray], float]
        Function that takes the *current* state ``x`` (shape (2,))
        and returns the next valve position ``u`` in [0, 1].
    """

    # --------------------------------------------------------------------- #
    # 2.1  Build the *nominal* tightened MPC once.                          #
    #      This returns:                                                    #
    #        • solver : casadi.nlpsol object                                #
    #        • meta   : dict with sizes and slices                          #
    #        • LBG    : lower bounds on g                                   #
    #        • UBG    : upper bounds on g                                   #
    # --------------------------------------------------------------------- #
    solver, meta, LBG, UBG = build_robust_mpc(horizon=N, params=params)

    # --------------------------------------------------------------------- #
    # 2.2  Wrap the solver call into a pure-Python controller.              #
    #      The user never sees CasADi directly.                             #
    # --------------------------------------------------------------------- #
    def _ctrl(state: np.ndarray) -> float:
        """
        Solve the MPC problem for *state* and return the first input u₀.

        The decision vector the solver expects is:

            [ V₀, ML₀, V₁, ML₁, …, V_N, ML_N,  u₀, u₁, …, u_{N-1} ]

        * Warm-start trick *:
          - Repeat the *current* state (V, ML) for all N+1 state nodes.
          - Use `meta["u_init"]` (e.g. a flat 0.5 profile) for the inputs.
          This speeds up IPOPT convergence considerably.
        """
        # Build the initial guess x0  (length = 2·(N+1) + N)
        x0 = np.hstack([
            np.tile(state, meta["N"] + 1),   # state trajectory guess
            meta["u_init"]                   # input trajectory guess
        ])

        # Solve the NLP:  minimise J  s.t.  g_L ≤ g ≤ g_U
        sol = solver(x0=x0, p=state, lbg=LBG, ubg=UBG)

        # Extract the optimal input sequence and return the first element
        u_opt = sol["x"].full().ravel()[meta["Uslice"]]  # slice = inputs only
        return float(u_opt[0])

    # Return the callable controller
    return _ctrl
