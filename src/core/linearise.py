"""
core/linearise.py
─────────────────────
Compute continuous-time Jacobian matrices A, B at a given operating point (x*, u*).

These matrices are used for:
- Linear model approximations
- Tube-based robust MPC (e.g. DLQR feedback)
"""

import casadi as ca
import numpy  as np
from core.params   import ProcessParams
from core.dynamics import casadi_rhs


def linearise(x_star: np.ndarray,
              u_star: float,
              P: ProcessParams):
    """
    Compute the Jacobians A = ∂f/∂x and B = ∂f/∂u at a given point (x*, u*).

    Parameters
    ----------
    x_star : ndarray
        Operating point state vector [V, ML]
    u_star : float
        Operating point input (valve opening)
    P : ProcessParams
        Process parameters (model constants)

    Returns
    -------
    A : ndarray
        Jacobian ∂f/∂x evaluated at (x*, u*) – shape (2×2)
    B : ndarray
        Jacobian ∂f/∂u evaluated at (x*, u*) – shape (2×1)
    """

    # Define symbolic variables
    x = ca.SX.sym("x", 2)   # state vector [V, ML]
    u = ca.SX.sym("u")      # input: dilution valve

    # Get symbolic RHS function f(x,u)
    f = casadi_rhs(P)

    # Compute symbolic Jacobians
    A = ca.jacobian(f(x, u), x)   # partial derivative w.r.t. x
    B = ca.jacobian(f(x, u), u)   # partial derivative w.r.t. u

    # Turn into callable functions
    A_f = ca.Function('A', [x, u], [A])
    B_f = ca.Function('B', [x, u], [B])

    # Evaluate numerically at operating point (x*, u*)
    A_val = np.array(A_f(x_star, u_star))
    B_val = np.array(B_f(x_star, u_star)).reshape(2, 1)

    return A_val, B_val
