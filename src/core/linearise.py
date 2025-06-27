"""Compute continuous-time Jacobian matrices (A,B) at (x*,u*)."""
import casadi as ca
import numpy  as np
from core.params   import ProcessParams
from core.dynamics import casadi_rhs

def linearise(x_star: np.ndarray,
              u_star: float,
              P: ProcessParams):
    x  = ca.SX.sym("x", 2)
    u  = ca.SX.sym("u")
    f  = casadi_rhs(P)
    A  = ca.jacobian(f(x, u), x)
    B  = ca.jacobian(f(x, u), u)
    A_f = ca.Function('A', [x, u], [A])
    B_f = ca.Function('B', [x, u], [B])
    return np.array(A_f(x_star, u_star)), np.array(B_f(x_star, u_star)).reshape(2,1)
