import casadi as ca
import numpy as np
from constants import *

def ca_rhs(x, u):
    V, ML = x[0], x[1]
    Vsafe = ca.fmax(V, 1e-6)
    cP = MP / Vsafe
    ratio = ca.fmax(cg / cP, 1e-6)
    p_raw = k * A * ca.log(ratio)
    p = ca.fmax(p_raw, 0)
    d = u * p
    cL = ML / Vsafe
    exp_term = ca.exp(p / (kM_L * A))
    cL_p = alpha * cL / (1 + (alpha - 1) * exp_term)
    return ca.vertcat(d - p, -cL_p * p)

def rk4_disc(dt):
    x = ca.SX.sym('x', 2)
    u = ca.SX.sym('u')
    k1 = ca_rhs(x, u)
    k2 = ca_rhs(x + 0.5*dt*k1, u)
    k3 = ca_rhs(x + 0.5*dt*k2, u)
    k4 = ca_rhs(x + dt*k3, u)
    x_next = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return ca.Function('F', [x, u], [x_next])

def build_mpc(N=20):
    F_disc = rk4_disc(dt_ctrl)
    X = ca.SX.sym('X', 2, N + 1)
    U = ca.SX.sym('U', 1, N)
    X0 = ca.SX.sym('X0', 2)
    g, lbg, ubg, J = [], [], [], 0
    g += [X[:, 0] - X0]; lbg += [0, 0]; ubg += [0, 0]
    for k in range(N):
        g += [X[:, k+1] - F_disc(X[:, k], U[:, k])]; lbg += [0, 0]; ubg += [0, 0]
        cL_k = X[1, k] / X[0, k]
        g += [cL_k]; lbg += [-ca.inf]; ubg += [cL_max]
        J += 1
    cP_N = MP / X[0, N]
    cL_N = X[1, N] / X[0, N]
    g += [cP_N, cL_N]; lbg += [cP_star, -ca.inf]; ubg += [ca.inf, cL_star]
    nlp = {'f': J, 'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)), 'p': X0, 'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})
    nX = (N + 1) * 2
    meta = {'nX': nX, 'N': N, 'Xslice': slice(0, nX), 'Uslice': slice(nX, nX + N)}
    return solver, meta, np.array(lbg), np.array(ubg)