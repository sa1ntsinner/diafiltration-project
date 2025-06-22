import numpy as np, casadi as ca
from .constants import *
from .model import casadi_rhs

def _rk4_disc(f, dt):
    x = ca.SX.sym("x",2); u = ca.SX.sym("u")
    k1 = f(x,u)
    k2 = f(x+0.5*dt*k1,u)
    k3 = f(x+0.5*dt*k2,u)
    k4 = f(x+dt*k3,u)
    return ca.Function("F",[x,u],[x+dt/6*(k1+2*k2+2*k3+k4)])

def build_mpc(N: int = 20):
    f   = casadi_rhs()
    F_d = _rk4_disc(f, dt_ctrl)

    X  = ca.SX.sym("X", 2, N+1)
    U  = ca.SX.sym("U", 1, N)
    X0 = ca.SX.sym("X0", 2)

    g, lbg, ubg = [], [], []
    g += [X[:,0]-X0]; lbg += [0,0]; ubg += [0,0]
    J = 0
    for k in range(N):
        g += [X[:,k+1] - F_d(X[:,k], U[:,k])]; lbg += [0,0]; ubg += [0,0]
        cL_k = X[1,k]/X[0,k]
        g += [cL_k]; lbg += [-ca.inf]; ubg += [cL_max]
        J += 1
    cP_N = MP / X[0,N]
    cL_N = X[1,N] / X[0,N]
    g += [cP_N, cL_N]; lbg += [cP_star, -ca.inf]; ubg += [ca.inf, cL_star]

    nlp = {"f":J,
           "x":ca.vertcat(ca.reshape(X,-1,1), ca.reshape(U,-1,1)),
           "p":X0,
           "g":ca.vertcat(*g)}
    opts = {"ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes"}   # ← ignores CasADi warning logs
    solver = ca.nlpsol("solver","ipopt",nlp,opts)

    nX = (N+1)*2
    meta = {"N":N, "nX":nX,
            "Xslice":slice(0,nX),
            "Uslice":slice(nX,nX+N)}
    return solver, meta, np.array(lbg), np.array(ubg)
