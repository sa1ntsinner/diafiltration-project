from __future__ import annotations
import casadi as ca, numpy as np
from core.params    import default as P_default
from core.linearise import linearise
from control.builder import build_mpc                # reuse nominal builder

def _dlqr(A,B,Q,R):
    """Discrete LQR gain via Riccati (simple helper)."""
    from scipy.linalg import solve_discrete_are, inv
    P = solve_discrete_are(A,B,Q,R)
    K = -inv(B.T@P@B+R) @ (B.T@P@A)
    return K

def offline_tube_gain(params=P_default, dt=None):
    """Returns stabilising feedback gain K (2×2)."""
    dt = dt or params.dt_ctrl
    A,B = linearise(np.array([params.V0, params.ML0]), 0.5, params)
    # forward-Euler disc. sufficient for gain design here
    Ad = np.eye(2) + A*dt
    Bd = B*dt
    Q  = np.diag([1e-4, 1e-4])
    R  = np.array([[1.0]])
    return _dlqr(Ad,Bd,Q,R)           # shape (1,2)

# --------------------------------------------------------------------------- #
def build_robust_mpc(
        horizon        : int = 20,
        base_mode      : str = "spec",    # "spec" or "time_opt"
        w_max          : np.ndarray = np.array([1e-6, 1e-5]),  # |disturbance|
        params                = P_default,
        K      : np.ndarray | None = None):
    """
    Tube-based tightening around nominal MPC.

    w_max : element-wise bound on additive discrepancy f̃(x,u)-f(x,u).
    """
    # 1. stabilising feedback gain
    K = K if K is not None else offline_tube_gain(params)

    # 2. compute invariant tube size (∞-norm) – simple worst-case bound
    #    Δx = Ω * w_max   with Ω = (I ⊕ (A+BK))^{-1}
    A,B = linearise(np.array([params.V0, params.ML0]), 0.5, params)
    Ad = np.eye(2) + A*params.dt_ctrl
    Bd = B*params.dt_ctrl
    Acl = Ad + Bd@K
    # solve Lyapunov: Ω = inv(I – |Acl|)
    Om = np.linalg.inv(np.eye(2) - np.abs(Acl))

    tighten = np.abs(Om) @ w_max          # 2-vector [ΔV, ΔML]

    # 3. call nominal builder and shrink constraints
    solver, meta, LBG, UBG = build_mpc(
        mode     = base_mode,
        horizon  = horizon,
        params   = params,
    )

    # indices of path-constraints in g: recall order from builder.py
    # For clarity we recompute mask positions once:
    cL_path_idx = []
    base = 0
    for k in range(horizon):
        base += 2 + 1   # dynamics equality (2) + input bound (1)
        cL_path_idx.append(base)   # cL ≤ cL_max
        base += 1
    cL_path_idx = np.array(cL_path_idx)

    # tighten UB for lactose and terminal spec
    UBG[cL_path_idx] -= (tighten[1] / params.V0)   # worst-case ↑cL
    UBG[-1]          -= (tighten[1] / params.V0)   # terminal cL_star
    # P.cP bounds go the other way round; skip – LP already safe.

    meta["K"]       = K
    meta["tighten"] = tighten
    return solver, meta, LBG, UBG
