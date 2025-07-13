"""
control/robust.py
─────────────────
Helpers that build a *tube-tightened* robust MPC.

Overview
--------
1. offline_tube_gain(...)   →  compute a stabilising DLQR gain  K ∈ ℝ¹×²  
2. build_robust_mpc(...)    →  call build_mpc() then shrink lactose
                              constraints by a worst-case tube size.
"""

from __future__ import annotations
import numpy as np

from core.params     import default as P_default          # nominal parameters
from core.linearise  import linearise                     # Jacobian of RHS
from control.builder import build_mpc                     # nominal MPC

# --------------------------------------------------------------------------- #
# 0.  DLQR helper with SciPy *or* Kleinman fallback                           #
# --------------------------------------------------------------------------- #
def _solve_are_kleinman(A, B, Q, R, max_iter: int = 30, eps: float = 1e-12):
    """
    Tiny pure-NumPy solver for the discrete algebraic Riccati equation
    using Kleinman iteration.  Only used if SciPy is not installed.
    """
    P = Q
    for _ in range(max_iter):
        K  = -np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
        Pn = (A + B @ K).T @ P @ (A + B @ K) + K.T @ R @ K + Q
        if np.linalg.norm(Pn - P, ord="fro") < eps:
            return Pn
        P = Pn
    return P


try:
    # Preferred path – use SciPy’s robust ARE solver if present
    from scipy.linalg import solve_discrete_are, inv          # noqa: F401

    def _dlqr(A, B, Q, R):
        """Discrete-time LQR gain via SciPy."""
        P = solve_discrete_are(A, B, Q, R)
        K = -inv(B.T @ P @ B + R) @ (B.T @ P @ A)
        return K

except ModuleNotFoundError:
    # Fallback – use the Kleinman routine above (pure NumPy, no SciPy)
    def _dlqr(A, B, Q, R):
        """Discrete LQR gain via Kleinman iteration (SciPy unavailable)."""
        P = _solve_are_kleinman(A, B, Q, R)
        K = -np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
        return K


# --------------------------------------------------------------------------- #
# 1.  One-time DLQR design for the tube feedback gain K                       #
# --------------------------------------------------------------------------- #
def offline_tube_gain(params=P_default, dt: float | None = None) -> np.ndarray:
    """
    Compute a stabilising state-feedback gain K (shape 1×2) at the
    nominal linearisation point.

    Parameters
    ----------
    params : ProcessParams
        Physical / process constants.
    dt      : float, optional
        Sampling time used for controller design.
        Defaults to `params.dt_ctrl`.

    Returns
    -------
    K : ndarray(1×2)
    """
    dt = dt or params.dt_ctrl

    # Linearise the *continuous* plant at (x0, u0 = 0.5)
    A, B = linearise(np.array([params.V0, params.ML0]), 0.5, params)

    # Simple Forward-Euler discretisation is sufficient for DLQR gain
    Ad = np.eye(2) + A * dt
    Bd = B * dt

    # Small Q,R give a gentle gain – we only need stability, not performance
    Q = np.diag([1e-4, 1e-4])
    R = np.array([[1.0]])

    return _dlqr(Ad, Bd, Q, R)          # shape (1, 2)


# --------------------------------------------------------------------------- #
# 2.  Build a robust MPC with tightened lactose constraints                   #
# --------------------------------------------------------------------------- #
def build_robust_mpc(
    *,
    horizon: int = 20,
    base_mode: str = "spec",                   # could also be "time_opt"
    w_max: np.ndarray = np.array([1e-6, 1e-5]),  # |additive plant error|
    params = P_default,
    K: np.ndarray | None = None,
):
    """
    Create a *tube MPC*:

        • Uses the nominal MPC from `build_mpc`.
        • Computes a tube size Δx∞ then shrinks all lactose upper-bounds
          so the real state (nominal + tube) never violates the original
          constraints.

    Parameters
    ----------
    horizon : int
        Prediction horizon N for the underlying MPC.
    base_mode : {"spec", "time_opt"}
        Which objective to tighten around.
    w_max : ndarray(2,)
        Worst-case additive model error ‖w‖∞ per state component.
    params : ProcessParams
    K : ndarray(1×2), optional
        Pre-computed feedback gain.  If None we design it on the fly.

    Returns
    -------
    solver : casadi.nlpsol
    meta   : dict (contains feedback gain and tube size)
    LBG    : ndarray – lower bounds
    UBG    : ndarray – *tightened* upper bounds
    """
    # 1 ───────────────── feedback gain K (stabilises the tube dynamics)
    K = K if K is not None else offline_tube_gain(params)

    # 2 ───────────────── worst-case tube size  Δx∞  via geometric series
    A, B = linearise(np.array([params.V0, params.ML0]), 0.5, params)
    Ad   = np.eye(2) + A * params.dt_ctrl
    Bd   = B * params.dt_ctrl
    Acl  = Ad + Bd @ K                               # closed-loop A-matrix

    # Ω = (I − |Acl|)⁻¹   ⇒  Δx∞ = Ω · w_max     (∞-norm bound)
    Om       = np.linalg.inv(np.eye(2) - np.abs(Acl))
    tighten  = np.abs(Om) @ w_max                  # [ΔV, ΔML]

    # 3 ───────────────── build the *nominal* MPC first
    solver, meta, LBG, UBG = build_mpc(
        mode    = base_mode,
        horizon = horizon,
        params  = params,
    )

    # 4 ───────────────── tighten every lactose constraint
    tighten_cL = tighten[1] / params.V0                # convert ΔML → ΔcL

    # Path constraints: element in UBG equals params.cL_max
    is_lactose = np.isclose(UBG, params.cL_max)

    # Shrink: new_UB  = max(LB + ε,  UB − Δ)
    UBG[is_lactose] = np.maximum(
        LBG[is_lactose] + 1e-9,
        UBG[is_lactose] - tighten_cL,
    )

    # Terminal lactose spec is the *last* element of g
    UBG[-1] = max(LBG[-1] + 1e-9, UBG[-1] - tighten_cL)

    # Package extra info for diagnostics
    meta["K"]       = K
    meta["tighten"] = tighten

    return solver, meta, LBG, UBG
