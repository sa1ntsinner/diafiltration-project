"""
dfp.tariff
==========
Day-ahead electricity price used by the economic-MPC extension.

The profile is a typical German spot-market weekday (EUR per kWh).  Prices are
interpolated with a *periodic cubic* spline so that the tariff, and hence the
economic objective, is :math:`C^2` – IPOPT converges markedly better than with
the piecewise-linear version, which has kinks on every hour boundary.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

__all__ = ["PRICE_PROFILE", "lambda_tou", "lambda_tou_casadi"]

#: Hourly day-ahead price, EUR/kWh, hours 0 … 23.
PRICE_PROFILE = np.array([
    0.09, 0.08, 0.08, 0.09, 0.10, 0.12,
    0.18, 0.25, 0.28, 0.30, 0.32, 0.35,
    0.37, 0.34, 0.30, 0.26, 0.23, 0.20,
    0.18, 0.16, 0.14, 0.12, 0.10, 0.09,
])

# knots 0 … 24 with the profile wrapped so the spline is periodic
_H = np.arange(25.0)
_P = np.append(PRICE_PROFILE, PRICE_PROFILE[0])
_SPLINE = ca.interpolant("tariff", "bspline", [list(_H)], list(_P))


def lambda_tou(t_sec: float) -> float:
    """Electricity price [EUR/kWh] ``t_sec`` seconds after midnight."""
    return float(_SPLINE((t_sec / 3600.0) % 24.0))


def lambda_tou_casadi(t_sec):
    """Symbolic version for use inside a CasADi NLP."""
    return _SPLINE(ca.fmod(t_sec / 3600.0, 24.0))
