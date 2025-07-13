"""
core/tariff.py
────────────────────────────────────────────────────────────────────────────
Day-ahead spot-market electricity tariff for energy-aware MPC.

* One typical **weekday** price profile (24 values, €/kWh).
* Linear interpolation gives a smooth €/kWh at any second t.
* The function signature stays the same → controllers need *no* change.
"""

from __future__ import annotations
import numpy as np

# 24-hour profile, EUR **per kWh**  (≈ 80–370 EUR/MWh)
#             0h  1h  2h  3h  4h  5h  6h  7h  8h  9h 10h 11h
_PRICE = np.array([0.09, 0.08, 0.08, 0.09, 0.10, 0.12,
                   0.18, 0.25, 0.28, 0.30, 0.32, 0.35,
#            12h 13h 14h 15h 16h 17h 18h 19h 20h 21h 22h 23h
                   0.37, 0.34, 0.30, 0.26, 0.23, 0.20,
                   0.18, 0.16, 0.14, 0.12, 0.10, 0.09])

def lambda_tou(t_sec: float) -> float:             # ← same name, same units
    """
    Continuous €/kWh tariff from a typical day-ahead spot curve.

    Parameters
    ----------
    t_sec : float
        Seconds since batch start (0 s = 0 h on the profile).

    Notes
    -----
    * For batches that run > 24 h we *wrap around* the profile.
    * Hour-to-hour prices are **linearly interpolated** to avoid jumps
      that could confuse gradient-based solvers in economic MPC.
    """
    # convert to fractional hour in [0, 24)
    h = (t_sec / 3600.0) % 24.0
    i = int(np.floor(h))            # index of the “left” hour
    j = (i + 1) % 24                # index of the “right” hour (wrap)
    θ = h - i                       # linear blend factor
    return float((1.0 - θ) * _PRICE[i] + θ * _PRICE[j])