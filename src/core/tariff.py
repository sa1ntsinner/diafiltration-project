"""
src/core/tariff.py
──────────────────
Defines a time-of-use (TOU) electricity pricing function for energy-aware MPC.

The function returns a tariff (price per kWh) depending on the time elapsed
since the start of the batch process.

Used in the economic and energy-aware MPC modes.
"""

import numpy as np

def lambda_tou(t: float) -> float:
    """
    Returns the electricity price [€/kWh] at a given batch time t [seconds].

    Time-of-Use pricing:
        - From  0 to 2 hours : 0.10 €/kWh  (off-peak)
        - From 2 to 4 hours : 0.35 €/kWh  (peak pricing)
        - From 4 to 6 hours : 0.10 €/kWh  (off-peak again)

    Parameters:
        t : float
            Time in seconds since the start of the process.

    Returns:
        float : Electricity price at time `t` [€/kWh].
    """
    h = t / 3600  # Convert time from seconds to hours

    if h < 2.0:
        return 0.10  # Off-peak
    elif h < 4.0:
        return 0.35  # Peak time
    else:
        return 0.10  # Back to off-peak
