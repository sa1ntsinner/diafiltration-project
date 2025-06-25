import numpy as np
from constants import MP

def threshold_policy(state):
    """
    Implements a simple threshold-based control strategy:
    If protein concentration cP ≥ 55 → u = 0.86
    Else → u = 0

    Parameters
    ----------
    state : array_like
        Current state [V, ML]

    Returns
    -------
    u : float
        Control action (either 0.86 or 0.0)
    """
    V, ML = state
    cP = MP / V
    return 0.86 if cP >= 55.0 else 0.0
