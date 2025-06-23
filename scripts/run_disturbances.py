"""
Runs param-mismatch study (0.75 / 0.5 / 0.25 · kM_L) and prints batch times.
"""

from diafiltration.disturbance import rk4_mismatch
from diafiltration.mpc         import build_mpc
from diafiltration.constants   import *

def batch_time_mismatch(factor, N=20, max_steps=400):
    solver, meta, LBG, UBG = build_mpc(N)
    state = np.array([V0, ML0])
    step  = 0
    while step < max_steps:
        cP_now = MP / state[0]
        cL_now = state[1] / state[0]
        if cP_now >= cP_star and cL_now <= cL_star:
            return step*dt_ctrl/3600   # h
        x0 = np.hstack([np.tile(state, meta['N']+1), 0.5*np.ones(meta['N'])])
        sol = solver(x0=x0, p=state, lbg=LBG, ubg=UBG)
        u_now = float(sol["x"].full().ravel()[meta['Uslice']][0])
        state = rk4_mismatch(state, u_now, factor)
        step += 1
    return None

for f in (0.75, 0.50, 0.25):
    t = batch_time_mismatch(f)
    if t is None:
        print(f"factor={f:4.2f}: specs NOT met within 6 h")
    else:
        print(f"factor={f:4.2f}: batch {t:5.2f} h")
