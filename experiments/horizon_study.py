"""
Influence of prediction horizon N ∈ {5, 20, 50} on tracking MPC
(mandatory task 3) :contentReference[oaicite:5]{index=5}.
"""
import numpy as np
from diafiltration_mpc import ProcessParameters, DiafiltrationModel, MPCController, simulate
from diafiltration_mpc.plotting import plot_trajectories, finalize_plots

p      = ProcessParameters()
plant  = DiafiltrationModel(p)

for N in (5, 20, 50):
    mpc = MPCController(plant, p, N=N, objective="tracking")
    res = simulate(plant, mpc.control, p, t_final_h=6)
    plot_trajectories(res, f'N={N}', p)

finalize_plots(p)
