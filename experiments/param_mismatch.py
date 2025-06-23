"""
Parametric plant-model mismatch study (additional task 1) :contentReference[oaicite:6]{index=6}.
"""
import numpy as np
from src import ProcessParameters, DiafiltrationModel, MPCController, simulate
from src.robustness import plant_param_mismatch, tightened_mpc
from src.plotting import plot_trajectories, finalize_plots

p_nom  = ProcessParameters()
designer_model = DiafiltrationModel(p_nom)
mpc_nom = MPCController(designer_model, p_nom, N=p_nom.horizon, objective="time_opt")

for f in (0.75, 0.5, 0.25):
    plant = plant_param_mismatch(p_nom, f)
    res   = simulate(plant, mpc_nom.control, p_nom, t_final_h=6)
    plot_trajectories(res, f'kML={f:.2f}·nom', p_nom, style=':')

# simple robust MPC with tightened lactose limit
mpc_rob = tightened_mpc(designer_model, p_nom, N=p_nom.horizon)
res_rob = simulate(plant_param_mismatch(p_nom, 0.25), mpc_rob.control, p_nom, t_final_h=6)
plot_trajectories(res_rob, 'tightened MPC', p_nom, style='--')

finalize_plots(p_nom)
