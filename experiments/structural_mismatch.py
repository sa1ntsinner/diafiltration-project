"""
Structural mismatch: protein leakage scenario (additional task 2) :contentReference[oaicite:7]{index=7}.
"""
from diafiltration_mpc import ProcessParameters, simulate
from diafiltration_mpc.robustness import plant_structural_mismatch
from diafiltration_mpc.mpc_controller import MPCController
from diafiltration_mpc.model import DiafiltrationModel
from diafiltration_mpc.plotting import plot_trajectories, finalize_plots

p     = ProcessParameters()
plant = plant_structural_mismatch(p)

# MPC designed *without* knowing the leakage
mpc_simple = MPCController(DiafiltrationModel(p), p, N=p.horizon, objective="time_opt")
res_simp   = simulate(plant, mpc_simple.control, p, t_final_h=6)
plot_trajectories(res_simp, 'MPC ignorant', p)

# MPC with full knowledge
mpc_true = MPCController(plant, p, N=p.horizon, objective="time_opt")
res_true = simulate(plant, mpc_true.control, p, t_final_h=6)
plot_trajectories(res_true, 'MPC aware', p, style='--')

finalize_plots(p)
