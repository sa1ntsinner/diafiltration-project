"""
main.py
-------
Точка входа. Запускает:
  1. Открытую петлю (u = 0.5, 0.6, 0.7)  – демо динамики
  2. Сравнение baseline vs. MPC (tracking)
  3. Сценарий 'tear' + MPC (time_opt)

Меняйте параметры, горизонты и т.д. при желании.
"""
from parameters import ProcessParameters
from model import DiafiltrationModel
from policies import baseline_policy
from mpc_controller import MPCController
from simulation import simulate
from plotting import plot_trajectories, finalize_plots

# --------------------------- 1. OPEN LOOP --------------------------------
p = ProcessParameters()
plant = DiafiltrationModel(p)

for u_const in (0.5, 0.6, 0.7):
    res = simulate(plant, controller=lambda x,u=u_const: u,
                   p=p, t_final_h=6.0)
    plot_trajectories(res, label=f'u={u_const:.1f}', p=p)

# --------------------------- 2. CLOSED LOOP ------------------------------
mpc = MPCController(model=plant, params=p, N=p.horizon, objective="tracking")
res_mpc = simulate(plant, controller=mpc.control, p=p, t_final_h=6.0)
plot_trajectories(res_mpc, label='MPC-tracking', p=p, style='--')

res_bl = simulate(plant, controller=lambda x: baseline_policy(x[0] and p.m_P / x[0]),
                  p=p, t_final_h=6.0)
plot_trajectories(res_bl, label='baseline policy', p=p, style=':')

# --------------------------- 3. TEAR SCENARIO ----------------------------
tear_plant = DiafiltrationModel(p, tear=True)
mpc_time = MPCController(model=tear_plant, params=p, N=p.horizon, objective="time_opt")
res_tear = simulate(tear_plant, controller=mpc_time.control,
                    p=p, t_final_h=6.0)
plot_trajectories(res_tear, label='MPC-tear', p=p, style='-.')
# -------------------------------------------------------------------------
finalize_plots(p)
