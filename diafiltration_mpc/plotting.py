"""
plotting.py
-----------
Генерация единых графиков для отчёта / презентации.
"""
import matplotlib.pyplot as plt
import numpy as np
from parameters import ProcessParameters

def plot_trajectories(res_dict: dict, label: str, p: ProcessParameters,
                      style: str = '-') -> None:
    t_h = res_dict['t']
    plt.figure(1); plt.plot(t_h, res_dict['c_L'], style, label=label)
    plt.figure(2); plt.plot(t_h, res_dict['c_P'], style, label=label)
    plt.figure(3); plt.step(t_h, res_dict['u'], style, where='post', label=label)
    plt.figure(4); plt.plot(t_h, res_dict['V']*1e3, style, label=label)  # [L]

def finalize_plots(p: ProcessParameters):
    titles = ['Lactose concentration [mol/m³]',
              'Protein concentration [mol/m³]',
              'Control ratio u = d/p [-]',
              'Volume [L]']
    for i in range(4):
        plt.figure(i+1)
        plt.title(titles[i])
        plt.xlabel('Time [h]')
        plt.legend()
        plt.grid()
    plt.show()
