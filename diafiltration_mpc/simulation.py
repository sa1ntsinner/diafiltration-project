"""
simulation.py
-------------
Универсальная функция для моделирования «plant + controller».
Контроллер — любой объект с методом `.control(x)`.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List
from .model import DiafiltrationModel
from .parameters import ProcessParameters

def simulate(model       : DiafiltrationModel,
             controller  : Callable[[np.ndarray], float],
             p           : ProcessParameters,
             t_final_h   : float = 6.0) -> Dict[str, np.ndarray]:
    """
    Parameters
    ----------
    model      : 'истинная' модель (может включать tear / mismatch).
    controller : функция (или объект) с сигнатурой u = f(x_now)
    p          : параметры (для dt)
    t_final_h  : длительность моделирования, ч.
    Returns
    -------
    dict с временным рядом t, V, c_L, c_P, u
    """

    n_steps = int(np.ceil(t_final_h / p.dt))
    t  : List[float] = [0.0]
    V  : List[float] = [p.V0]
    cL : List[float] = [p.c_L0]
    u_hist : List[float] = []

    x = np.array([p.V0, p.c_L0])

    for k in range(n_steps):
        # управляющее воздействие
        u_k = controller(x)
        u_hist.append(u_k)

        # интеграция
        x = model.rk4_step(x, u_k, p.dt)
        t.append(t[-1] + p.dt)
        V.append(x[0])
        cL.append(x[1])

        # останов по достижению терминальных целей (ускоряет time_opt MPC)
        cP = p.m_P / x[0]
        if (x[1] <= p.c_L_f) and (cP >= p.c_P_f):
            break

    t_arr  = np.array(t)
    V_arr  = np.array(V)
    cL_arr = np.array(cL)
    cP_arr = p.m_P / V_arr
    u_arr  = np.array(u_hist + [np.nan])   # выровняем длину с t

    return dict(t=t_arr, V=V_arr, c_L=cL_arr, c_P=cP_arr, u=u_arr)
