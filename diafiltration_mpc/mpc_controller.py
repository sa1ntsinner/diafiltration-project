"""
mpc_controller.py
-----------------
MPC-контроллер на базе CasADi Opti.

Доступно два типа целевой функции:
    * "tracking"  – ∑(c_L − c_L_f)² + (c_P − c_P_f)²   (eq. 3)
    * "time_opt"  – ∑1  +  λ⋅концу точности            (минимизация N_act)

Для смены цели передайте objective="time_opt".

NB: ограничение c_L ≤ 570 всегда «жёсткое», терминальные требования
в варианте time_opt ставятся как мягкие с малым штрафом (λ = 1e4).
"""
import casadi as ca
import numpy as np
from .parameters import ProcessParameters
from .model import DiafiltrationModel
from typing import Union, Sequence


class MPCController:
    def __init__(self,
                 model      : DiafiltrationModel,
                 params     : ProcessParameters,
                 N          : int,
                 objective  : str = "tracking"):
        self.m  = model
        self.p  = params
        self.N  = N
        self.obj_type = objective

        # интегратор для дискретной модели
        self.F = self.m.casadi_integrator(params.dt)
        # компиляция MPC-оптимизатора
        self._build_opti()

    # ---------------------------------------------------------------------

    def _build_opti(self):
        p = self.p
        N = self.N
        opti = ca.Opti()

        X = opti.variable(2, N+1)   # состояния
        U = opti.variable(1, N)     # управляющие
        x0 = opti.parameter(2)      # параметр: текущее состояние

        # начальное условие
        opti.subject_to(X[:,0] == x0)

        J = 0  # целевая функция
        λ = 1e4  # штраф за терминальную неточность (time_opt)

        for k in range(N):
            # динамические ограничения
            opti.subject_to(X[:,k+1] == self.F(X[:,k], U[:,k]))

            # ограничение 0 <= u <= 1
            opti.subject_to(X[0,k] >= 1e-4)
            # ограничение c_L ≤ c_L_max
            opti.subject_to(X[1,k] <= p.c_L_max)

            # расчёт c_P по инварианту
            c_P_k = p.m_P / X[0,k]

            # ---------------- целевая функция -----------------------------
            if self.obj_type == "tracking":
                J += (X[1,k]-p.c_L_f)**2 + (c_P_k-p.c_P_f)**2
            elif self.obj_type == "time_opt":
                J += 1.0  # «плата» за каждый шаг
            else:
                raise ValueError("Unknown objective type")

        # терминальные ограничения
        c_P_N = p.m_P / X[0,N]
        opti.subject_to(X[1,N] <= p.c_L_f)   # ≤, чтобы допустить «лучше цели»
        opti.subject_to(c_P_N  >= p.c_P_f)   # ≥ 100 (концентрация только растёт)

        if self.obj_type == "tracking":
            J += (X[1,N]-p.c_L_f)**2 + (c_P_N-p.c_P_f)**2
        else:  # time_opt: мягкое требование
            J += λ*((ca.fmax(0, X[1,N]-p.c_L_f))**2 +
                    (ca.fmax(0, p.c_P_f-c_P_N))**2)

        # настройка вирт. солвера
        opti.minimize(J)
        opts = {"ipopt.tol":1e-6, "ipopt.print_level":0, "print_time":False}
        opti.solver('ipopt', opts)

        # сохранить для быстрого вызова
        self._opti  = opti
        self._X_var = X
        self._U_var = U
        self._x0    = x0

    # ---------------------------------------------------------------------

    def control(self, x_now: Union[np.ndarray, Sequence[float]]) -> float:
        """
        Решаем NLP и возвращаем первый оптимальный шаг управления u₀.
        """
        # --- гарантируем NumPy-массив ---------------------------------------
        x_now = np.asarray(x_now, dtype=float)

        # обновляем параметр "текущее состояние"
        self._opti.set_value(self._x0, x_now)

        # --------------------- initial guess / warm-start -------------------
        if hasattr(self, "_last_sol"):
            # тёплый старт предыдущим решением
            self._opti.set_initial(self._X_var, self._last_sol.value(self._X_var))
            self._opti.set_initial(self._U_var, self._last_sol.value(self._U_var))
        else:
            # первый вызов: «замораживаем» состояние, u = 0.5
            self._opti.set_initial(
                self._X_var,
                np.tile(x_now.reshape(-1, 1), (1, self.N + 1))
            )
            self._opti.set_initial(self._U_var, 0.5)

        # ----------------------------- решаем NLP ---------------------------
        try:
            sol = self._opti.solve()
            self._last_sol = sol
            u0 = sol.value(self._U_var[0, 0])
        except RuntimeError:              # если оптимизатор не сошёлся
            u0 = (float(self._last_sol.value(self._U_var[0, 0]))
                if hasattr(self, "_last_sol") else 0.5)

        return float(np.clip(u0, 0.0, 1.0))

