"""
model.py
--------
Непрерывная модель диафильтрации + численная интеграция.

СОСТОЯНИЯ
---------
x = [V, c_L]^T
V   – объём раствора  [m³]
c_L – концентрация лактозы [mol m-3]

Концентрация белка выводится из инварианта массы:
    c_P(V) = m_P / V

УПРАВЛЕНИЕ
----------
u = d/p ∈ [0, 1] – отношение расхода добавляемого растворителя
к расходу пермеата.

Динамика получена из материального баланса, см. pdf :contentReference[oaicite:1]{index=1}.
"""
from __future__ import annotations
import numpy as np
import casadi as ca
from parameters import ProcessParameters

class DiafiltrationModel:
    def __init__(self, p: ProcessParameters, tear: bool = False):
        """
        Parameters
        ----------
        p    : ProcessParameters – все константы процесса
        tear : bool – если True, применяется сценарий 'filter-cake tear',
                      то есть p̂ = 2 p при 30 ≤ c_P ≤ 60 mol/m³ (eq. 5).
        """
        self.p = p
        self.tear = tear

    # ---------- «физические» выражения (CasADi-совместимые) --------------

    def permeate_flow(self, c_P: ca.SX | np.ndarray) -> ca.SX:
        """ eq.(1): p(c_P) """
        p = self.p
        flow = p.k * p.A * ca.log(p.c_g / c_P)
        return flow

    def lactose_permeate_conc(self, c_L: ca.SX | np.ndarray, flow: ca.SX) -> ca.SX:
        """ eq.(2): c_L,p(c_L, p) """
        p = self.p
        num = p.alpha
        den = 1.0 + (p.alpha - 1.0) * ca.exp(flow / (p.k_M_L * p.A))
        return c_L * num / den

    # ------------------------ непрерывная динамика ------------------------

    def rhs(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """
        Описание ODE  ẋ = f(x,u).

        x = [V, c_L],  u = ratio d/p.

        Возвращает casadi-SX столбец [dV/dt, dc_L/dt].
        """
        V, c_L = ca.vertsplit(x)
        p      = self.p

        # белок: инвариант массы
        c_P = p.m_P / V

        # расход пермеата (возможна «авария tear»)
        flow = self.permeate_flow(c_P)
        if self.tear:
            flow = ca.if_else(ca.logic_and(c_P >= 30, c_P <= 60),
                              2.0 * flow, flow)        # eq.(5)

        # концентрация лактозы в пермеате
        c_L_p = self.lactose_permeate_conc(c_L, flow)

        # ODE --------------------------------------------------------------
        dVdt   = (u - 1.0) * flow
        dcLdt  = (-c_L_p * flow) / V - c_L * (u - 1.0) * flow / V
        return ca.vcat([dVdt, dcLdt])

    # ------------------------- дискретизация ------------------------------

    def rk4_step(self, x_k: np.ndarray, u_k: float, dt: float) -> np.ndarray:
        """
        Одношаговый RK4-интегратор (NumPy).
        Используется в 'честной' симуляции (plant).
        """
        f = lambda x, u: ca.Function('f', [ca.MX.sym('x',2), ca.MX.sym('u')], 
                                     [self.rhs(ca.MX.sym('x',2), ca.MX.sym('u'))])(x, u).full().flatten()

        k1 = f(x_k,          u_k)
        k2 = f(x_k + 0.5*dt*k1, u_k)
        k3 = f(x_k + 0.5*dt*k2, u_k)
        k4 = f(x_k +     dt*k3, u_k)
        return x_k + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)

    # -------------------- построение CasADi-интегратора -------------------

    def casadi_integrator(self, dt: float) -> ca.Function:
        """
        Создаёт CasADi-интегратор F(x_k, u_k) → x_{k+1}
        (RK4 внутри, чтобы избежать зависимости от cvodes).
        """
        x  = ca.SX.sym('x', 2)
        u  = ca.SX.sym('u')
        x_next = self._rk4_sym(x, u, dt)
        return ca.Function('F', [x, u], [x_next])

    def _rk4_sym(self, x, u, dt):
        f = self.rhs
        k1 = f(x,          u)
        k2 = f(x + 0.5*dt*k1, u)
        k3 = f(x + 0.5*dt*k2, u)
        k4 = f(x +     dt*k3, u)
        return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
