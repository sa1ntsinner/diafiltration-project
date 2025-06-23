"""
parameters.py
-------------
Все численные константы процесса + удобный dataclass-контейнер.

Используется всеми модулями, поэтому НЕ меняйте имена полей без
соответствующих правок в остальном коде.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ProcessParameters:
    # ------------------------ физические параметры ------------------------
    k: float      = 4.79e-6        # [m s-2]    permeation coefficient (eq. 1) :contentReference[oaicite:0]{index=0}
    A: float      = 1.0            # [m²]       membrane area
    c_g: float    = 319.0          # [mol m-3]  gel concentration
    k_M_L: float  = 1.6e-5         # [m s-1]    mass-transfer coef. lactose (eq. 2)
    alpha: float  = 1.3            # [-]        partition factor lactose
    V0: float     = 0.1            # [m³]       100 L initial volume
    c_L0: float   = 150.0          # [mol m-3]  initial lactose concentration
    c_P0: float   = 10.0           # [mol m-3]  initial protein concentration
    m_P: float    = c_P0 * V0      # [mol]      total protein (proteins are fully retained)
    # -------------------------- ограничения / цели ------------------------
    c_L_max: float = 570.0         # [mol m-3]  максимальная допустимая [L] (кристаллизация)
    c_L_f: float   = 15.0          # [mol m-3]  целевая конечная [L]
    c_P_f: float   = 100.0         # [mol m-3]  целевая конечная [P]
    # ------------------------ симулятор / MPC -----------------------------
    dt: float      = 10.0 / 60.0   # [h]        шаг дискретизации 10 мин
    horizon: int   = 20            # N шагов предсказания (можно изменить в main.py)
