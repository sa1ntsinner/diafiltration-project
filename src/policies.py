"""
policies.py
-----------
Набор «ручных» стратегий управления – нужен для сравнения с MPC.
"""
def baseline_policy(c_P: float) -> float:
    """
    Политика из задания (eq. 4). Возвращает u∈[0,1].
    """
    return 0.0 if c_P < 55.0 else 0.86
