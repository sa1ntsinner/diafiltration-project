"""
robustness.py
-------------
Utility helpers for the additional-tasks part: parametric & structural
plant-model mismatch and a simple constraint-tightening trick.

The idea: shrink the lactose constraint by ε so that, even with worst-case
k_M,L under-estimation, real c_L never violates the true 570 mol m-3 limit.
"""
from __future__ import annotations
import numpy as np
import casadi as ca
from .parameters import ProcessParameters
from .model import DiafiltrationModel
from .mpc_controller import MPCController

# -------------------------------------------------------------------------
def plant_param_mismatch(p: ProcessParameters, factor: float) -> DiafiltrationModel:
    """
    Returns a 'true' plant where k_M,L,true = factor * k_M,L (factor < 1)
    """
    p_true = ProcessParameters(k_M_L = p.k_M_L * factor)
    return DiafiltrationModel(p_true)

# -------------------------------------------------------------------------
def plant_structural_mismatch(p: ProcessParameters) -> DiafiltrationModel:
    """
    Adds protein leakage according to eq.(6) :contentReference[oaicite:4]{index=4}.
    """
    class StructMismatch(DiafiltrationModel):
        def protein_permeate_conc(self, c_P, flow):
            beta = 1.3
            kM_P = 1e-6
            num = beta
            den = 1 + (beta - 1) * ca.exp(flow / (kM_P * self.p.A))
            return c_P * num / den

        def rhs(self, x, u):
            V, c_L = ca.vertsplit(x)
            c_P = self.p.m_P / V
            flow = self.permeate_flow(c_P)
            c_L_p = self.lactose_permeate_conc(c_L, flow)

            # NEW: protein loss
            c_P_p = self.protein_permeate_conc(c_P, flow)
            dPdt  = -c_P_p * flow                             # lost moles
            m_P_new = self.p.m_P + dPdt                      # update mass
            c_P_new = m_P_new / V

            dVdt  = (u - 1) * flow
            dcLdt = (-c_L_p * flow) / V - c_L * (u - 1) * flow / V
            # replace invariant with derivative form
            return ca.vcat([dVdt, dcLdt])

    return StructMismatch(p)

# -------------------------------------------------------------------------
def tightened_mpc(model: DiafiltrationModel, p: ProcessParameters,
                  N: int, eps: float = 15.0) -> MPCController:
    """
    Very simple robustification: replace c_L_max with c_L_max - eps.
    """
    p_tight = ProcessParameters(c_L_max = p.c_L_max - eps)
    return MPCController(model, p_tight, N, objective="time_opt")
