"""
dfp.controllers.base
====================
Small utilities shared by every optimisation-based controller.

:class:`VarStack` removes the most common source of bugs in hand-written
CasADi NLPs: keeping the decision vector, its bounds and its initial guess in
three separate lists that must stay aligned.  Variables are registered by
name, and the stack returns *views* into the flat vector, so bounds and
warm-start values can always be addressed symbolically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import casadi as ca
import numpy as np

__all__ = ["VarStack", "ConstraintStack", "ControllerBase"]


class VarStack:
    """Named blocks of NLP decision variables with bounds and initial guess."""

    def __init__(self) -> None:
        self._sym: List[ca.SX] = []
        self._lb: List[np.ndarray] = []
        self._ub: List[np.ndarray] = []
        self._x0: List[np.ndarray] = []
        self._slices: Dict[str, slice] = {}
        self._shapes: Dict[str, Tuple[int, int]] = {}
        self._n = 0

    def add(self, name: str, shape: Tuple[int, int] | int, lb=-ca.inf, ub=ca.inf,
            x0=0.0) -> ca.SX:
        rows, cols = (shape, 1) if isinstance(shape, int) else shape
        sym = ca.SX.sym(name, rows, cols)
        n = rows * cols
        self._sym.append(ca.reshape(sym, n, 1))
        self._lb.append(np.broadcast_to(np.asarray(lb, float), (n,)).astype(float).copy())
        self._ub.append(np.broadcast_to(np.asarray(ub, float), (n,)).astype(float).copy())
        self._x0.append(np.broadcast_to(np.asarray(x0, float), (n,)).astype(float).copy())
        self._slices[name] = slice(self._n, self._n + n)
        self._shapes[name] = (rows, cols)
        self._n += n
        return sym

    # ── accessors ──────────────────────────────────────────────────────────
    @property
    def vector(self) -> ca.SX:
        return ca.vertcat(*self._sym) if self._sym else ca.SX.zeros(0)

    @property
    def lb(self) -> np.ndarray:
        return np.concatenate(self._lb) if self._lb else np.zeros(0)

    @property
    def ub(self) -> np.ndarray:
        return np.concatenate(self._ub) if self._ub else np.zeros(0)

    @property
    def x0(self) -> np.ndarray:
        return np.concatenate(self._x0) if self._x0 else np.zeros(0)

    @property
    def n(self) -> int:
        return self._n

    def slice(self, name: str) -> slice:
        return self._slices[name]

    def extract(self, flat: np.ndarray, name: str) -> np.ndarray:
        rows, cols = self._shapes[name]
        block = np.asarray(flat).ravel()[self._slices[name]]
        return block.reshape(rows, cols, order="F") if cols > 1 else block

    def set_x0(self, flat: np.ndarray, name: str, value) -> None:
        rows, cols = self._shapes[name]
        v = np.asarray(value, float)
        flat[self._slices[name]] = (v.reshape(-1, order="F") if cols > 1
                                    else np.broadcast_to(v, (rows,)).ravel())


class ConstraintStack:
    """Accumulates ``lbg ≤ g(x) ≤ ubg`` in one place."""

    def __init__(self) -> None:
        self.g: List[ca.SX] = []
        self.lbg: List[np.ndarray] = []
        self.ubg: List[np.ndarray] = []

    def add(self, expr: ca.SX, lb=0.0, ub=0.0) -> None:
        expr = ca.reshape(expr, expr.numel(), 1)
        n = expr.numel()
        self.g.append(expr)
        self.lbg.append(np.broadcast_to(np.asarray(lb, float), (n,)).astype(float))
        self.ubg.append(np.broadcast_to(np.asarray(ub, float), (n,)).astype(float))

    def eq(self, expr: ca.SX) -> None:
        self.add(expr, 0.0, 0.0)

    def leq(self, expr: ca.SX) -> None:
        """``expr <= 0``."""
        self.add(expr, -ca.inf, 0.0)

    def build(self):
        g = ca.vertcat(*self.g) if self.g else ca.SX.zeros(0)
        lbg = np.concatenate(self.lbg) if self.lbg else np.zeros(0)
        ubg = np.concatenate(self.ubg) if self.ubg else np.zeros(0)
        return g, lbg, ubg


@dataclass
class ControllerBase:
    """Common bookkeeping: warm start, timing statistics, failure counter."""

    label: str = "controller"
    n_fail: int = 0
    stats: List[str] = field(default_factory=list)
    _w0: np.ndarray | None = field(default=None, repr=False)

    def reset(self) -> None:
        self._w0 = None
        self.n_fail = 0
        self.stats = []
