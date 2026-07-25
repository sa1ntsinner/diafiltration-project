"""
dfp.experiments.montecarlo
==========================
Monte-Carlo robustness campaign.

Every draw perturbs the parameters that are genuinely uncertain in a membrane
plant:

======  =========================================  ==============
symbol  physical meaning                           sampled range
======  =========================================  ==============
kM_L    lactose mass-transfer coefficient          0.25 – 1.10 ×
k       permeability (fouling / batch-to-batch)    0.80 – 1.20 ×
alpha   partition function                         0.95 – 1.15 ×
cg      gel concentration                          0.95 – 1.05 ×
======  =========================================  ==============

``kM_L`` deliberately spans the range of the task sheet so that the campaign
covers the mismatch study as a special case.  The controller always keeps its
*nominal* model – only the plant changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..config import NOMINAL, ProcessParams
from ..plant import Plant
from ..simulate import closed_loop

__all__ = ["MonteCarloResult", "sample_params", "run_campaign", "DEFAULT_RANGES"]

DEFAULT_RANGES: Dict[str, tuple] = {
    "kM_L": (0.25, 1.10),
    "k": (0.80, 1.20),
    "alpha": (0.95, 1.15),
    "cg": (0.95, 1.05),
}


def sample_params(rng: np.random.Generator, base: ProcessParams = NOMINAL,
                  ranges: Dict[str, tuple] = DEFAULT_RANGES) -> ProcessParams:
    """Draw one random plant (multiplicative, uniform)."""
    return base.scaled(**{k: float(rng.uniform(*v)) for k, v in ranges.items()})


@dataclass
class MonteCarloResult:
    label: str
    batch_times_h: np.ndarray
    cL_peaks: np.ndarray
    cL_violations: np.ndarray
    finished: np.ndarray
    spec_ok: np.ndarray
    kappas: np.ndarray
    solve_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.spec_ok))

    @property
    def constraint_satisfaction(self) -> float:
        return float(np.mean(self.cL_violations <= 1e-6))

    def summary(self) -> Dict[str, float]:
        ok = self.spec_ok
        return {
            "label": self.label,
            "n": int(self.spec_ok.size),
            "finished_%": round(100.0 * float(np.mean(self.finished)), 1),
            "on_spec_%": round(100.0 * self.success_rate, 1),
            "cL_constraint_ok_%": round(100.0 * self.constraint_satisfaction, 1),
            "median_T_h": round(float(np.median(self.batch_times_h[ok])), 3) if ok.any() else None,
            "p90_T_h": round(float(np.percentile(self.batch_times_h[ok], 90)), 3) if ok.any() else None,
            "p90_cL_peak": round(float(np.percentile(self.cL_peaks, 90)), 1),
            "max_cL_violation": round(float(np.max(self.cL_violations)), 2),
            "mean_solve_ms": round(float(np.mean(self.solve_ms)), 1) if self.solve_ms.size else None,
        }


def run_campaign(
    controller_factory: Callable[[], object],
    *,
    label: str,
    n_draws: int = 60,
    seed: int = 20250725,
    params: ProcessParams = NOMINAL,
    ranges: Dict[str, tuple] = DEFAULT_RANGES,
    progress: bool = False,
    rebuild: bool = False,
) -> MonteCarloResult:
    """Simulate ``n_draws`` random plants.

    The controller keeps its *nominal* model in every draw, so by default it is
    built **once** and only ``reset()`` between draws: every IPOPT instance owns
    a compiled expression graph, and building one per draw exhausts the memory of
    a normal machine long before the campaign ends.  Pass ``rebuild=True`` to
    force a fresh controller per draw.
    """
    rng = np.random.default_rng(seed)
    T, peaks, viol, fin, ok, kap, ms = [], [], [], [], [], [], []
    shared = None if rebuild else controller_factory()
    for i in range(n_draws):
        p_true = sample_params(rng, params, ranges)
        plant = Plant(name=f"draw {i}", params=params, theta_true=p_true.theta)
        ctrl = controller_factory() if rebuild else shared
        res = closed_loop(ctrl, plant, params=params, label=label)
        T.append(res.batch_time_h)
        peaks.append(res.cL_peak)
        viol.append(res.cL_violation)
        fin.append(res.finished)
        ok.append(res.spec_ok)
        kap.append(p_true.kM_L / params.kM_L)
        ms.append(float(np.mean(res.solve_times)) * 1e3 if res.solve_times.size else 0.0)
        if progress and (i + 1) % 10 == 0:
            print(f"    {label}: {i + 1}/{n_draws}", flush=True)
    return MonteCarloResult(label, np.array(T), np.array(peaks), np.array(viol),
                            np.array(fin), np.array(ok), np.array(kap), np.array(ms))
