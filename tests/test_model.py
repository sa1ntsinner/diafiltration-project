"""Model and integrator tests – the numerical foundation of every study."""

from __future__ import annotations

import numpy as np
import pytest

from dfp.config import NOMINAL
from dfp.integrate import cvodes_integrator, rk4_integrator
from dfp.model import build_model
from dfp.plant import leakage_plant, nominal_plant, tear_plant

P = NOMINAL


def test_initial_outputs_match_task_sheet():
    m = build_model(params=P)
    cP, cL, p, rL = m.outputs(P.x0).ravel()
    assert cP == pytest.approx(P.cP0)
    assert cL == pytest.approx(P.cL0)
    assert p == pytest.approx(P.k * P.A * np.log(P.cg / P.cP0), rel=1e-12)
    assert 0.0 < rL <= 1.0


def test_partition_ratio_bounded_on_the_whole_reachable_set():
    """Eq. (2) may never enrich the permeate above the retentate."""
    m = build_model(params=P)
    for cP in np.linspace(P.cP0, 0.97 * P.cg, 200):
        x = np.array([P.MP0 / cP, P.cL0 * P.V0, P.MP0])
        _, _, p, rL = m.outputs(x).ravel()
        assert p > 0.0
        assert 0.0 < rL <= 1.0 + 1e-12


def test_volume_is_monotonically_non_increasing():
    """u <= 1 implies dV/dt <= 0, hence cP can never overshoot."""
    m = build_model(params=P)
    for u in (0.0, 0.3, 1.0):
        for cP in (10.0, 50.0, 100.0, 250.0):
            x = np.array([P.MP0 / cP, 5.0, P.MP0])
            assert m.rhs(x, u)[0] <= 1e-18


def test_protein_is_conserved_without_leakage():
    plant = nominal_plant(P)
    x = P.x0.copy()
    for _ in range(20):
        x = plant.step(x, 0.4, P.dt_ctrl)
    assert x[2] == pytest.approx(P.MP0, rel=1e-14)


def test_lactose_hold_up_decreases_monotonically():
    plant = nominal_plant(P)
    x, prev = P.x0.copy(), P.ML0
    for _ in range(30):
        x = plant.step(x, 0.7, P.dt_ctrl)
        assert x[1] < prev
        prev = x[1]


def test_rk4_is_fourth_order():
    """Halving the step must cut the error by 2**4.

    The error is measured *component-wise relative*: the absolute error of the
    lactose hold-up happens to change sign around 150 s per sub-step, so the
    plain max-norm is non-monotone there and would hide the true order.
    """
    m = build_model(params=P)
    ref = np.asarray(rk4_integrator(m.f, 4096)(P.x0, 0.3, P.theta, 3600.0)).ravel()
    err = {}
    for M in (1, 2, 4, 8):
        xM = np.asarray(rk4_integrator(m.f, M)(P.x0, 0.3, P.theta, 3600.0)).ravel()
        err[M] = float(np.max(np.abs(xM - ref) / np.maximum(np.abs(ref), 1e-12)))
    for a, b in ((1, 2), (2, 4), (4, 8)):
        assert np.log2(err[a] / err[b]) == pytest.approx(4.0, abs=0.35)


def test_rk4_agrees_with_cvodes():
    m = build_model(params=P)
    args = (P.x0, 0.3, P.theta, 3600.0)
    a = np.asarray(rk4_integrator(m.f, 256)(*args)).ravel()
    b = np.asarray(cvodes_integrator(m.f, 3600.0)(*args)).ravel()
    assert np.allclose(a, b, rtol=1e-8)


def test_tear_doubles_the_flux_only_inside_the_window():
    nom, torn = nominal_plant(P), tear_plant(P)
    for cP, expect in ((20.0, 1.0), (45.0, 2.0), (80.0, 1.0)):
        x = np.array([P.MP0 / cP, 5.0, P.MP0])
        assert (torn.model.rhs(x, 0.0, torn.theta_true)[0]
                == pytest.approx(expect * nom.model.rhs(x, 0.0, nom.theta_true)[0]))


def test_leakage_loses_protein_and_is_integrator_independent():
    """The leakage scenario must be a genuine ODE, not a mutated attribute."""
    plant = leakage_plant(P, beta=1.3, kM_P=3e-6)
    x_a = P.x0.copy()
    for _ in range(6):
        x_a = plant.step(x_a, 0.5, P.dt_ctrl, n_sub=60)
    x_b = P.x0.copy()
    for _ in range(6):
        x_b = plant.step(x_b, 0.5, P.dt_ctrl, n_sub=600)
    assert x_a[2] < P.MP0
    assert np.allclose(x_a, x_b, rtol=1e-6)
