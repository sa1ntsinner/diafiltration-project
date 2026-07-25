"""Simulator tests – event detection, metrics and estimator behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from dfp.config import NOMINAL
from dfp.controllers import BangBang, ConstantU, analytic_optimum, build_nmpc
from dfp.estimation import EKF, MHE, AdaptiveNMPC, measurement
from dfp.plant import Plant, mismatch_plant, nominal_plant
from dfp.simulate import closed_loop

P = NOMINAL


def test_batch_time_is_not_quantised_by_the_control_grid():
    """The terminal event is located by bisection, not snapped to the grid."""
    r = closed_loop(BangBang(P), nominal_plant(P))
    assert abs(r.batch_time_h * 3600.0 % P.dt_ctrl) > 1.0


def test_terminal_event_is_tight():
    r = closed_loop(BangBang(P), nominal_plant(P))
    assert r.cL[-1] == pytest.approx(P.cL_f, rel=2e-3)
    assert r.cP[-1] >= P.cP_f * (1 - 1e-3)


def test_coarse_plant_integration_biases_the_batch_time():
    """Motivates params.n_sub_plant: one RK4 step of 600 s is not enough."""
    a = analytic_optimum(P)
    coarse = closed_loop(BangBang(P), Plant(name="coarse", params=P, n_sub=1))
    fine = closed_loop(BangBang(P), nominal_plant(P))
    assert abs(fine.batch_time_h - a.T_h) < abs(coarse.batch_time_h - a.T_h)


def test_metrics_are_self_consistent():
    r = closed_loop(ConstantU(0.6, P), nominal_plant(P))
    assert r.x.shape == (3, r.t.size)
    assert r.u.size == r.t.size - 1
    assert r.cL_peak == pytest.approx(float(np.max(r.cL)))
    assert r.energy() > 0.0
    assert 0.0 < r.energy_cost() < r.energy()          # price is well below 1 EUR/kWh
    assert set(r.summary()) >= {"batch_time_h", "spec_ok", "cL_violation"}


def test_open_loop_over_concentrates_at_small_u():
    r = closed_loop(ConstantU(0.5, P), nominal_plant(P))
    assert r.cP_overshoot > 50.0
    assert not r.spec_ok


@pytest.mark.parametrize("factor", [0.5, 0.25])
def test_mhe_identifies_the_mass_transfer_coefficient(factor):
    P8 = P.with_(t_max=8 * 3600)
    plant = mismatch_plant(factor, P8)
    mhe = MHE(P8)
    x, rng = P8.x0.copy(), np.random.default_rng(3)
    inputs = [0.0] * 12 + [1.0] * 12
    u_prev = 0.0
    for u in inputs:
        mhe.update(u_prev, measurement(x, rng))
        x = plant.step(x, u, P8.dt_ctrl)
        u_prev = u
    assert mhe.kappa == pytest.approx(factor, rel=0.15)


def test_adaptive_nmpc_recovers_constraint_satisfaction():
    P8 = P.with_(t_max=8 * 3600)
    ad = AdaptiveNMPC(build_nmpc("min_time", 20, params=P8), MHE(P8), params=P8,
                      rng=np.random.default_rng(7))
    blind = closed_loop(build_nmpc("min_time", 20, params=P8),
                        mismatch_plant(0.25, P8), params=P8)
    smart = closed_loop(ad, mismatch_plant(0.25, P8), params=P8)
    assert smart.cL_violation < 0.15 * blind.cL_violation


def test_ekf_stays_inside_its_bounds():
    ekf = EKF(P)
    x, rng = P.x0.copy(), np.random.default_rng(0)
    plant = mismatch_plant(0.5, P)
    u_prev = 0.0
    for u in [0.0] * 10 + [1.0] * 10:
        ekf.update(u_prev, measurement(x, rng))
        assert 0.05 <= ekf.kappa <= 4.0
        x = plant.step(x, u, P.dt_ctrl)
        u_prev = u
