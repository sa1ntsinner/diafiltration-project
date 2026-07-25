"""Controller tests – optimality, feasibility and constraint satisfaction."""

from __future__ import annotations

import numpy as np
import pytest

from dfp.config import KM_L_UNCERTAINTY, NOMINAL
from dfp.controllers import (BangBang, ConstantU, ThresholdPolicy, analytic_optimum,
                             build_multistage_nmpc, build_nmpc, solve_min_time_ocp,
                             switching_price)
from dfp.plant import mismatch_plant, nominal_plant, tear_plant
from dfp.simulate import closed_loop

P = NOMINAL


# ── the analytic optimum ────────────────────────────────────────────────────
def test_washing_price_is_minimal_at_the_protein_target():
    """Justifies the bang-bang structure: no interior (singular) minimum."""
    cP, price = switching_price(P)
    assert int(np.argmin(price)) == price.size - 1
    assert np.all(np.diff(price) < 0.0)


def test_analytic_optimum_matches_the_direct_ocp():
    a = analytic_optimum(P)
    ocp = solve_min_time_ocp(P, N=150)
    assert a.T_h == pytest.approx(ocp["T_h"], rel=2e-4)
    assert a.bang_bang


def test_optimum_respects_the_crystallisation_limit():
    a = analytic_optimum(P)
    assert a.cL_peak < P.cL_max


# ── closed loop on the nominal plant ────────────────────────────────────────
def test_bang_bang_closed_loop_reaches_the_analytic_optimum():
    a = analytic_optimum(P)
    r = closed_loop(BangBang(P), nominal_plant(P))
    assert r.spec_ok
    assert r.batch_time_h == pytest.approx(a.T_h, rel=1e-3)


@pytest.mark.parametrize("N", [5, 20])
def test_min_time_mpc_is_near_optimal_for_any_horizon(N):
    a = analytic_optimum(P)
    r = closed_loop(build_nmpc("min_time", N), nominal_plant(P))
    assert r.spec_ok
    assert r.batch_time_h < a.T_h * 1.02


def test_min_time_mpc_beats_the_heuristic_policy():
    mpc = closed_loop(build_nmpc("min_time", 20), nominal_plant(P))
    pol = closed_loop(ThresholdPolicy(params=P), nominal_plant(P))
    assert mpc.batch_time_h < pol.batch_time_h
    assert mpc.spec_ok and not pol.spec_ok      # the policy over-concentrates


def test_quadratic_tracking_is_clearly_suboptimal():
    """The point of task 3: Eq. (3) is a poor surrogate for time optimality."""
    a = analytic_optimum(P)
    r = closed_loop(build_nmpc("tracking", 20), nominal_plant(P))
    assert r.batch_time_h > a.T_h * 1.25


def test_short_horizon_tracking_never_finishes_but_stays_feasible():
    mpc = build_nmpc("tracking", 5)
    r = closed_loop(mpc, nominal_plant(P))
    assert not r.finished
    assert mpc.n_fail == 0            # the NLP must never be infeasible
    assert mpc.slack_active > 0       # ... it is the slack that absorbs it


def test_no_controller_violates_the_input_bounds():
    for ctrl in (build_nmpc("min_time", 20), build_nmpc("l1_time", 20),
                 ThresholdPolicy(params=P), ConstantU(0.6, P)):
        r = closed_loop(ctrl, nominal_plant(P))
        assert r.u.min() >= P.u_min - 1e-9
        assert r.u.max() <= P.u_max + 1e-9


# ── disturbances and mismatch ──────────────────────────────────────────────
def test_mpc_exploits_the_filter_cake_tear_without_going_off_spec():
    torn = closed_loop(build_nmpc("min_time", 20), tear_plant(P))
    nom = closed_loop(build_nmpc("min_time", 20), nominal_plant(P))
    assert torn.batch_time_h < nom.batch_time_h      # extra flux helps
    assert torn.spec_ok
    pol = closed_loop(ThresholdPolicy(params=P), tear_plant(P))
    assert pol.cP_overshoot > 10.0                   # the fixed policy does not


def test_nominal_model_mpc_violates_cL_max_under_strong_mismatch():
    P8 = P.with_(t_max=8 * 3600)
    r = closed_loop(build_nmpc("min_time", 20, params=P8), mismatch_plant(0.25, P8),
                    params=P8)
    assert r.cL_violation > 50.0


def test_multistage_nmpc_restores_constraint_satisfaction():
    P8 = P.with_(t_max=8 * 3600)
    ms = build_multistage_nmpc(20, params=P8, uncertainty=KM_L_UNCERTAINTY)
    for factor in (1.0, 0.5, 0.25):
        ms.reset()
        r = closed_loop(ms, mismatch_plant(factor, P8), params=P8)
        assert r.finished
        assert r.cL_violation < 0.01 * P8.cL_max
        assert ms.n_fail == 0


def test_multistage_costs_nothing_on_the_nominal_plant():
    a = analytic_optimum(P)
    P8 = P.with_(t_max=8 * 3600)
    ms = build_multistage_nmpc(20, params=P8)
    r = closed_loop(ms, nominal_plant(P8), params=P8)
    assert r.batch_time_h < a.T_h * 1.02
