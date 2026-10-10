import dataclasses

import numpy as np

from imufusion.config.settings import DEFAULT_CONFIG
from imufusion.core.gyro_only import run_gyro_only
from imufusion.core.kalman import KalmanState, initial_state, kalman_predict, kalman_update, run_kalman
from imufusion.core.tilt import accel_unit
from tests.conftest import angle_deg, make_trial

TRUE_BIAS = (0.02, -0.01, 0.015)
NO_GATE = dataclasses.replace(DEFAULT_CONFIG, gate=dataclasses.replace(DEFAULT_CONFIG.gate, use_gate=False))


def start_state() -> KalmanState:
    return initial_state(np.array([0.0, 0.0, 1.0]), DEFAULT_CONFIG.kalman)


def test_predict_leaves_bias_unchanged():
    state = dataclasses.replace(start_state(), bias=np.array([0.1, 0.2, 0.3]))
    out = kalman_predict(state, np.array([0.3, -0.2, 0.1]), 0.01, DEFAULT_CONFIG.kalman)
    assert np.array_equal(out.bias, state.bias)


def test_predict_keeps_gravity_near_unit():
    out = kalman_predict(start_state(), np.array([0.3, -0.2, 0.1]), 0.01, DEFAULT_CONFIG.kalman)
    assert abs(np.linalg.norm(out.gravity) - 1) < 1e-3


def test_covariance_stays_symmetric_positive_definite():
    state, rng = start_state(), np.random.default_rng(0)
    for _ in range(10000):
        state = kalman_predict(state, rng.normal(0, 0.3, 3), 0.01, DEFAULT_CONFIG.kalman)
        state = kalman_update(state, accel_unit(np.array([[0, 0.1, 9.81]]))[0], DEFAULT_CONFIG.kalman, 1.0)
    assert np.allclose(state.covariance, state.covariance.T, atol=1e-9)
    assert np.linalg.eigvalsh((state.covariance + state.covariance.T) / 2).min() > -1e-9


def test_update_returns_unit_gravity():
    out = kalman_update(start_state(), np.array([0.0, 0.6, 0.8]), DEFAULT_CONFIG.kalman, 1.0)
    assert np.isclose(np.linalg.norm(out.gravity), 1.0)


def test_bias_converges_on_synthetic_data():
    series, _ = make_trial(duration_s=120, gyro_noise=0.01, accel_noise=0.1, bias=TRUE_BIAS)
    _, bias = run_kalman(series, NO_GATE)
    assert np.abs(bias[6000:] - np.asarray(TRUE_BIAS)).max() < 0.003


def test_beats_gyro_only_and_not_worse_than_accel_only():
    series, truth = make_trial(duration_s=60, gyro_noise=0.01, accel_noise=0.5, bias=TRUE_BIAS, seed=2)
    kal = angle_deg(run_kalman(series, NO_GATE)[0], truth.gravity_in_sensor).mean()
    gyro = angle_deg(run_gyro_only(series), truth.gravity_in_sensor).mean()
    acc = angle_deg(accel_unit(series.accel_mps2), truth.gravity_in_sensor).mean()
    assert kal < gyro and kal <= acc


def test_r_inflation_reduces_error_during_shake():
    series, truth = make_trial(shake=8.0, accel_noise=0.05)
    window = slice(2000, 3000)
    flat = dataclasses.replace(DEFAULT_CONFIG, gate=dataclasses.replace(DEFAULT_CONFIG.gate, kalman_r_inflation=1.0))
    plain = angle_deg(run_kalman(series, flat)[0], truth.gravity_in_sensor)[window].mean()
    inflated = angle_deg(run_kalman(series, DEFAULT_CONFIG)[0], truth.gravity_in_sensor)[window].mean()
    assert inflated < plain
