import dataclasses

import numpy as np

from imufusion.config.settings import DEFAULT_CONFIG
from imufusion.core.complementary import complementary_step, run_complementary
from imufusion.core.gyro_only import run_gyro_only
from imufusion.core.tilt import accel_unit
from tests.conftest import angle_deg, make_trial

NO_GATE = dataclasses.replace(DEFAULT_CONFIG, gate=dataclasses.replace(DEFAULT_CONFIG.gate, use_gate=False))


def test_noise_free_tracks_truth():
    series, truth = make_trial()
    assert angle_deg(run_complementary(series, NO_GATE), truth.gravity_in_sensor).max() < 1.0


def test_beats_accel_only_with_accel_noise():
    series, truth = make_trial(accel_noise=0.8, seed=5)
    comp = angle_deg(run_complementary(series, NO_GATE), truth.gravity_in_sensor).mean()
    acc = angle_deg(accel_unit(series.accel_mps2), truth.gravity_in_sensor).mean()
    assert comp < acc


def test_bias_error_stays_bounded_while_gyro_grows():
    series, truth = make_trial(bias=(0.02, -0.01, 0.015), duration_s=120)
    comp = angle_deg(run_complementary(series, NO_GATE), truth.gravity_in_sensor)
    gyro = angle_deg(run_gyro_only(series), truth.gravity_in_sensor)
    assert comp[2000:].max() < 2 * comp[2000:4000].max() and gyro[-1] > comp[-1]


def test_alpha_one_equals_gyro_only():
    series, _ = make_trial(duration_s=10, bias=(0.01, 0, 0))
    g = accel_unit(series.accel_mps2[:1])[0]
    ref = run_gyro_only(series)
    for k in range(1, 50):
        g = complementary_step(g, series.gyro_radps[k], accel_unit(series.accel_mps2[k : k + 1])[0], 0.01, 1.0)
    assert np.allclose(g, ref[49])


def test_alpha_zero_equals_accel_only():
    a = np.array([0.0, 0.6, 0.8])
    assert np.allclose(complementary_step(np.array([0, 0, 1.0]), np.ones(3), a, 0.01, 0.0), a)


def test_gate_reduces_error_during_shake():
    series, truth = make_trial(shake=8.0, accel_noise=0.05)
    window = slice(2000, 3000)
    off = angle_deg(run_complementary(series, NO_GATE), truth.gravity_in_sensor)[window]
    on = angle_deg(run_complementary(series, DEFAULT_CONFIG), truth.gravity_in_sensor)[window]
    assert on.mean() < off.mean()
