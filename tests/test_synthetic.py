import numpy as np

from imufusion.core.tilt import propagate_gravity
from tests.conftest import angle_deg, make_trial


def test_same_seed_identical():
    a, _ = make_trial(duration_s=5, seed=3, gyro_noise=0.01, accel_noise=0.1)
    b, _ = make_trial(duration_s=5, seed=3, gyro_noise=0.01, accel_noise=0.1)
    assert np.array_equal(a.gyro_radps, b.gyro_radps) and np.array_equal(a.accel_mps2, b.accel_mps2)


def test_different_seeds_differ():
    a, _ = make_trial(duration_s=5, seed=3, accel_noise=0.1)
    b, _ = make_trial(duration_s=5, seed=4, accel_noise=0.1)
    assert not np.array_equal(a.accel_mps2, b.accel_mps2)


def test_noise_free_accel_is_gravity():
    series, truth = make_trial(duration_s=5)
    assert np.allclose(series.accel_mps2 / 9.81, truth.gravity_in_sensor, atol=1e-9)


def test_noise_free_gyro_integration_matches_truth():
    series, truth = make_trial(duration_s=60, rate_hz=100)
    g = truth.gravity_in_sensor[0]
    worst = 0.0
    for k in range(1, len(series.time_s)):
        # midpoint rate: using the end-of-interval sample alone adds a half-sample lag of about 0.15 degrees
        g = propagate_gravity(g, (series.gyro_radps[k] + series.gyro_radps[k - 1]) / 2, 0.01)
        worst = max(worst, angle_deg(g[None], truth.gravity_in_sensor[k][None])[0])
    assert worst < 0.1


def test_shake_adds_requested_norm():
    series, truth = make_trial(duration_s=40, shake=3.0)
    extra = np.linalg.norm(series.accel_mps2 - 9.81 * truth.gravity_in_sensor, axis=1)
    assert np.isclose(extra[2500], 3.0) and np.isclose(extra[100], 0.0)
