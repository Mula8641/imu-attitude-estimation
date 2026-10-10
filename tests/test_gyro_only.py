import numpy as np

from imufusion.core.gyro_only import run_gyro_only
from tests.conftest import angle_deg, make_trial


def test_noise_free_tracks_truth():
    series, truth = make_trial()
    assert angle_deg(run_gyro_only(series), truth.gravity_in_sensor).max() < 0.5


def test_bias_makes_error_grow():
    series, truth = make_trial(bias=(0.01, 0.01, 0.01))
    err = angle_deg(run_gyro_only(series), truth.gravity_in_sensor)
    assert err[-1] > 4 * err[1000]
