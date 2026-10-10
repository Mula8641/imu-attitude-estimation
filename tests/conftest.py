import numpy as np
import pytest

from imufusion.data.synthetic import SyntheticConfig, make_synthetic_trial


def make_trial(duration_s=60.0, rate_hz=100.0, seed=1, gyro_noise=0.0, bias=(0.0, 0.0, 0.0), accel_noise=0.0, shake=0.0):
    return make_synthetic_trial(SyntheticConfig(duration_s, rate_hz, seed, gyro_noise, bias, accel_noise, shake))


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


@pytest.fixture
def clean_trial():
    return make_trial()
