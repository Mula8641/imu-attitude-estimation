from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImuSeries:
    time_s: np.ndarray
    gyro_radps: np.ndarray
    accel_mps2: np.ndarray


@dataclass(frozen=True)
class GravityTruth:
    time_s: np.ndarray
    gravity_in_sensor: np.ndarray
