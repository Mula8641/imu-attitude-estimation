import numpy as np

from imufusion.core.tilt import accel_unit, propagate_gravity
from imufusion.data.types import ImuSeries


def run_gyro_only(series: ImuSeries) -> np.ndarray:
    n = len(series.time_s)
    estimates = np.empty((n, 3))
    estimates[0] = accel_unit(series.accel_mps2[:1])[0]
    dts = np.diff(series.time_s)
    for k in range(1, n):
        estimates[k] = propagate_gravity(estimates[k - 1], series.gyro_radps[k], dts[k - 1])
    return estimates
