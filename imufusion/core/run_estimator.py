import numpy as np

from imufusion.config.settings import FilterConfig
from imufusion.core.complementary import run_complementary
from imufusion.core.gyro_only import run_gyro_only
from imufusion.core.kalman import run_kalman
from imufusion.core.tilt import accel_unit
from imufusion.data.types import ImuSeries

ESTIMATORS = ("accel_only", "gyro_only", "complementary", "kalman")


def run_estimator(name: str, series: ImuSeries, config: FilterConfig) -> np.ndarray:
    if name == "accel_only":
        result = accel_unit(series.accel_mps2)
    elif name == "gyro_only":
        result = run_gyro_only(series)
    elif name == "complementary":
        result = run_complementary(series, config)
    elif name == "kalman":
        result = run_kalman(series, config)[0]
    else:
        raise ValueError(f"unknown estimator: {name}")
    return result
