import numpy as np

from imufusion.config.settings import FilterConfig
from imufusion.core.accel_gate import is_dynamic
from imufusion.core.tilt import accel_unit, propagate_gravity
from imufusion.data.types import ImuSeries


def complementary_step(gravity_prev: np.ndarray, omega_radps: np.ndarray, accel_unit_k: np.ndarray, dt_s: float, alpha: float) -> np.ndarray:
    blended = alpha * propagate_gravity(gravity_prev, omega_radps, dt_s) + (1.0 - alpha) * accel_unit_k
    return blended / np.linalg.norm(blended)


def run_complementary(series: ImuSeries, config: FilterConfig) -> np.ndarray:
    n = len(series.time_s)
    units = accel_unit(series.accel_mps2)
    dts = np.diff(series.time_s)
    estimates = np.empty((n, 3))
    estimates[0] = units[0]
    tau = config.complementary.tau_s
    for k in range(1, n):
        dt = dts[k - 1]
        alpha = tau / (tau + dt)
        if config.gate.use_gate and is_dynamic(series.accel_mps2[k], config.gate.gate_ms2):
            alpha = 1.0
        estimates[k] = complementary_step(estimates[k - 1], series.gyro_radps[k], units[k], dt, alpha)
    return estimates
