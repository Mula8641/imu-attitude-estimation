from dataclasses import dataclass

import numpy as np

from imufusion.config.settings import FilterConfig, KalmanConfig
from imufusion.core.accel_gate import is_dynamic
from imufusion.core.tilt import accel_unit
from imufusion.data.types import ImuSeries


@dataclass(frozen=True)
class KalmanState:
    gravity: np.ndarray
    bias: np.ndarray
    covariance: np.ndarray


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def initial_state(first_accel_unit: np.ndarray, config: KalmanConfig) -> KalmanState:
    cov = np.diag([config.p0_gravity] * 3 + [config.p0_bias] * 3)
    return KalmanState(first_accel_unit.copy(), np.zeros(3), cov)


def kalman_predict(state: KalmanState, omega_radps: np.ndarray, dt_s: float, config: KalmanConfig) -> KalmanState:
    gravity = state.gravity - dt_s * np.cross(omega_radps - state.bias, state.gravity)
    jac = np.eye(6)
    jac[:3, :3] -= dt_s * _skew(omega_radps)
    jac[:3, 3:] = -dt_s * _skew(state.gravity)
    # q values are spectral densities (per second), so Q scales with dt
    process = np.diag([config.q_gravity] * 3 + [config.q_bias] * 3) * dt_s
    return KalmanState(gravity, state.bias, jac @ state.covariance @ jac.T + process)


def kalman_update(state: KalmanState, accel_unit_k: np.ndarray, config: KalmanConfig, r_scale: float) -> KalmanState:
    h = np.hstack([np.eye(3), np.zeros((3, 3))])
    r = config.r_measurement * r_scale * np.eye(3)
    gain = state.covariance @ h.T @ np.linalg.inv(h @ state.covariance @ h.T + r)
    x = np.concatenate([state.gravity, state.bias]) + gain @ (accel_unit_k - state.gravity)
    # Joseph form keeps P symmetric positive definite over long runs
    ikh = np.eye(6) - gain @ h
    covariance = ikh @ state.covariance @ ikh.T + gain @ r @ gain.T
    # the linear update does not preserve unit length
    return KalmanState(x[:3] / np.linalg.norm(x[:3]), x[3:], covariance)


def run_kalman(series: ImuSeries, config: FilterConfig) -> tuple[np.ndarray, np.ndarray]:
    n = len(series.time_s)
    units = accel_unit(series.accel_mps2)
    dts = np.diff(series.time_s)
    gravity = np.empty((n, 3))
    bias = np.empty((n, 3))
    state = initial_state(units[0], config.kalman)
    gravity[0], bias[0] = state.gravity, state.bias
    for k in range(1, n):
        dynamic = config.gate.use_gate and is_dynamic(series.accel_mps2[k], config.gate.gate_ms2)
        state = kalman_predict(state, series.gyro_radps[k], dts[k - 1], config.kalman)
        state = kalman_update(state, units[k], config.kalman, config.gate.kalman_r_inflation if dynamic else 1.0)
        gravity[k], bias[k] = state.gravity, state.bias
    return gravity, bias
