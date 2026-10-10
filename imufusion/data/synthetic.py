from dataclasses import dataclass

import numpy as np

from imufusion.core.tilt import gravity_from_roll_pitch
from imufusion.data.types import GravityTruth, ImuSeries

G = 9.81
SHAKE_WINDOW_S = (20.0, 30.0)
YAW_RATE_RADPS = 0.02
# (amplitude_deg, frequency_hz, phase_rad) terms for roll and pitch
ROLL_TERMS = ((20.0, 0.05, 0.0), (10.0, 0.13, 1.0))
PITCH_TERMS = ((18.0, 0.07, 0.5), (12.0, 0.11, 0.0))


@dataclass(frozen=True)
class SyntheticConfig:
    duration_s: float
    rate_hz: float
    seed: int
    gyro_noise_radps: float
    gyro_bias_radps: tuple[float, float, float]
    accel_noise_mps2: float
    shake_accel_mps2: float


def _sinusoids(t: np.ndarray, terms: tuple) -> tuple[np.ndarray, np.ndarray]:
    angle = sum(np.radians(a) * np.sin(2 * np.pi * f * t + p) for a, f, p in terms)
    rate = sum(np.radians(a) * 2 * np.pi * f * np.cos(2 * np.pi * f * t + p) for a, f, p in terms)
    return angle, rate


def _body_rates(roll: np.ndarray, pitch: np.ndarray, roll_dot: np.ndarray, pitch_dot: np.ndarray) -> np.ndarray:
    # Z-Y-X Euler rate equations with constant yaw rate
    p = roll_dot - YAW_RATE_RADPS * np.sin(pitch)
    q = pitch_dot * np.cos(roll) + YAW_RATE_RADPS * np.sin(roll) * np.cos(pitch)
    r = -pitch_dot * np.sin(roll) + YAW_RATE_RADPS * np.cos(roll) * np.cos(pitch)
    return np.stack([p, q, r], axis=1)


def make_synthetic_trial(config: SyntheticConfig) -> tuple[ImuSeries, GravityTruth]:
    rng = np.random.default_rng(config.seed)
    t = np.arange(int(config.duration_s * config.rate_hz)) / config.rate_hz
    roll, roll_dot = _sinusoids(t, ROLL_TERMS)
    pitch, pitch_dot = _sinusoids(t, PITCH_TERMS)
    gravity = gravity_from_roll_pitch(np.stack([roll, pitch], axis=1))
    gyro = _body_rates(roll, pitch, roll_dot, pitch_dot) + np.asarray(config.gyro_bias_radps)
    gyro = gyro + rng.normal(0, config.gyro_noise_radps, gyro.shape)
    accel = G * gravity + rng.normal(0, config.accel_noise_mps2, gravity.shape)
    in_window = (t >= SHAKE_WINDOW_S[0]) & (t < SHAKE_WINDOW_S[1])
    # burst is perpendicular to gravity so it corrupts the tilt direction; its own magnitude is shake_accel_mps2
    sideways = np.cross(gravity, [1.0, 0.0, 0.0])
    sideways /= np.linalg.norm(sideways, axis=1, keepdims=True)
    accel[in_window] += config.shake_accel_mps2 * sideways[in_window]
    return ImuSeries(t, gyro, accel), GravityTruth(t, gravity)
