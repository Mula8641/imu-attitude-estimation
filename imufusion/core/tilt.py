import numpy as np


def accel_unit(accel_mps2: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(accel_mps2, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("accelerometer sample with zero norm")
    return accel_mps2 / norms


def roll_pitch_from_accel(accel_mps2: np.ndarray) -> np.ndarray:
    ax, ay, az = accel_mps2[:, 0], accel_mps2[:, 1], accel_mps2[:, 2]
    return np.stack([np.arctan2(ay, az), np.arctan2(-ax, np.hypot(ay, az))], axis=1)


def gravity_from_roll_pitch(roll_pitch_rad: np.ndarray) -> np.ndarray:
    roll, pitch = roll_pitch_rad[:, 0], roll_pitch_rad[:, 1]
    return np.stack([-np.sin(pitch), np.sin(roll) * np.cos(pitch), np.cos(roll) * np.cos(pitch)], axis=1)


def propagate_gravity(gravity: np.ndarray, omega_radps: np.ndarray, dt_s: float) -> np.ndarray:
    predicted = gravity - dt_s * np.cross(omega_radps, gravity)
    # the first order step leaves the unit sphere slightly; project back
    return predicted / np.linalg.norm(predicted)
