from pathlib import Path

import h5py
import numpy as np

from imufusion.data.quaternion import gravity_in_sensor_from_quaternion
from imufusion.data.types import GravityTruth, ImuSeries
from imufusion.utils.log import get_logger

log = get_logger(__name__)
REQUIRED_KEYS = ("imu_acc", "imu_gyr", "opt_quat", "movement")


def list_broad_trials(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"BROAD directory not found: {directory}")
    return sorted(directory.glob("*.hdf5"))


def _block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    usable = len(values) // factor * factor
    return values[:usable].reshape(-1, factor, *values.shape[1:]).mean(axis=1)


def load_broad_trial(path: Path, quat_order: str, quat_rotates: str, imu_index: int = 0, decimate: int = 1) -> tuple[ImuSeries, GravityTruth, np.ndarray]:
    # BROAD files hold exactly one IMU and are already time aligned to the optical data, so imu_index must be 0
    if imu_index != 0:
        raise ValueError(f"BROAD trials contain a single IMU, got imu_index={imu_index}")
    if not Path(path).is_file():
        raise FileNotFoundError(f"trial file not found: {path}")
    with h5py.File(path, "r") as f:
        missing = [k for k in REQUIRED_KEYS if k not in f]
        if missing:
            raise ValueError(f"{path}: missing keys {missing}")
        rate_hz = float(f.attrs["sampling_rate"])
        gyro, accel = f["imu_gyr"][()], f["imu_acc"][()]
        quat, movement = f["opt_quat"][()], f["movement"][()].astype(float)
    gravity = gravity_in_sensor_from_quaternion(quat, quat_order, quat_rotates)
    # block averaging keeps the decimated rate free of aliased noise
    gyro, accel, gravity, movement = (_block_mean(a, decimate) for a in (gyro, accel, gravity, movement))
    gravity = gravity / np.linalg.norm(gravity, axis=1, keepdims=True)
    time_s = np.arange(len(gyro)) * decimate / rate_hz
    log.info("loaded %s: %d samples at %.1f Hz", Path(path).name, len(time_s), rate_hz / decimate)
    # optical dropouts leave NaN quaternions; those samples are excluded from evaluation
    eval_mask = (movement > 0.5) & np.isfinite(gravity).all(axis=1)
    return ImuSeries(time_s, gyro, accel), GravityTruth(time_s, gravity), eval_mask
