from pathlib import Path

import h5py
import numpy as np
import pytest

from imufusion.data.broad_loader import list_broad_trials, load_broad_trial

RATE_HZ = 100.0
N = 400


def write_trial(path: Path, drop: str | None = None, nan_rows: slice | None = None) -> None:
    rng = np.random.default_rng(0)
    data = {
        "imu_acc": np.tile([0.0, 0.0, 9.81], (N, 1)) + rng.normal(0, 0.01, (N, 3)),
        "imu_gyr": rng.normal(0, 0.01, (N, 3)),
        "opt_quat": np.tile([1.0, 0, 0, 0], (N, 1)),
        "movement": np.arange(N) >= 100,
    }
    if nan_rows is not None:
        data["opt_quat"][nan_rows] = np.nan
    with h5py.File(path, "w") as f:
        f.attrs["sampling_rate"] = RATE_HZ
        for key, value in data.items():
            if key != drop:
                f[key] = value


def test_units_time_and_lengths(tmp_path):
    write_trial(tmp_path / "t.hdf5")
    series, truth, mask = load_broad_trial(tmp_path / "t.hdf5", "wxyz", "sensor_to_world", 0)
    assert len(series.time_s) == len(truth.time_s) == N == len(mask)
    assert np.isclose(series.time_s[1], 1 / RATE_HZ) and np.allclose(truth.gravity_in_sensor, [0, 0, 1])
    assert np.isclose(np.linalg.norm(series.accel_mps2, axis=1).mean(), 9.81, atol=0.05)


def test_decimation_shortens_series(tmp_path):
    write_trial(tmp_path / "t.hdf5")
    series, truth, _ = load_broad_trial(tmp_path / "t.hdf5", "wxyz", "sensor_to_world", 0, decimate=4)
    assert len(series.time_s) == N // 4 and np.isclose(series.time_s[1], 0.04)


def test_dropouts_are_masked_out(tmp_path):
    write_trial(tmp_path / "t.hdf5", nan_rows=slice(200, 220))
    _, _, mask = load_broad_trial(tmp_path / "t.hdf5", "wxyz", "sensor_to_world", 0)
    assert mask[150] and not mask[210] and not mask[50]


def test_missing_key_raises(tmp_path):
    write_trial(tmp_path / "t.hdf5", drop="imu_gyr")
    with pytest.raises(ValueError, match="imu_gyr"):
        load_broad_trial(tmp_path / "t.hdf5", "wxyz", "sensor_to_world", 0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_broad_trial(tmp_path / "nope.hdf5", "wxyz", "sensor_to_world", 0)


def test_list_trials_sorted(tmp_path):
    for name in ("b.hdf5", "a.hdf5"):
        write_trial(tmp_path / name)
    assert [p.name for p in list_broad_trials(tmp_path)] == ["a.hdf5", "b.hdf5"]
