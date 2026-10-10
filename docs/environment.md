# Environment and data notes

## Software
Python 3.14.4, numpy 2.5.3, matplotlib 3.11.2, pytest 9.1.1, h5py (HDF5 reading). The code only needs Python 3.11 or newer.

## Dataset: BROAD
- Source: https://github.com/dlaidig/broad (TU Berlin), cloned 2026-10-10. Licence: Creative Commons Attribution 4.0 (CC BY 4.0). Only the HDF5 trial files are copied to `data/broad/`, which is git ignored.
- Format: one HDF5 file per trial, 39 trials (23 undisturbed, names containing `_undisturbed_`; 16 disturbed, `_disturbed_`).
- Keys: `imu_acc` (m/s squared), `imu_gyr` (rad/s), `imu_mag` (microtesla, unused), `opt_quat` (unit quaternion, w first, relative to an ENU frame), `opt_pos` (m, unused), `movement` (boolean mask of the movement phase). Attribute `sampling_rate` is 285.714 Hz (about 286 Hz).
- One IMU per trial, so `imu_index` is always 0. The optical data is already synchronised and aligned to the IMU sample grid by the dataset authors, so no resampling is done.
- Optical dropouts: many trials contain NaN quaternions. These samples are excluded from the metrics (the filters still see every IMU sample).

## Quaternion convention (static check)
Compared the mean accelerometer direction over a 1 s window with gyroscope magnitude below 0.02 rad/s against the four convention candidates:

| trial | wxyz sensor_to_world | wxyz world_to_sensor | xyzw sensor_to_world | xyzw world_to_sensor |
|---|---|---|---|---|
| 01_undisturbed_slow_rotation_A | 0.20 deg | 5.13 deg | 177.74 deg | 178.10 deg |
| 10_undisturbed_slow_translation_A | 0.19 deg | 5.16 deg | 177.75 deg | 178.10 deg |

Chosen: `order = wxyz`, `rotates = sensor_to_world`, `imu_index = 0`. Only two trials had a qualifying static window (the others never hold the gyroscope magnitude under the threshold for a full second), not the three the check asked for. As a further check, over whole trials (valid samples only) `sensor_to_world` has the lower median accelerometer-to-truth angle than `world_to_sensor` on 28, 16, 07 and 31 (for example 8.44 versus 21.44 deg on trial 28).

## Deviations from the plan, and why
- The loader returns a third value, a boolean evaluation mask (movement phase and valid ground truth), because dropouts and the movement phase are needed by the metric.
- The loader can decimate by block averaging. The benchmark uses a factor of 3 (95.2 Hz) and tuning a factor of 6 (47.6 Hz) to keep run time reasonable in pure Python.
- The synthetic yaw rate is 0.02 rad/s. At 0.2 rad/s the yaw rotation averages a constant body-frame gyroscope bias out and the gyroscope-only error no longer grows.
- The shake burst in the synthetic generator is perpendicular to gravity, so it actually corrupts the tilt direction.
- Held-out split has 13 trials (every third file starting from the first); tuning has 6 (every sixth starting from the third). The rule was fixed before any filter was run.
- Tuning picks `gate_ms2` with the complementary filter and reuses it for the Kalman grid, which keeps the search at 20 + 27 runs rather than a full product.
- The synthetic test of gyroscope integration uses the mean of two consecutive gyroscope samples; using only the end-of-interval sample adds a half-sample lag worth about 0.15 degrees over 60 s. The filters use the sample at the end of each interval, as in the method description.
- Phase 6 (C++ port) was skipped.
