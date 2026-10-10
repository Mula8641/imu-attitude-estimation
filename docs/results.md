# Results

Held-out BROAD trials (13 trials: 8 undisturbed, 5 disturbed), inclination error against optical motion capture, evaluated on the samples BROAD marks as movement with valid ground truth. RMSE columns are the mean of per-trial RMSE; the maximum is over all trials. Source: `results/metrics_heldout.json`. Parameters were fixed from `results/tuning.json` before this run, and the run was done once.

| estimator | undisturbed RMSE (deg) | disturbed RMSE (deg) | maximum error (deg) | trials |
|---|---|---|---|---|
| accel_only | 19.54 | 24.04 | 179.7 | 8 + 5 |
| gyro_only | 20.76 | 26.42 | 89.7 | 8 + 5 |
| complementary | 4.17 | 4.14 | 39.4 | 8 + 5 |
| complementary_gated | 8.28 | 5.00 | 92.4 | 8 + 5 |
| kalman | 4.51 | 3.33 | 31.1 | 8 + 5 |
| kalman_gated | 5.17 | 5.04 | 42.1 | 8 + 5 |

- The Kalman filter has the lowest disturbed RMSE (3.33 deg) and the complementary filter the lowest undisturbed RMSE (4.17 deg); both are several times better than accelerometer-only (19.54 and 24.04 deg) and gyroscope-only (20.76 and 26.42 deg).
- The accelerometer gate made things worse on average: gated complementary is 8.28 deg undisturbed and 5.00 deg disturbed versus 4.17 and 4.14 without it, and gated Kalman is 5.17 and 5.04 versus 4.51 and 3.33 (the tuned threshold is 3.0 m/s squared; I did not measure how often the gate fires, but the fast-motion trials are where the gated filters lose most, for example 44.77 versus 12.09 deg for the complementary filter on `16_undisturbed_fast_translation_B`).
- The Kalman filter is not better than the complementary filter on the undisturbed trials (4.51 versus 4.17 deg), and on individual fast-rotation trials it is clearly worse (for example 15.58 versus 9.17 deg on `07_undisturbed_fast_rotation_B`).

The Kalman parameters chosen by tuning sit at the edge of the grid (smallest `q_gravity` and `q_bias`, largest `r_measurement`), so a wider grid might do better; I did not extend it. Optical dropouts leave some trial samples without ground truth; those samples are excluded from the metrics.
