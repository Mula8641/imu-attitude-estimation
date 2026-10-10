# Method

Goal: estimate the direction of gravity in the sensor frame (tilt: roll and pitch) from a gyroscope and an accelerometer. Yaw is not observable without a magnetometer and is not estimated.

## Gravity vector formulation
The state is the unit vector `g` that an accelerometer at rest would read (divided by 9.81), expressed in the sensor frame. Compared with Euler angles it has no singularity at 90 degrees pitch, needs no heading, and the metric (angle between two unit vectors) is directly comparable to the optical ground truth.

## Measurement
`a_unit = a / |a|`. Roll and pitch for plotting: `roll = atan2(a_y, a_z)`, `pitch = atan2(-a_x, sqrt(a_y^2 + a_z^2))`.

## Propagation
A world-fixed vector seen from a body with angular rate `omega` changes as `dg/dt = -omega x g`. Discretely, `g_pred = g - dt * (omega x g)`, then renormalised. This is a first-order step; the renormalisation removes the small norm growth it causes.

## Estimators
1. **Accelerometer only**: `g = a_unit`.
2. **Gyroscope only**: start from the first `a_unit`, then propagate with the raw gyroscope.
3. **Complementary**: `g = normalise(alpha * propagate(g_prev) + (1 - alpha) * a_unit)`, with `alpha = tau / (tau + dt)`.
4. **Extended Kalman filter**, state `[g, b]` with `b` the gyroscope bias (`omega_true = omega - b`). Dynamics `dg/dt = -(omega - b) x g`, `db/dt = 0`. Jacobian `F = I + dt * [[-[omega]x, -[g]x], [0, 0]]`. Prediction `P = F P F^T + Q`; update with `H = [I3, 0]`, `R = r I3`, applied in Joseph form so `P` stays symmetric positive definite; `g` is renormalised after the update, a small deliberate approximation because the linear update does not keep unit length.

**Process noise convention**: `q_gravity` and `q_bias` are per-second spectral densities, and `Q = diag(q_gravity I3, q_bias I3) * dt`.

## Accelerometer gate
A sample is dynamic when `| |a| - 9.81 | > gate_ms2`. The complementary filter then uses `alpha = 1` for that sample (ignores the accelerometer); the Kalman filter multiplies `r` by `kalman_r_inflation` (1000).

## Metric
`inclination_error = arccos(g_est . g_true)`, with `g_true` the world up vector rotated into the sensor frame by the optical quaternion (`wxyz`, sensor to world). Reported as RMSE, mean and maximum over the movement phase, ignoring samples without ground truth.

## Approximations
First-order propagation; renormalisation instead of constrained estimation; gyroscope sample at the end of each interval used for that whole interval.
