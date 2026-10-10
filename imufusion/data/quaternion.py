import numpy as np

ORDERS = ("wxyz", "xyzw")
ROTATIONS = ("sensor_to_world", "world_to_sensor")


def gravity_in_sensor_from_quaternion(quat: np.ndarray, order: str, rotates: str) -> np.ndarray:
    if order not in ORDERS:
        raise ValueError(f"unknown quaternion order: {order}")
    if rotates not in ROTATIONS:
        raise ValueError(f"unknown rotation convention: {rotates}")
    q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    w, x, y, z = (q[:, i] for i in (0, 1, 2, 3)) if order == "wxyz" else (q[:, 3], q[:, 0], q[:, 1], q[:, 2])
    # third row of R(q) is world z expressed in the sensor frame when q maps sensor to world
    row = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1)
    # third column of R(q) is the same vector when q maps world to sensor
    col = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)
    return row if rotates == "sensor_to_world" else col
