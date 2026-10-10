import numpy as np

STANDARD_GRAVITY = 9.81


def is_dynamic(accel_mps2: np.ndarray, gate_ms2: float) -> bool:
    return bool(abs(np.linalg.norm(accel_mps2) - STANDARD_GRAVITY) > gate_ms2)
