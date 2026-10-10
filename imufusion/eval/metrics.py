import numpy as np


def inclination_error_deg(gravity_est: np.ndarray, gravity_true: np.ndarray) -> np.ndarray:
    cosine = np.clip(np.sum(gravity_est * gravity_true, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def summarize(error_deg: np.ndarray) -> dict[str, float]:
    return {
        "rmse_deg": float(np.sqrt(np.mean(error_deg**2))),
        "max_deg": float(np.max(error_deg)),
        "mean_deg": float(np.mean(error_deg)),
    }
