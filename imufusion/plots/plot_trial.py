from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imufusion.core.tilt import roll_pitch_from_accel
from imufusion.data.types import GravityTruth, ImuSeries
from imufusion.eval.metrics import inclination_error_deg

COLORS = {"accel_only": "tab:red", "gyro_only": "tab:green", "complementary": "tab:blue", "kalman": "tab:orange"}


def plot_trial(time_s: np.ndarray, series: ImuSeries, estimates: dict[str, np.ndarray], truth: GravityTruth, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    truth_angles = np.degrees(roll_pitch_from_accel(truth.gravity_in_sensor))
    raw_angles = np.degrees(roll_pitch_from_accel(series.accel_mps2))
    for idx, label in enumerate(("Roll", "Pitch")):
        ax = axes[idx]
        ax.plot(time_s, raw_angles[:, idx], color="lightgray", lw=0.6, label="raw accelerometer")
        for name, est in estimates.items():
            ax.plot(time_s, np.degrees(roll_pitch_from_accel(est))[:, idx], color=COLORS.get(name), lw=0.9, label=name)
        ax.plot(time_s, truth_angles[:, idx], "k--", lw=1.0, label="ground truth")
        ax.set_ylabel(f"{label} (deg)")
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    for name, est in estimates.items():
        axes[2].plot(time_s, inclination_error_deg(est, truth.gravity_in_sensor), color=COLORS.get(name), lw=0.8, label=name)
    axes[2].set_ylabel("Inclination error (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].set_title("Tilt estimates versus optical motion capture ground truth")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
