from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_bias(time_s: np.ndarray, bias_est: np.ndarray, bias_true: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for axis, name in enumerate("xyz"):
        line, = ax.plot(time_s, bias_est[:, axis], label=f"estimated bias {name}")
        ax.axhline(bias_true[axis], color=line.get_color(), ls="--", lw=0.8, label=f"true bias {name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gyroscope bias (rad/s)")
    ax.set_title("Kalman filter bias estimate (synthetic trial)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
