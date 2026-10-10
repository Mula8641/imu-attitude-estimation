import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def group_means(per_trial: dict) -> dict[str, dict[str, float]]:
    labels = next(iter(per_trial.values())).keys()
    groups = {"undisturbed": [], "disturbed": []}
    for trial in per_trial:
        groups["disturbed" if "_disturbed_" in trial else "undisturbed"].append(trial)
    return {
        label: {g: float(np.mean([per_trial[t][label]["rmse_deg"] for t in names])) for g, names in groups.items()}
        for label in labels
    }


def plot_summary(summary_json: Path, out_path: Path) -> None:
    means = group_means(json.loads(Path(summary_json).read_text())["per_trial"])
    labels = list(means)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, group in ((-0.2, "undisturbed"), (0.2, "disturbed")):
        ax.bar(x + offset, [means[label][group] for label in labels], 0.4, label=f"{group} trials")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Mean inclination RMSE (deg, log scale)")
    ax.set_title("Held-out BROAD trials: mean RMSE per estimator")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
