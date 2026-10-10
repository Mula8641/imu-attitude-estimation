import argparse
from pathlib import Path

import numpy as np

from imufusion.config.settings import DEFAULT_CONFIG
from imufusion.core.kalman import run_kalman
from imufusion.core.run_estimator import ESTIMATORS, run_estimator
from imufusion.data.broad_loader import load_broad_trial
from imufusion.data.synthetic import SyntheticConfig, make_synthetic_trial
from imufusion.eval.run_benchmark import QUAT_ORDER, QUAT_ROTATES
from imufusion.plots.plot_bias import plot_bias
from imufusion.plots.plot_summary import plot_summary
from imufusion.plots.plot_trial import plot_trial

SYNTHETIC_BIAS = (0.02, -0.01, 0.015)


def trial_figure(trial: Path, out_path: Path) -> None:
    series, truth, _ = load_broad_trial(trial, QUAT_ORDER, QUAT_ROTATES, 0, 3)
    estimates = {name: run_estimator(name, series, DEFAULT_CONFIG) for name in ESTIMATORS if name != "accel_only"}
    plot_trial(series.time_s, series, estimates, truth, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--undisturbed", type=Path, required=True)
    parser.add_argument("--disturbed", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics_heldout.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/img"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trial_figure(args.undisturbed, args.out / "trial_undisturbed.png")
    trial_figure(args.disturbed, args.out / "trial_disturbed.png")
    plot_summary(args.metrics, args.out / "summary_rmse.png")
    config = SyntheticConfig(120.0, 100.0, 7, 0.01, SYNTHETIC_BIAS, 0.1, 0.0)
    series, _ = make_synthetic_trial(config)
    _, bias = run_kalman(series, DEFAULT_CONFIG)
    plot_bias(series.time_s, bias, np.asarray(SYNTHETIC_BIAS), args.out / "kalman_bias_synthetic.png")


if __name__ == "__main__":
    main()
