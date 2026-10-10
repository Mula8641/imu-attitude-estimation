import argparse
import dataclasses
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from imufusion.config.settings import DEFAULT_CONFIG, ComplementaryConfig, FilterConfig, KalmanConfig
from imufusion.core.run_estimator import run_estimator
from imufusion.data.broad_loader import load_broad_trial
from imufusion.eval.metrics import inclination_error_deg, summarize
from imufusion.eval.run_benchmark import QUAT_ORDER, QUAT_ROTATES, read_trial_list
from imufusion.utils.log import get_logger

log = get_logger(__name__)
GRID = {
    "tau_s": [0.3, 0.5, 1.0, 2.0, 4.0],
    "gate_ms2": [0.5, 1.0, 1.5, 3.0],
    "q_gravity": [1e-8, 1e-6, 1e-4],
    "q_bias": [1e-10, 1e-8, 1e-6],
    "r_measurement": [1e-4, 1e-3, 1e-2],
}


def load_trials(paths: list[Path], decimate: int) -> list[tuple]:
    return [load_broad_trial(p, QUAT_ORDER, QUAT_ROTATES, 0, decimate) for p in paths]


def mean_rmse(name: str, config: FilterConfig, trials: list[tuple]) -> float:
    rmses = []
    for series, truth, mask in trials:
        estimate = run_estimator(name, series, config)
        rmses.append(summarize(inclination_error_deg(estimate[mask], truth.gravity_in_sensor[mask]))["rmse_deg"])
    return float(np.mean(rmses))


def tune_complementary(trials: list[tuple]) -> tuple[list[dict], dict]:
    rows = []
    for tau, gate in itertools.product(GRID["tau_s"], GRID["gate_ms2"]):
        config = dataclasses.replace(
            DEFAULT_CONFIG,
            complementary=ComplementaryConfig(tau),
            gate=dataclasses.replace(DEFAULT_CONFIG.gate, gate_ms2=gate),
        )
        rows.append({"tau_s": tau, "gate_ms2": gate, "mean_rmse_deg": mean_rmse("complementary", config, trials)})
        log.info("complementary %s", rows[-1])
    return rows, min(rows, key=lambda r: r["mean_rmse_deg"])


def tune_kalman(trials: list[tuple], gate_ms2: float) -> tuple[list[dict], dict]:
    rows = []
    gate = dataclasses.replace(DEFAULT_CONFIG.gate, gate_ms2=gate_ms2)
    for qg, qb, r in itertools.product(GRID["q_gravity"], GRID["q_bias"], GRID["r_measurement"]):
        kalman = KalmanConfig(qg, qb, r, DEFAULT_CONFIG.kalman.p0_gravity, DEFAULT_CONFIG.kalman.p0_bias)
        config = dataclasses.replace(DEFAULT_CONFIG, gate=gate, kalman=kalman)
        rows.append({"q_gravity": qg, "q_bias": qb, "r_measurement": r, "mean_rmse_deg": mean_rmse("kalman", config, trials)})
        log.info("kalman %s", rows[-1])
    return rows, min(rows, key=lambda r: r["mean_rmse_deg"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--decimate", type=int, default=6)
    args = parser.parse_args(argv)
    if args.split == "heldout":
        print("refusing to tune on the held-out split", file=sys.stderr)
        return 2
    trials = load_trials(read_trial_list(args.trials), args.decimate)
    comp_rows, comp_best = tune_complementary(trials)
    kal_rows, kal_best = tune_kalman(trials, comp_best["gate_ms2"])
    result = {
        "split": args.split,
        "decimate": args.decimate,
        "grid": GRID,
        "note": "gate_ms2 chosen with the complementary filter, then reused for the Kalman grid",
        "complementary": {"rows": comp_rows, "best": comp_best},
        "kalman": {"rows": kal_rows, "best": kal_best},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tuning.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"complementary": comp_best, "kalman": kal_best}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
