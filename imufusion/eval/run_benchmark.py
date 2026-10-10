import argparse
import dataclasses
import datetime
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

from imufusion.config.settings import DEFAULT_CONFIG, FilterConfig
from imufusion.core.run_estimator import run_estimator
from imufusion.data.broad_loader import load_broad_trial
from imufusion.eval.metrics import inclination_error_deg, summarize
from imufusion.utils.log import get_logger

log = get_logger(__name__)
QUAT_ORDER, QUAT_ROTATES = "wxyz", "sensor_to_world"
# (label, estimator, gate on)
VARIANTS = (
    ("accel_only", "accel_only", False),
    ("gyro_only", "gyro_only", False),
    ("complementary", "complementary", False),
    ("complementary_gated", "complementary", True),
    ("kalman", "kalman", False),
    ("kalman_gated", "kalman", True),
)


def with_gate(config: FilterConfig, use_gate: bool) -> FilterConfig:
    return dataclasses.replace(config, gate=dataclasses.replace(config.gate, use_gate=use_gate))


def evaluate_trial(path: Path, config: FilterConfig, decimate: int) -> dict[str, dict[str, float]]:
    series, truth, mask = load_broad_trial(path, QUAT_ORDER, QUAT_ROTATES, 0, decimate)
    results = {}
    for label, name, use_gate in VARIANTS:
        estimate = run_estimator(name, series, with_gate(config, use_gate))
        error = inclination_error_deg(estimate[mask], truth.gravity_in_sensor[mask])
        results[label] = summarize(error)
        log.info("%s %s rmse=%.2f deg", path.name, label, results[label]["rmse_deg"])
    return results


def read_trial_list(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def run(trials: list[Path], config: FilterConfig, decimate: int) -> dict:
    per_trial, skipped = {}, []
    for trial in trials:
        try:
            per_trial[trial.stem] = evaluate_trial(trial, config, decimate)
        except (FileNotFoundError, ValueError, OSError) as err:
            log.error("skipping %s: %s", trial, err)
            skipped.append({"trial": str(trial), "reason": str(err)})
    return {"per_trial": per_trial, "trials_skipped": skipped}


def markdown_table(per_trial: dict) -> str:
    lines = ["| trial | " + " | ".join(v[0] for v in VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for trial, res in per_trial.items():
        lines.append(f"| {trial} | " + " | ".join(f"{res[v[0]]['rmse_deg']:.2f}" for v in VARIANTS) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True, help="text file, one trial path per line")
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--decimate", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="only the first N trials (dry run)")
    args = parser.parse_args(argv)
    trials = read_trial_list(args.trials)
    trials = trials[: args.limit] if args.limit else trials
    outcome = run(trials, DEFAULT_CONFIG, args.decimate)
    outcome.update(
        split=args.split,
        config=dataclasses.asdict(DEFAULT_CONFIG),
        decimate=args.decimate,
        versions={"numpy": np.__version__, "matplotlib": matplotlib.__version__, "python": sys.version.split()[0]},
        date=datetime.date.today().isoformat(),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"metrics_{args.split}.json").write_text(json.dumps(outcome, indent=2))
    print(markdown_table(outcome["per_trial"]))
    return 1 if outcome["trials_skipped"] else 0


if __name__ == "__main__":
    sys.exit(main())
