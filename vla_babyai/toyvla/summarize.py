"""Combine the results.json of several training runs of one variant (different training seeds,
held-out combo sets, or both) into a mean with a 95% confidence interval across runs.

  python -m toyvla.summarize runs/calib_A/GoToLocal runs/calib_B/GoToLocal runs/calib_C/GoToLocal

Each argument is a run directory (with results.json) or a results.json file. The interval is
mean +- t * sd / sqrt(n) over the runs, so it covers run-to-run spread, which on held-out combos
is several times the per-run evaluation noise (each run's own Wilson interval in results.json).
"""
import argparse
import json
import math
import os

import numpy as np

# Two-sided 95% Student t quantiles by degrees of freedom (runs - 1).
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
METRICS = ["success_rate", "direct_success_rate", "length_ratio"]


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    with open(path) as f:
        return json.load(f)


def across_runs(values):
    """(mean, half-width of the 95% interval) over runs; half-width is NaN for a single run."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan")
    t = T95.get(len(values) - 1, 1.96)
    return mean, t * float(values.std(ddof=1)) / math.sqrt(len(values))


def summarize(results):
    out = {}
    for split in results[0]:
        out[split] = {}
        for m in METRICS:
            values = [r[split][m] for r in results if split in r and m in r[split]]
            if len(values) == len(results):
                mean, half = across_runs(values)
                out[split][m] = {"mean": mean, "ci95": half, "runs": values}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories or results.json files")
    p.add_argument("--out", help="write the summary JSON here")
    args = p.parse_args()

    summary = summarize([load(r) for r in args.runs])
    for split, metrics in summary.items():
        for m, s in metrics.items():
            print("%-15s %-20s %.3f +- %.3f   runs: %s"
                  % (split, m, s["mean"], s["ci95"], " ".join("%.3f" % v for v in s["runs"])))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
