"""M1: convert tracking data into Match files and build-up training samples.

    python -m fpcoach.prepare_data --pff-root /path/to/pff --out data/pff          # every game
    python -m fpcoach.prepare_data --pff-root /path/to/pff --games 10517 3812 --out data/pff
    python -m fpcoach.prepare_data --synthetic --out data/synthetic                # no data needed

Writes, per game: `<id>.match.npz` (the Match), `<id>.samples.npz` (build-up samples), and one
`summary.json` with the train/test split and label statistics, to sanity-check alignment.
"""
import argparse
import json
import os

import numpy as np

from .buildup import build_ups, match_split
from .grf import A, ACTION_NAMES, NUM_ACTIONS
from .labels import label_match
from .match import velocities
from .samples import match_samples


def process(match, out_dir):
    vel = velocities(match.pos, match.t, match.period)
    labels = label_match(match, vel)
    samples = match_samples(match, labels)
    match.save(os.path.join(out_dir, "%s.match.npz" % match.match_id))
    if samples:
        np.savez_compressed(os.path.join(out_dir, "%s.samples.npz" % match.match_id), **samples)

    shots = [e for e in labels["events"] if e[2] == A["shot"]]
    bus = build_ups(match, shots)
    hist = np.zeros(NUM_ACTIONS, dtype=int)
    if samples:
        a = samples["action"][~samples["own_gk"]]  # outfield players only, as in training
        hist = np.bincount(a[a >= 0], minlength=NUM_ACTIONS)
    return {
        "match_id": match.match_id,
        "split": match_split(match.match_id),
        "frames": int(match.num_frames),
        "build_ups": len(bus),
        "build_ups_reaching_final_third": int(sum(b.reached_final_third for b in bus)),
        "samples": int(len(samples["frame"])) if samples else 0,
        "ball_events": {ACTION_NAMES[k]: int(sum(e[2] == k for e in labels["events"]))
                        for k in (A["short_pass"], A["long_pass"], A["high_pass"], A["shot"])},
        "outfield_action_counts": {ACTION_NAMES[i]: int(c) for i, c in enumerate(hist) if c},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pff-root", help="directory with PFF's metadata/, rosters/ and tracking/")
    ap.add_argument("--games", nargs="*", help="game ids; default: every file in <pff-root>/metadata")
    ap.add_argument("--synthetic", action="store_true", help="use the scripted build-up instead")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.synthetic:
        from .synthetic import scripted_match
        matches = [scripted_match]
    else:
        from .load_pff import load_pff
        if not args.pff_root:
            ap.error("--pff-root or --synthetic is required")
        games = args.games or sorted(f.split(".")[0] for f in os.listdir(os.path.join(args.pff_root, "metadata")))
        matches = [lambda g=g: load_pff(args.pff_root, g) for g in games]

    summary = []
    for make in matches:
        match = make()
        s = process(match, args.out)
        print(json.dumps(s))
        summary.append(s)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
