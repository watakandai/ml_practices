"""Success rate and episode length vs. the bot, on the three evaluation splits.

A policy implements:
  reset(missions)      start len(missions) new episodes
  act(images, idx)     actions for the running episodes idx (images[j] is episode idx[j]'s frame)

  python -m toyvla.evaluate --policy baseline --ckpt runs/baseline/GoToLocal/model.pt
  python -m toyvla.evaluate --policy vla --ckpt runs/vla/GoToLocal/final
"""
import argparse
import json
import os

import numpy as np
import torch

from .common import make_env, quiet_reset
from .data import load_meta, load_split

# Table column -> split file. "Seen seeds" replays seeds the policy was trained on.
EVAL_SPLITS = {"seen_seeds": "train", "heldout_seeds": "test_seeds", "heldout_combos": "test_combos"}


@torch.no_grad()
def run_episodes(policy, level, seeds, tile_size=8, batch_size=32):
    """Roll the policy out on each seed. Returns (successes, lengths) arrays aligned with seeds."""
    successes, lengths = [], []
    envs = [make_env(level, tile_size) for _ in range(min(batch_size, len(seeds)))]
    for b in range(0, len(seeds), batch_size):
        chunk = seeds[b:b + batch_size]
        obs = [quiet_reset(env, int(s))[0] for env, s in zip(envs, chunk)]
        policy.reset([o["mission"] for o in obs])
        running = list(range(len(chunk)))
        success = np.zeros(len(chunk), dtype=bool)
        length = np.zeros(len(chunk), dtype=np.int64)
        while running:
            actions = policy.act(np.stack([obs[i]["image"] for i in running]), running)
            still = []
            for i, a in zip(running, actions):
                obs[i], reward, terminated, truncated, _ = envs[i].step(int(a))
                length[i] += 1
                if terminated or truncated:
                    success[i] = reward > 0
                else:
                    still.append(i)
            running = still
        successes.append(success)
        lengths.append(length)
    for env in envs:
        env.close()
    return np.concatenate(successes), np.concatenate(lengths)


def evaluate_split(policy, level, seeds, bot_lengths, **kw):
    success, length = run_episodes(policy, level, seeds, **kw)
    ratio = [length[i] / bot_lengths[s] for i, s in enumerate(seeds) if success[i]]
    return {
        "episodes": len(seeds),
        "success_rate": float(success.mean()),
        # Mean (policy length / bot length) over successful episodes; >1 means wandering.
        "length_ratio": float(np.mean(ratio)) if ratio else float("nan"),
    }


def evaluate_all(policy, data_dir, episodes=200, batch_size=32, splits=EVAL_SPLITS):
    meta = load_meta(data_dir)
    results = {}
    for column, split in splits.items():
        path = os.path.join(data_dir, split + ".npz")
        if not os.path.exists(path):
            continue
        data = load_split(data_dir, split)
        seeds = data.seeds[:episodes].tolist()
        results[column] = evaluate_split(policy, meta["level"], seeds, data.bot_lengths(),
                                         tile_size=meta["tile_size"], batch_size=batch_size)
        r = results[column]
        print("  %-15s success %.3f  length/bot %.2f  (%d episodes)"
              % (column, r["success_rate"], r["length_ratio"], r["episodes"]))
    return results


def load_policy(kind, ckpt, device):
    if kind == "baseline":
        from .baseline import BaselinePolicy
        return BaselinePolicy.load(ckpt, device)
    if kind == "vla":
        from .vla import VLAPolicy
        return VLAPolicy.load(ckpt, device)
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["baseline", "vla"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/GoToLocal")
    p.add_argument("--episodes", type=int, default=200, help="episodes per split")
    p.add_argument("--batch-size", type=int, default=32, help="environments run in parallel")
    p.add_argument("--out", help="write results JSON here")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy(args.policy, args.ckpt, device)
    results = evaluate_all(policy, args.data, args.episodes, args.batch_size)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
