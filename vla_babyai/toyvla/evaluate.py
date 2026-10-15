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


# no_grad: we only run the model forward, so PyTorch needn't store anything for backprop.
@torch.no_grad()
def run_episodes(policy, level, seeds, tile_size=8, batch_size=32, max_steps=None):
    """Roll the policy out on each seed. Returns (successes, lengths) arrays aligned with seeds.
    max_steps, if set, cuts episodes short of the level's own limit and counts them as failures
    (GoToSeq allows 2304 steps, which makes evaluating a wandering policy very slow)."""
    successes, lengths = [], []
    # Run up to batch_size environments side by side, so the model sees one batch per step
    # instead of one frame at a time (much faster on a GPU).
    envs = [make_env(level, tile_size) for _ in range(min(batch_size, len(seeds)))]
    for b in range(0, len(seeds), batch_size):  # b = 0, 32, 64, ...
        chunk = seeds[b:b + batch_size]  # the seeds for this group of parallel episodes
        # reset() returns (obs, info); [0] keeps the observation. zip stops at the shorter
        # list, so the last chunk may use fewer envs than were built.
        obs = [quiet_reset(env, int(s))[0] for env, s in zip(envs, chunk)]
        policy.reset([o["mission"] for o in obs])  # give the policy each episode's instruction
        running = list(range(len(chunk)))  # indices of episodes that haven't ended yet
        success = np.zeros(len(chunk), dtype=bool)
        length = np.zeros(len(chunk), dtype=np.int64)
        while running:
            # One batched decision for all still-running episodes.
            actions = policy.act(np.stack([obs[i]["image"] for i in running]), running)
            still = []
            for i, a in zip(running, actions):
                obs[i], reward, terminated, truncated, _ = envs[i].step(int(a))
                length[i] += 1
                if terminated or truncated:
                    success[i] = reward > 0  # positive reward = mission accomplished
                elif max_steps and length[i] >= max_steps:
                    success[i] = False
                else:
                    still.append(i)  # keep stepping this one
            running = still
        successes.append(success)
        lengths.append(length)
    for env in envs:
        env.close()
    # Join the per-chunk arrays into one array per metric, in the same order as `seeds`.
    return np.concatenate(successes), np.concatenate(lengths)


def evaluate_split(policy, level, seeds, bot_lengths, **kw):
    # **kw forwards extra options (tile_size, batch_size) straight to run_episodes.
    success, length = run_episodes(policy, level, seeds, **kw)
    # Steps taken / the bot's steps on the same seed, for successful episodes only.
    ratio = [length[i] / bot_lengths[s] for i, s in enumerate(seeds) if success[i]]
    return {
        "episodes": len(seeds),
        "success_rate": float(success.mean()),  # mean of True/False = fraction of successes
        # Mean (policy length / bot length) over successful episodes; >1 means wandering.
        "length_ratio": float(np.mean(ratio)) if ratio else float("nan"),
    }


def evaluate_all(policy, data_dir, episodes=200, batch_size=32, splits=EVAL_SPLITS, max_steps=None):
    meta = load_meta(data_dir)  # level name and tile size the data was built with
    results = {}
    for column, split in splits.items():
        path = os.path.join(data_dir, split + ".npz")
        if not os.path.exists(path):  # e.g. a level with no held-out combos
            continue
        data = load_split(data_dir, split)
        # Only the seeds are needed: the environment regenerates each level from its seed.
        seeds = data.seeds[:episodes].tolist()
        results[column] = evaluate_split(policy, meta["level"], seeds, data.bot_lengths(),
                                         tile_size=meta["tile_size"], batch_size=batch_size, max_steps=max_steps)
        r = results[column]
        print("  %-15s success %.3f  length/bot %.2f  (%d episodes)"
              % (column, r["success_rate"], r["length_ratio"], r["episodes"]))
    return results


def load_policy(kind, ckpt, device):
    # Imports sit inside the branches so evaluating the baseline never loads transformers.
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
    p.add_argument("--max-steps", type=int, help="cut episodes at this many steps (counted as failures)")
    p.add_argument("--out", help="write results JSON here")
    args = p.parse_args()
    if args.out:  # create the output directory up front, before any slow work
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Use the GPU when there is one.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy(args.policy, args.ckpt, device)
    results = evaluate_all(policy, args.data, args.episodes, args.batch_size, max_steps=args.max_steps)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
