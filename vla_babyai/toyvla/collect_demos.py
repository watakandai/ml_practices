"""M2: roll out the BabyAI bot and store (frames, mission, actions) per episode.

Writes three splits under <out>/<level>/:
  train.npz        bot demos on training seeds, held-out combos excluded
  test_seeds.npz   held-out seeds (disjoint seed range), training combos only
  test_combos.npz  held-out seeds whose mission names a held-out color-object combo

Every split has the same layout (see toyvla.data.EpisodeData), including the bot's
episode length per seed, which evaluation uses as the reference length.

  python -m toyvla.collect_demos --level GoToLocal --episodes 10000
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np
from minigrid.utils.baby_ai_bot import BabyAIBot

from .common import (DEFAULT_HELDOUT_COMBOS, TEST_SEED_START, TRAIN_SEED_START,
                     is_heldout_mission, make_env, parse_combos, quiet_reset)

CHUNK = 200


def rollout_bot(env, seed):
    """One bot episode. Returns (frames, actions, mission), or None if the bot fails."""
    obs, _ = quiet_reset(env, seed)
    mission = obs["mission"]
    bot = BabyAIBot(env.unwrapped)
    frames, actions = [], []
    while True:
        try:
            action = int(bot.replan())
        except Exception:  # the bot occasionally gets stuck on rare layouts
            return None
        frames.append(obs["image"])
        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    if reward <= 0:
        return None
    return np.stack(frames), np.asarray(actions, dtype=np.int8), mission


def _collect_chunk(args):
    level, tile_size, seed_start, heldout_combos = args
    env = make_env(level, tile_size)
    episodes = []
    for seed in range(seed_start, seed_start + CHUNK):
        ep = rollout_bot(env, seed)
        if ep is not None:
            frames, actions, mission = ep
            episodes.append((seed, frames, actions, mission, is_heldout_mission(mission, heldout_combos)))
    env.close()
    return episodes


def save_split(path, episodes):
    lengths = np.array([len(ep[2]) for ep in episodes], dtype=np.int64)
    np.savez_compressed(
        path,
        frames=np.concatenate([ep[1] for ep in episodes]),
        actions=np.concatenate([ep[2] for ep in episodes]),
        ep_start=np.concatenate([[0], np.cumsum(lengths)]),
        seeds=np.array([ep[0] for ep in episodes], dtype=np.int64),
        missions=np.array([ep[3] for ep in episodes]),
    )
    print("  wrote %s: %d episodes, %d steps (mean length %.1f)" % (path, len(episodes), lengths.sum(), lengths.mean()))


def collect(level, seed_start, wants, heldout_combos, tile_size, workers):
    """Run seeds from seed_start until every split in wants = {split: (keep_heldout, count)} is full."""
    out = {name: [] for name in wants}
    chunks = ((level, tile_size, s, heldout_combos) for s in itertools.count(seed_start, CHUNK))
    with mp.Pool(workers) as pool:
        for episodes in pool.imap(_collect_chunk, chunks):
            for ep in episodes:
                for name, (keep_heldout, count) in wants.items():
                    if ep[4] == keep_heldout and len(out[name]) < count:
                        out[name].append(ep)
                        break
            if all(len(out[n]) >= c for n, (_, c) in wants.items()):
                pool.terminate()
                break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", default="GoToLocal")
    p.add_argument("--episodes", type=int, default=10000, help="training episodes")
    p.add_argument("--test-episodes", type=int, default=500, help="episodes per test split")
    p.add_argument("--heldout-combos", default=",".join("%s-%s" % c for c in DEFAULT_HELDOUT_COMBOS),
                   help="comma-separated color-type pairs whose instructions are excluded from training")
    p.add_argument("--tile-size", type=int, default=8, help="pixels per grid cell; the frame is 7*tile_size square")
    p.add_argument("--out", default="data")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = p.parse_args()

    combos = parse_combos(args.heldout_combos)
    out_dir = os.path.join(args.out, args.level)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    train = collect(args.level, TRAIN_SEED_START, {"train": (False, args.episodes)}, combos, args.tile_size, args.workers)
    test = collect(args.level, TEST_SEED_START,
                   {"test_seeds": (False, args.test_episodes), "test_combos": (True, args.test_episodes)},
                   combos, args.tile_size, args.workers)
    for name, episodes in {**train, **test}.items():
        if len(episodes) == 0:
            print("  %s: no episodes (does this level name any held-out combo?)" % name)
            continue
        save_split(os.path.join(out_dir, name + ".npz"), episodes)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"level": args.level, "tile_size": args.tile_size, "heldout_combos": combos}, f, indent=2)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
