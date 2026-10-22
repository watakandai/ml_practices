"""M2: roll out the BabyAI bot and store (frames, mission, actions) per episode.

Writes four splits under <out>/<level>/:
  train.npz        bot demos on training seeds, held-out combos excluded
  val.npz          validation seeds (disjoint range), training combos only; for model selection
  test_seeds.npz   held-out seeds (disjoint seed range), training combos only
  test_combos.npz  held-out seeds whose mission names a held-out color-object combo

Every split has the same layout (see toyvla.data.EpisodeData), including the bot's
episode length per seed, which evaluation uses as the reference length.

  python -m toyvla.collect_demos --level GoToLocal --episodes 10000
  python -m toyvla.collect_demos --level GoToLocal --splits val   # add val to an existing data dir
  python -m toyvla.collect_demos --heldout-combos blue-ball,purple-key,red-box,grey-ball --out data/heldout_B
"""
import argparse  # command-line flags
import json
import multiprocessing as mp  # run several bots in parallel on different CPU cores
import os
import time

import numpy as np
# A scripted expert that plans a path to solve any BabyAI mission; our "teacher" for imitation.
from minigrid.utils.baby_ai_bot import BabyAIBot

from .common import (DEFAULT_HELDOUT_COMBOS, TEST_SEED_START, TRAIN_SEED_START, VAL_SEED_START,
                     is_heldout_mission, make_env, parse_combos, quiet_reset)

# Each worker process handles this many consecutive seeds per task.
CHUNK = 200


def rollout_bot(env, seed):
    """One bot episode. Returns (frames, actions, mission), or None if the bot fails."""
    obs, _ = quiet_reset(env, seed)  # build the level for this seed
    mission = obs["mission"]  # e.g. "go to the red ball"
    # The bot reads the true environment state (env.unwrapped skips the wrappers).
    bot = BabyAIBot(env.unwrapped)
    frames, actions = [], []
    while True:
        try:
            action = int(bot.replan())  # the bot's next action, as an integer 0..6
        except Exception:  # the bot occasionally gets stuck on rare layouts
            return None
        # Record what the agent saw *before* acting, paired with the action taken:
        # this (observation, action) pair is one training example.
        frames.append(obs["image"])
        actions.append(action)
        # Apply the action. terminated = mission solved (or failed for good);
        # truncated = ran out of the level's step budget.
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    if reward <= 0:  # BabyAI gives a positive reward only on success; drop failed demos
        return None
    # np.stack turns the list of (56, 56, 3) frames into one (T, 56, 56, 3) array.
    # int8 is enough for 7 actions and keeps the file small.
    return np.stack(frames), np.asarray(actions, dtype=np.int8), mission


def _collect_chunk(args):
    # Runs inside a worker process; the arguments arrive packed in one tuple
    # because pool.imap passes a single object per task.
    level, tile_size, seed_start, heldout_combos = args
    env = make_env(level, tile_size)  # each process builds its own environment
    episodes = []
    for seed in range(seed_start, seed_start + CHUNK):
        ep = rollout_bot(env, seed)
        if ep is not None:
            frames, actions, mission = ep
            # Tag each episode with whether its mission names a held-out combo,
            # so the main process can route it to the right split.
            episodes.append((seed, frames, actions, mission, is_heldout_mission(mission, heldout_combos)))
    env.close()
    return episodes


def save_split(path, episodes):
    # ep = (seed, frames, actions, mission, heldout); ep[2] is the action array.
    lengths = np.array([len(ep[2]) for ep in episodes], dtype=np.int64)
    np.savez_compressed(  # one zip-compressed file holding several named arrays
        path,
        # Concatenate all episodes back to back into single arrays...
        frames=np.concatenate([ep[1] for ep in episodes]),
        actions=np.concatenate([ep[2] for ep in episodes]),
        # ...and record where each one starts: [0, len0, len0+len1, ...].
        ep_start=np.concatenate([[0], np.cumsum(lengths)]),
        seeds=np.array([ep[0] for ep in episodes], dtype=np.int64),
        missions=np.array([ep[3] for ep in episodes]),
    )
    print("  wrote %s: %d episodes, %d steps (mean length %.1f)" % (path, len(episodes), lengths.sum(), lengths.mean()))


def collect(level, seed_start, wants, heldout_combos, tile_size, workers):
    """Run seeds from seed_start until every split in wants = {split: (keep_heldout, count)} is full."""
    out = {name: [] for name in wants}  # split name -> list of episodes collected so far
    start = seed_start
    with mp.Pool(workers) as pool:  # a pool of worker processes
        while not all(len(out[n]) >= c for n, (_, c) in wants.items()):
            # One round of `workers` chunks at a time. Every round runs to completion: terminating
            # the pool while workers are still sending back their (large) results can deadlock.
            chunks = [(level, tile_size, s, heldout_combos) for s in range(start, start + workers * CHUNK, CHUNK)]
            start += workers * CHUNK
            # imap hands chunks to workers and yields their results in seed order.
            for episodes in pool.imap(_collect_chunk, chunks):
                for ep in episodes:
                    # Put the episode in the first split that wants this kind (held out or not)
                    # and still has room.
                    for name, (keep_heldout, count) in wants.items():
                        if ep[4] == keep_heldout and len(out[name]) < count:
                            out[name].append(ep)
                            break
    return out


def main():
    # Command-line options; each --flag becomes args.flag (dashes turn into underscores).
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", default="GoToLocal")
    p.add_argument("--episodes", type=int, default=10000, help="training episodes")
    p.add_argument("--test-episodes", type=int, default=500, help="episodes per test split")
    p.add_argument("--val-episodes", type=int, default=500)
    p.add_argument("--splits", default="train,val,test",
                   help="which to collect: train, val, test (test = test_seeds and test_combos)")
    p.add_argument("--heldout-combos", default=",".join("%s-%s" % c for c in DEFAULT_HELDOUT_COMBOS),
                   help="comma-separated color-type pairs whose instructions are excluded from training")
    p.add_argument("--tile-size", type=int, default=8, help="pixels per grid cell; the frame is 7*tile_size square")
    p.add_argument("--out", default="data")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))  # leave one core free
    args = p.parse_args()

    combos = parse_combos(args.heldout_combos)  # "red-ball,..." -> [("red", "ball"), ...]
    out_dir = os.path.join(args.out, args.level)  # e.g. data/GoToLocal
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    splits = args.splits.split(",")
    meta_path = os.path.join(out_dir, "meta.json")
    if "train" not in splits and os.path.exists(meta_path):
        # Adding splits to existing data: they must use the same held-out combos and tile size.
        with open(meta_path) as f:
            meta = json.load(f)
        if [tuple(c) for c in meta["heldout_combos"]] != combos or meta["tile_size"] != args.tile_size:
            raise SystemExit("%s was collected with held-out combos %s and tile size %d"
                             % (out_dir, meta["heldout_combos"], meta["tile_size"]))
    out = {}
    if "train" in splits:
        # Training data: seeds 0, 1, 2, ... keeping only missions that do NOT name a held-out combo.
        out.update(collect(args.level, TRAIN_SEED_START, {"train": (False, args.episodes)}, combos,
                           args.tile_size, args.workers))
    if "val" in splits:
        # Validation data: seeds from 9,000,000 up, training combos only, so selecting a checkpoint
        # on it never looks at the test levels or the held-out combos.
        out.update(collect(args.level, VAL_SEED_START, {"val": (False, args.val_episodes)}, combos,
                           args.tile_size, args.workers))
    if "test" in splits:
        # Test data: seeds from 10,000,000 up, split by whether the mission names a held-out combo.
        out.update(collect(args.level, TEST_SEED_START,
                           {"test_seeds": (False, args.test_episodes), "test_combos": (True, args.test_episodes)},
                           combos, args.tile_size, args.workers))
    for name, episodes in out.items():
        if len(episodes) == 0:
            print("  %s: no episodes (does this level name any held-out combo?)" % name)
            continue
        save_split(os.path.join(out_dir, name + ".npz"), episodes)

    # Record how the data was made, so evaluation can rebuild exactly the same environments.
    with open(meta_path, "w") as f:
        json.dump({"level": args.level, "tile_size": args.tile_size, "heldout_combos": combos}, f, indent=2)
    print("done in %.0fs" % (time.time() - t0))


# Runs main() only when the file is executed (python -m toyvla.collect_demos), not when imported.
if __name__ == "__main__":
    main()
