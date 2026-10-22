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
from minigrid.envs.babyai.core.verifier import GoToInstr

from .common import faced_object, make_env, mission_combos, quiet_reset, wilson_ci
from .data import load_meta, load_split

# Table column -> split file. "Seen seeds" replays seeds the policy was trained on.
EVAL_SPLITS = {"seen_seeds": "train", "heldout_seeds": "test_seeds", "heldout_combos": "test_combos"}


# no_grad: we only run the model forward, so PyTorch needn't store anything for backprop.
@torch.no_grad()
def run_episodes(policy, level, seeds, tile_size=8, batch_size=32, max_steps=None, shuffle_missions=False):
    """Roll the policy out on each seed. Returns (successes, lengths, same_target) arrays aligned
    with seeds. max_steps, if set, cuts episodes short of the level's own limit and counts them as
    failures (GoToSeq allows 2304 steps, which makes evaluating a wandering policy very slow).
    shuffle_missions is the language ablation of toyvla.diagnose: each policy is told the next
    episode's mission; same_target marks episodes where that mission names the same objects.
    Also returns wrong: on single-GoTo levels, how many non-target objects the agent turned to
    face before the episode ended (as in toyvla.diagnose); -1 on other levels."""
    successes, lengths, same_targets, wrongs = [], [], [], []
    # Run up to batch_size environments side by side, so the model sees one batch per step
    # instead of one frame at a time (much faster on a GPU).
    envs = [make_env(level, tile_size) for _ in range(min(batch_size, len(seeds)))]
    for b in range(0, len(seeds), batch_size):  # b = 0, 32, 64, ...
        chunk = seeds[b:b + batch_size]  # the seeds for this group of parallel episodes
        # reset() returns (obs, info); [0] keeps the observation. zip stops at the shorter
        # list, so the last chunk may use fewer envs than were built.
        obs = [quiet_reset(env, int(s))[0] for env, s in zip(envs, chunk)]
        missions = [o["mission"] for o in obs]
        given = missions[1:] + missions[:1] if shuffle_missions and len(missions) > 1 else missions
        same_targets.append(np.array([mission_combos(g) == mission_combos(m) for g, m in zip(given, missions)]))
        policy.reset(given)  # give the policy each episode's instruction
        running = list(range(len(chunk)))  # indices of episodes that haven't ended yet
        success = np.zeros(len(chunk), dtype=bool)
        length = np.zeros(len(chunk), dtype=np.int64)
        # A GoTo episode succeeds the moment the agent faces a matching object, so facing any
        # other object along the way means it did not head straight for the target.
        goto = [isinstance(envs[i].unwrapped.instrs, GoToInstr) for i in running]
        wrong = np.where(goto, 0, -1)
        last_faced = [None] * len(chunk)
        while running:
            # One batched decision for all still-running episodes.
            actions = policy.act(np.stack([obs[i]["image"] for i in running]), running)
            still = []
            for i, a in zip(running, actions):
                obs[i], reward, terminated, truncated, _ = envs[i].step(int(a))
                length[i] += 1
                if goto[i]:
                    obj = faced_object(envs[i].unwrapped)
                    # Count each object once per approach, not once per step spent facing it.
                    if obj is not None and obj is not last_faced[i]:
                        if not (terminated and reward > 0):
                            wrong[i] += 1
                        last_faced[i] = obj
                if terminated or truncated:
                    success[i] = reward > 0  # positive reward = mission accomplished
                elif max_steps and length[i] >= max_steps:
                    success[i] = False
                else:
                    still.append(i)  # keep stepping this one
            running = still
        successes.append(success)
        lengths.append(length)
        wrongs.append(wrong)
    for env in envs:
        env.close()
    # Join the per-chunk arrays into one array per metric, in the same order as `seeds`.
    return (np.concatenate(successes), np.concatenate(lengths), np.concatenate(same_targets),
            np.concatenate(wrongs))


def evaluate_split(policy, level, seeds, bot_lengths, **kw):
    # **kw forwards extra options (tile_size, batch_size) straight to run_episodes.
    success, length, same_target, wrong = run_episodes(policy, level, seeds, **kw)
    # Steps taken / the bot's steps on the same seed, for successful episodes only.
    ratio = [length[i] / bot_lengths[s] for i, s in enumerate(seeds) if success[i]]
    r = {
        "episodes": len(seeds),
        "success_rate": float(success.mean()),  # mean of True/False = fraction of successes
        "success_ci95": wilson_ci(int(success.sum()), len(seeds)),
        # Mean (policy length / bot length) over successful episodes; >1 means wandering.
        "length_ratio": float(np.mean(ratio)) if ratio else float("nan"),
    }
    if (wrong >= 0).all():
        # GoTo levels: solved without facing any other object first. The main GoTo metric,
        # since plain success also rewards turning to face objects until one matches.
        direct = success & (wrong == 0)
        r["direct_success_rate"] = float(direct.mean())
        r["direct_success_ci95"] = wilson_ci(int(direct.sum()), len(seeds))
    if kw.get("shuffle_missions"):
        # Success on episodes whose given mission names other objects than the true one.
        other = ~same_target
        r["success_rate_given_other_target"] = float(success[other].mean()) if other.any() else float("nan")
    return r


def evaluate_all(policy, data_dir, episodes=200, batch_size=32, splits=EVAL_SPLITS, max_steps=None,
                 shuffle_missions=False):
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
                                         tile_size=meta["tile_size"], batch_size=batch_size, max_steps=max_steps,
                                         shuffle_missions=shuffle_missions)
        r = results[column]
        print("  %-15s success %.3f [%.3f, %.3f]%s  length/bot %.2f  (%d episodes)%s"
              % (column, r["success_rate"], *r["success_ci95"],
                 "  direct %.3f [%.3f, %.3f]" % (r["direct_success_rate"], *r["direct_success_ci95"])
                 if "direct_success_rate" in r else "",
                 r["length_ratio"], r["episodes"],
                 "  given another target's mission %.3f" % r["success_rate_given_other_target"]
                 if shuffle_missions else ""))
    return results


def load_policy(kind, ckpt, device, guidance=1.0):
    # Imports sit inside the branches so evaluating the baseline never loads transformers.
    if kind == "baseline":
        if guidance != 1.0:
            raise ValueError("--guidance needs a VLA")
        from .baseline import BaselinePolicy
        return BaselinePolicy.load(ckpt, device)
    if kind == "vla":
        from .vla import VLAPolicy
        return VLAPolicy.load(ckpt, device, guidance)
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["baseline", "vla"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/GoToLocal")
    p.add_argument("--episodes", type=int, default=200, help="episodes per split")
    p.add_argument("--batch-size", type=int, default=32, help="environments run in parallel")
    p.add_argument("--max-steps", type=int, help="cut episodes at this many steps (counted as failures)")
    p.add_argument("--shuffle-missions", action="store_true",
                   help="language ablation: give the policy another episode's mission")
    p.add_argument("--guidance", type=float, default=1.0,
                   help="VLA classifier-free guidance weight (1 = none); see VLAPolicy")
    p.add_argument("--out", help="write results JSON here")
    args = p.parse_args()
    if args.out:  # create the output directory up front, before any slow work
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Use the GPU when there is one.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy(args.policy, args.ckpt, device, args.guidance)
    results = evaluate_all(policy, args.data, args.episodes, args.batch_size, max_steps=args.max_steps,
                           shuffle_missions=args.shuffle_missions)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
