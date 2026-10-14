"""Diagnose how a GoTo policy succeeds or fails: does it head for the instructed object, or roam
until it happens to face it?

For every episode it records which objects the agent faced along the way (a GoTo episode ends
with success the moment the agent faces a matching object; facing any other object does
nothing), whether a target was already in view at the start, and how each wrong object relates
to the instruction (same color, same type, or neither). The summary separates:
  - grounding failures: a target was in view, yet the agent faced wrong objects first
  - compositional confusion: wrong objects that share the instruction's color or type
  - exploration: no target in view at the start, so some searching is expected

  python -m toyvla.diagnose --policy vla --ckpt runs/vla256/GoToLocal/final
  python -m toyvla.diagnose --policy baseline --ckpt runs/baseline/GoToLocal/model.pt
  python -m toyvla.diagnose --policy bot    # reference: the bot's own demos

Writes <out>/<split>_episodes.jsonl and <out>/<split>_summary.json.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from .common import make_env, quiet_reset
from .data import load_meta, load_split
from .evaluate import EVAL_SPLITS, load_policy

# Things a mission can name; walls and empty floor are not objects.
OBJECT_TYPES = ("ball", "box", "key", "door")


def faced_object(u):
    """The object in the cell directly in front of the agent, or None."""
    # u is the raw MiniGrid env. front_pos = (x, y) of the cell the agent faces;
    # grid.get returns whatever is there (None for empty floor).
    obj = u.grid.get(*u.front_pos)
    return obj if obj is not None and obj.type in OBJECT_TYPES else None


def relation(obj, desc):
    """How a wrong object relates to the instruction's object description."""
    # desc is BabyAI's parsed instruction: desc.type ("ball"), desc.color ("red", or None
    # when the mission names no color).
    same_type = obj.type == desc.type
    same_color = desc.color is not None and obj.color == desc.color
    if same_type and same_color:
        return "match"  # only possible with a location qualifier; not the case in GoToLocal
    return "same_type" if same_type else "same_color" if same_color else "neither"


@torch.no_grad()
def run(policy, level, tile_size, seeds, bot_lengths, batch_size=32):
    """Roll out the policy on each seed and return one record (dict) per episode.

    The batching works like evaluate.run_episodes, with extra bookkeeping at every step.
    """
    records = []
    envs = [make_env(level, tile_size) for _ in range(min(batch_size, len(seeds)))]
    for b in range(0, len(seeds), batch_size):
        chunk = seeds[b:b + batch_size]
        obs = [quiet_reset(env, int(s))[0] for env, s in zip(envs, chunk)]
        recs = []
        for env, s, o in zip(envs, chunk, obs):
            u = env.unwrapped  # the raw MiniGrid env, which knows the full level state
            desc = u.instrs.desc  # the instruction's target: color, type, and matching objects
            recs.append({
                "seed": int(s), "mission": o["mission"], "target": [desc.color, desc.type],
                "bot_steps": int(bot_lengths[s]),
                # desc.obj_poss lists the (x, y) of every object matching the instruction;
                # in_view says whether a cell is inside the agent's 7x7 view.
                "target_in_view_at_start": any(bool(u.in_view(*pos)) for pos in desc.obj_poss),
                # faced: every object the agent turned to face, in order.
                # wrong: how each non-final faced object relates to the target.
                "faced": [], "wrong": [], "steps": 0, "success": False,
            })
        policy.reset([o["mission"] for o in obs])
        running = list(range(len(chunk)))
        while running:
            actions = policy.act(np.stack([obs[i]["image"] for i in running]), running)
            still = []
            for i, a in zip(running, actions):
                obs[i], reward, terminated, truncated, _ = envs[i].step(int(a))
                r = recs[i]
                r["steps"] += 1
                obj = faced_object(envs[i].unwrapped)  # what the agent faces after this action
                name = "%s %s" % (obj.color, obj.type) if obj else None  # e.g. "red ball"
                # Count each object once per approach, not once per step spent facing it.
                if obj and (not r["faced"] or r["faced"][-1] != name):
                    r["faced"].append(name)
                    # Facing the target ends the episode with a reward; any other object
                    # faced along the way counts as wrong.
                    if not (terminated and reward > 0):
                        r["wrong"].append(relation(obj, envs[i].unwrapped.instrs.desc))
                if terminated or truncated:
                    r["success"] = bool(reward > 0)
                else:
                    still.append(i)
            running = still
        records.extend(recs)
    for env in envs:
        env.close()
    for r in records:
        r["length_ratio"] = r["steps"] / r["bot_steps"]  # 1.0 = as fast as the bot
        # "Direct" = succeeded without facing any other object first.
        r["direct"] = r["success"] and not r["wrong"]
    return records


class ReplayPolicy:
    """The bot's recorded demo actions, as a reference: what 'direct' looks like for an expert
    that also turns past objects on its way. Episodes must be run in the split's seed order."""

    def __init__(self, data):
        self.data, self.next_episode = data, 0

    def reset(self, missions):
        # run() resets one batch of consecutive seeds at a time, so this batch holds the
        # next len(missions) episodes of the split.
        self.episodes = list(range(self.next_episode, self.next_episode + len(missions)))
        self.next_episode += len(missions)
        self.t = [0] * len(missions)  # step counter per episode

    def act(self, images, idx):
        # The images are ignored: just play back the bot's recorded actions in order.
        out = []
        for i in idx:
            # episode(...) returns (frames, actions, mission); [1] picks the actions.
            out.append(int(self.data.episode(self.episodes[i])[1][self.t[i]]))
            self.t[i] += 1
        return out


def rate(xs):
    # Mean of a list of True/False (or numbers); NaN ("not a number") when the list is empty.
    return float(np.mean(xs)) if len(xs) else float("nan")


def summarize(records):
    # Split the episodes into the groups the statistics below compare.
    succ = [r for r in records if r["success"]]
    fail = [r for r in records if not r["success"]]
    vis = [r for r in records if r["target_in_view_at_start"]]
    hidden = [r for r in records if not r["target_in_view_at_start"]]
    wrong = [w for r in records for w in r["wrong"]]  # every wrong object faced, across all episodes
    ratios = [r["length_ratio"] for r in succ]
    # defaultdict(list) creates an empty list the first time a key is used.
    by_target = defaultdict(list)
    for r in records:
        by_target["%s %s" % tuple(r["target"])].append(r["success"])
    return {
        "episodes": len(records),
        "success_rate": rate([r["success"] for r in records]),
        # p90 = 90th percentile: 90% of successes took at most this many times the bot's steps.
        "length_ratio_successes": {"mean": rate(ratios), "median": float(np.median(ratios)) if ratios else None,
                                   "p90": float(np.percentile(ratios, 90)) if ratios else None},
        "target_in_view_at_start": rate([r["target_in_view_at_start"] for r in records]),
        "success_rate_target_in_view": rate([r["success"] for r in vis]),
        "success_rate_target_hidden": rate([r["success"] for r in hidden]),
        # Of the successes: reached the target without facing any wrong object first.
        "direct_successes": rate([r["direct"] for r in succ]),
        "direct_successes_target_in_view": rate([r["direct"] for r in succ if r["target_in_view_at_start"]]),
        "wrong_objects_per_success": rate([len(r["wrong"]) for r in succ]),
        "wrong_objects_per_failure": rate([len(r["wrong"]) for r in fail]),
        # Failures that never faced any object: e.g. spinning in place or walking into walls.
        "failures_never_faced_an_object": rate([not r["faced"] for r in fail]),
        # Share of wrong objects that had the target's color, its type, or neither.
        "wrong_object_relation": {k: wrong.count(k) / len(wrong) for k in ("same_color", "same_type", "neither")}
        if wrong else {},
        "success_rate_by_target": {k: {"episodes": len(v), "success_rate": rate(v)} for k, v in sorted(by_target.items())},
    }


def print_summary(s):
    # Formats 0.534 as "53%". x == x is False only for NaN, which prints as "n/a".
    pct = lambda x: "%.0f%%" % (100 * x) if x == x else "n/a"  # noqa: E731
    print("  success %s over %d episodes; on successes length/bot mean %.2f, median %.2f, p90 %.2f"
          % (pct(s["success_rate"]), s["episodes"], s["length_ratio_successes"]["mean"],
             s["length_ratio_successes"]["median"] or float("nan"), s["length_ratio_successes"]["p90"] or float("nan")))
    print("  target in view at start: %s of episodes; success %s when in view, %s when hidden"
          % (pct(s["target_in_view_at_start"]), pct(s["success_rate_target_in_view"]), pct(s["success_rate_target_hidden"])))
    print("  direct successes (no wrong object faced first): %s overall, %s when the target was in view"
          % (pct(s["direct_successes"]), pct(s["direct_successes_target_in_view"])))
    print("  wrong objects faced: %.2f per success, %.2f per failure; failures that never faced an object: %s"
          % (s["wrong_objects_per_success"], s["wrong_objects_per_failure"], pct(s["failures_never_faced_an_object"])))
    if s["wrong_object_relation"]:
        print("  wrong objects share the instruction's: " + ", ".join(
            "%s %s" % (k.replace("_", " "), pct(v)) for k, v in s["wrong_object_relation"].items()))
    print("  success by target: " + ", ".join(
        "%s %s (%d)" % (k, pct(v["success_rate"]), v["episodes"]) for k, v in s["success_rate_by_target"].items()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["baseline", "vla", "bot"], required=True,
                   help="bot replays the recorded demos, as the reference")
    p.add_argument("--ckpt")
    p.add_argument("--data", default="data/GoToLocal")
    p.add_argument("--splits", default="heldout_combos,heldout_seeds")
    p.add_argument("--episodes", type=int, default=500, help="per split")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", help="default: runs/diagnose/<policy>_<level>")
    args = p.parse_args()

    meta = load_meta(args.data)
    out = args.out or os.path.join("runs", "diagnose", "%s_%s" % (args.policy, meta["level"]))
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = None if args.policy == "bot" else load_policy(args.policy, args.ckpt, device)
    for split in args.splits.split(","):
        data = load_split(args.data, EVAL_SPLITS[split])
        if args.policy == "bot":
            policy = ReplayPolicy(data)  # replays this split's demos, so rebuilt per split
        records = run(policy, meta["level"], meta["tile_size"], data.seeds[:args.episodes].tolist(),
                      data.bot_lengths(), args.batch_size)
        summary = summarize(records)
        print(split)
        print_summary(summary)
        # JSON Lines: one episode record per line, easy to load with pandas.read_json(lines=True).
        with open(os.path.join(out, split + "_episodes.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        with open(os.path.join(out, split + "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
