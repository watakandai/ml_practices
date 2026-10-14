"""Render policy rollouts as GIFs: the top-down level (the agent's view is highlighted) next to
the egocentric frame the policy actually sees, with the mission, step and chosen action.

  python -m toyvla.render --policy vla --ckpt runs/vla256/GoToLocal/final --split heldout_combos
  python -m toyvla.render --policy baseline --ckpt runs/baseline/GoToLocal/model.pt --select fail

Writes <out>/<split>_seed<seed>_<success|fail>.gif.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from .common import ACTION_NAMES, make_env, quiet_reset
from .data import load_meta, load_split
from .evaluate import EVAL_SPLITS, load_policy

HEADER, FOOTER, GAP, EGO_SCALE = 34, 22, 8, 3
GREEN, RED, WHITE, GREY = (80, 200, 120), (230, 80, 80), (235, 235, 235), (150, 150, 150)


def compose(grid, ego, mission, caption, banner=None):
    """One GIF frame: header with the mission, top-down grid | upscaled egocentric view, footer."""
    ego = np.asarray(Image.fromarray(ego).resize((ego.shape[1] * EGO_SCALE, ego.shape[0] * EGO_SCALE), Image.NEAREST))
    w = grid.shape[1] + GAP + ego.shape[1]
    h = HEADER + max(grid.shape[0], ego.shape[0]) + FOOTER
    img = Image.new("RGB", (w, h), (24, 24, 24))
    img.paste(Image.fromarray(grid), (0, HEADER))
    img.paste(Image.fromarray(ego), (grid.shape[1] + GAP, HEADER))
    d = ImageDraw.Draw(img)
    d.text((6, 4), mission, fill=WHITE)
    d.text((grid.shape[1] + GAP, HEADER - 14), "agent's view", fill=GREY)
    d.text((6, h - FOOTER + 5), caption, fill=WHITE)
    if banner:
        text, color = banner
        d.text((6, 18), text, fill=color)
    return img


@torch.no_grad()
def rollout(policy, env, seed, bot_len):
    obs, _ = quiet_reset(env, seed)
    mission = obs["mission"]
    policy.reset([mission])
    frames, t = [], 0
    while True:
        grid = env.render()
        action = int(policy.act(obs["image"][None], [0])[0])
        frames.append(compose(grid, obs["image"], mission, "step %d   action: %s" % (t, ACTION_NAMES[action])))
        obs, reward, terminated, truncated, _ = env.step(action)
        t += 1
        if terminated or truncated:
            break
    success = reward > 0
    result = ("SUCCESS in %d steps (bot: %d)" % (t, bot_len) if success
              else "FAILED after %d steps (bot: %d)" % (t, bot_len), GREEN if success else RED)
    frames.append(compose(env.render(), obs["image"], mission, "step %d   done" % t, banner=result))
    return frames, success


def save_gif(frames, path, ms_per_step=300, hold_last=8):
    frames = frames + [frames[-1]] * hold_last
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms_per_step, loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["baseline", "vla"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/GoToLocal")
    p.add_argument("--split", choices=list(EVAL_SPLITS), default="heldout_combos")
    p.add_argument("--episodes", type=int, default=4, help="GIFs to write")
    p.add_argument("--select", choices=["any", "success", "fail"], default="any",
                   help="keep only successful or failed episodes (tries up to 50 seeds)")
    p.add_argument("--out", help="default: media/<policy>_<level>")
    p.add_argument("--ms-per-step", type=int, default=300)
    args = p.parse_args()

    meta = load_meta(args.data)
    out = args.out or os.path.join("media", "%s_%s" % (args.policy, meta["level"]))
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy(args.policy, args.ckpt, device)
    data = load_split(args.data, EVAL_SPLITS[args.split])
    bot_lengths = data.bot_lengths()
    env = make_env(meta["level"], meta["tile_size"], render_mode="rgb_array")

    written = 0
    for seed in data.seeds[:50].tolist():
        frames, success = rollout(policy, env, seed, bot_lengths[seed])
        if args.select == "success" and not success or args.select == "fail" and success:
            continue
        path = os.path.join(out, "%s_seed%d_%s.gif" % (args.split, seed, "success" if success else "fail"))
        save_gif(frames, path, args.ms_per_step)
        print("wrote %s (%d steps)" % (path, len(frames) - 1))
        written += 1
        if written == args.episodes:
            break


if __name__ == "__main__":
    main()
