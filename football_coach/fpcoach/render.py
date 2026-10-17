"""Draw a build-up with its inferred labels, to check by eye that the labels match the play.

    python -m fpcoach.render --match data/synthetic/synthetic.match.npz --build-up 0 --out runs/bu0.gif

Own team (in possession, attacking right) in blue, opponents in red, ball in white. Each own
player's arrow is the direction label; a thicker outline means sprint; the ball action
(pass or shot) is written above the kicker on its release frame.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from .buildup import build_ups
from .grf import A, ACTION_NAMES, GRF_STEP_SECONDS, GRF_X_MAX, GRF_Y_MAX
from .labels import label_match
from .match import HOME, Match, attacking_frame, velocities

PX = 300  # pixels per GRF unit
MARGIN = 20
DIR_ANGLES = np.arange(8) * np.pi / 4  # direction 1..8 -> angle (0 = right, pi/2 = bottom)


def _to_px(xy):
    return (MARGIN + (xy[0] + GRF_X_MAX) * PX, MARGIN + (xy[1] + GRF_Y_MAX) * PX)


def _pitch():
    w, h = int(2 * GRF_X_MAX * PX + 2 * MARGIN), int(2 * GRF_Y_MAX * PX + 2 * MARGIN)
    img = Image.new("RGB", (w, h), (34, 110, 52))
    d = ImageDraw.Draw(img)
    line = (220, 235, 220)
    d.rectangle([_to_px((-1, -GRF_Y_MAX)), _to_px((1, GRF_Y_MAX))], outline=line, width=2)
    d.line([_to_px((0, -GRF_Y_MAX)), _to_px((0, GRF_Y_MAX))], fill=line, width=2)
    for x in (-1 / 3, 1 / 3):  # thirds, which define build-up and the value head's outcome
        d.line([_to_px((x, -GRF_Y_MAX)), _to_px((x, GRF_Y_MAX))], fill=(60, 140, 75), width=1)
    return img


def draw_frame(match, labels, t, team):
    right = team == HOME
    img = _pitch()
    d = ImageDraw.Draw(img)
    events = {(e[0], e[1]): e[2] for e in labels["events"]}
    for p in range(match.pos.shape[1]):
        if np.isnan(match.pos[t, p, 0]):
            continue
        cx, cy = _to_px(attacking_frame(match.pos[t, p], right))
        own = match.team[p] == team
        color = (60, 120, 255) if own else (230, 60, 60)
        width = 3 if labels["sprint"][t, p] else 1
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=color, outline=(255, 255, 255), width=width)
        k = labels["direction"][t, p]
        if own and k > 0:
            ang = DIR_ANGLES[k - 1] + (0 if right else np.pi)
            d.line([(cx, cy), (cx + 22 * np.cos(ang), cy + 22 * np.sin(ang))], fill=(255, 255, 0), width=2)
        if (t, p) in events:
            d.text((cx - 20, cy - 22), ACTION_NAMES[events[(t, p)]], fill=(255, 255, 255))
    if not np.isnan(match.ball[t, 0]):
        bx, by = _to_px(attacking_frame(match.ball[t, :2], right))
        d.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=(255, 255, 255))
    d.text((MARGIN, 2), "%s  frame %d  t=%.1fs" % (match.match_id, t, match.t[t]), fill=(255, 255, 255))
    return img


def render_build_up(match, index, out, extra_frames=20):
    vel = velocities(match.pos, match.t, match.period)
    labels = label_match(match, vel)
    shots = [e for e in labels["events"] if e[2] == A["shot"]]
    bu = build_ups(match, shots)[index]
    stop = min(bu.possession.stop, bu.stop + extra_frames)
    frames = [draw_frame(match, labels, t, bu.possession.team) for t in range(bu.possession.start, stop)]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(GRF_STEP_SECONDS * 1000), loop=0)
    return len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True, help="a <id>.match.npz from prepare_data")
    ap.add_argument("--build-up", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = render_build_up(Match.load(args.match), args.build_up, args.out)
    print("wrote %d frames to %s" % (n, args.out))


if __name__ == "__main__":
    main()
