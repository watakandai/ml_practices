"""Possessions, build-up phases, and the match-level train/test split."""
import hashlib
from dataclasses import dataclass

import numpy as np

from .grf import GRF_STEP_SECONDS
from .match import HOME, NO_TEAM

OWN_THIRD_X = -1.0 / 3  # in the attacking frame, x < -1/3 is the team's own third
FINAL_THIRD_X = 1.0 / 3


@dataclass
class Possession:
    start: int    # first frame (inclusive)
    stop: int     # last frame + 1
    team: int     # HOME or AWAY

    def __len__(self):
        return self.stop - self.start


@dataclass
class BuildUp:
    possession: Possession
    stop: int                  # end of the build-up phase: ball crosses halfway or possession ends
    reached_final_third: bool  # the outcome the value model V(state) predicts
    shot: bool


def possessions(possession, period, max_gap=5):
    """Runs of one team's possession. Dead-ball gaps of up to `max_gap` frames inside a run
    are bridged (tracking providers flag short stoppages); a change of period always splits."""
    runs = []
    start, team, last = None, NO_TEAM, None
    for i, (p, per) in enumerate(zip(possession, period)):
        if p == NO_TEAM:
            continue
        if team == p and per == period[last] and i - last - 1 <= max_gap:
            last = i
            continue
        if team != NO_TEAM:
            runs.append(Possession(start, last + 1, team))
        start, team, last = i, p, i
    if team != NO_TEAM:
        runs.append(Possession(start, last + 1, team))
    return runs


def build_ups(match, shots=(), min_seconds=2.0):
    """Possessions that start in the team's own third, cut where the ball first crosses halfway.

    Outcome labels look at the whole possession: did the ball reach the final third, and did
    the team shoot? `shots` is a list of (frame, slot, action) shot events from labels.
    """
    shot_frames = {(t, match.team[p]) for t, p, _ in shots}
    out = []
    min_len = int(round(min_seconds / GRF_STEP_SECONDS))
    for pos in possessions(match.possession, match.period):
        sign = 1 if pos.team == HOME else -1  # ball x in the team's attacking frame is sign * x
        bx = sign * match.ball[pos.start:pos.stop, 0]
        if len(pos) < min_len or not bx[0] < OWN_THIRD_X:  # NaN start also fails
            continue
        crossed = np.flatnonzero(bx >= 0)
        stop = pos.start + crossed[0] if len(crossed) else pos.stop
        shot = any((t, pos.team) in shot_frames for t in range(pos.start, pos.stop))
        out.append(BuildUp(pos, stop, bool(np.nanmax(bx) >= FINAL_THIRD_X), shot))
    return out


def match_split(match_id, test_fraction=0.2, salt="fpcoach"):
    """'train' or 'test' for a whole match, from a hash of its id, so frames of one match
    never land on both sides and the split does not depend on which files were loaded."""
    h = int(hashlib.sha256(("%s:%s" % (salt, match_id)).encode()).hexdigest()[:8], 16)
    return "test" if h / 0xFFFFFFFF < test_fraction else "train"
