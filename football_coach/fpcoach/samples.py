"""Turn a Match into training samples: one per build-up frame, in the attacking frame.

Each sample holds 11 slots per team (goalkeeper first, padded with hidden slots after a red
card), the per-player state s = (x, y, theta, vx, vy), the ball (x, y, z, vx, vy), GRF action
labels, the position 2 s ahead (the target head) and the possession's outcome (the value head).
Masking and minimap rendering happen at training time, so every epoch sees new masks.
"""
import numpy as np

from .buildup import build_ups
from .grf import A, GRF_STEP_SECONDS, ROTATE_180
from .labels import NO_ACTION, label_match
from .match import HOME, attacking_frame, headings, rotate_angle, velocities

TEAM_SIZE = 11
TARGET_SECONDS = 2.0


def _slots(match, t, team):
    """Up to 11 on-pitch players of `team` at frame t, goalkeeper first, padded with -1."""
    on = np.flatnonzero((match.team == team) & ~np.isnan(match.pos[t, :, 0]))
    on = sorted(on, key=lambda p: (not match.is_gk[p], p))[:TEAM_SIZE]
    return np.array(list(on) + [-1] * (TEAM_SIZE - len(on)))


def _take(arr, idx, fill=np.nan):
    """arr[idx] along axis 0 with idx == -1 giving `fill`."""
    out = np.full((len(idx),) + arr.shape[1:], fill, dtype=np.result_type(arr, type(fill)))
    out[idx >= 0] = arr[idx[idx >= 0]]
    return out


def _rotate_actions(actions, team_attacks_right):
    """Labels are inferred in the Match's static orientation; rotate them with the pitch."""
    actions = actions.astype(np.int8)
    if team_attacks_right:
        return actions
    return np.where(actions >= 0, ROTATE_180[np.maximum(actions, 0)], actions).astype(np.int8)


def match_samples(match, labels=None):
    """Dict of stacked arrays, one row per build-up frame. Empty arrays if there are none."""
    vel = velocities(match.pos, match.t, match.period)
    ball_vel = velocities(match.ball[:, :2], match.t, match.period)
    theta = headings(vel, match.scale)
    if labels is None:
        labels = label_match(match, vel)
    shots = [e for e in labels["events"] if e[2] == A["shot"]]
    horizon = int(round(TARGET_SECONDS / GRF_STEP_SECONDS))

    rows = []
    for bu in build_ups(match, shots):
        team = bu.possession.team
        right = team == HOME
        for t in range(bu.possession.start, bu.stop):
            own, opp = _slots(match, t, team), _slots(match, t, 1 - team)
            states = []
            for slots in (own, opp):
                xy = attacking_frame(_take(match.pos[t], slots), right)
                v = attacking_frame(_take(vel[t], slots), right)
                # Unknown heading (not yet moved): face the opponents' goal, in either orientation.
                th = np.nan_to_num(rotate_angle(_take(theta[t], slots), right), nan=0.0)
                states.append(np.concatenate([xy, th[:, None], v], axis=1))
            ball = np.concatenate([attacking_frame(match.ball[t, :2], right), match.ball[t, 2:3],
                                   attacking_frame(ball_vel[t], right)])
            # Where each own player is `horizon` steps later, if still in the same period.
            t2 = t + horizon
            if t2 < match.num_frames and match.period[t2] == match.period[t]:
                target = attacking_frame(_take(match.pos[t2], own), right)
            else:
                target = np.full((TEAM_SIZE, 2), np.nan)
            carrier = labels["carrier"][t]
            rows.append({
                "own": states[0], "opp": states[1], "ball": ball,
                "own_visible": own >= 0, "opp_visible": opp >= 0,
                "own_gk": _take(match.is_gk, own, False), "opp_gk": _take(match.is_gk, opp, False),
                "has_ball": own == carrier if carrier >= 0 else np.zeros(TEAM_SIZE, dtype=bool),
                "action": _rotate_actions(_take(labels["actions"][t], own, NO_ACTION), right),
                "target": target,
                "reached_final_third": bu.reached_final_third, "shot": bu.shot,
                "frame": t, "own_slots": own,
            })
    if not rows:
        return {}
    return {k: np.stack([np.asarray(r[k]) for r in rows]) for k in rows[0]}
