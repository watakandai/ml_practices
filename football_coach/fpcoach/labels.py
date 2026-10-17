"""Infer GRF action labels from tracking data.

Movement comes from each player's velocity (nearest of 8 directions, sprint above a speed
threshold). Passes and shots come from the ball: when the ball leaves one carrier and reaches
another, the first carrier's release frame gets a pass or shot label. Labels follow GRF's
"sticky" semantics so that replaying them in GRF reproduces the movement:
- a direction action keeps the player running that way until another direction or
  release_direction is issued;
- sprint stays on until release_sprint.
GRF's dribble mode has no clear signature in tracking data and is not labelled.
"""
from dataclasses import dataclass

import numpy as np

from .grf import A, DIRECTION_ACTIONS, NUM_ACTIONS
from .match import HOME

NO_ACTION = -1  # label for frames where the player is off the pitch


@dataclass
class LabelConfig:
    idle_speed: float = 1.0          # m/s; below this a player is standing still
    sprint_on: float = 7.0           # m/s; about 25 km/h, a common sprint threshold
    sprint_off: float = 5.5          # m/s; hysteresis so sprint does not flicker on and off
    carry_radius: float = 1.5        # m; the ball is "carried" by the nearest player within this
    carry_max_height: float = 1.5    # m; a ball higher than this is in flight
    min_pass_distance: float = 3.0   # m; shorter carrier changes are close control, not passes
    long_pass_distance: float = 30.0  # m; GRF's long pass vs short pass
    high_pass_height: float = 2.0    # m; peak ball height that makes a pass "high"
    shot_range: float = 35.0         # m from goal; further releases are never shots
    shot_margin: float = 2.0         # m outside each post that still counts as aimed at goal
    goal_width: float = 7.32


def movement_directions(vel, scale, cfg=LabelConfig()):
    """Nearest of 8 directions for each player and frame: 0 = idle, 1..8 index DIRECTION_ACTIONS
    (1 = right, 3 = bottom, 7 = top). Also returns sprint [T, P] bool. vel: [T, P, 2] GRF/step."""
    v_m = scale.to_m(vel)  # measure angles in metres; GRF units squash the y axis
    speed = scale.speed_mps(vel)
    angle = np.arctan2(v_m[..., 1], v_m[..., 0])
    # Round the angle to the nearest multiple of 45 degrees; index 0 is "right".
    sector = np.round(np.nan_to_num(angle) / (np.pi / 4)).astype(int) % 8
    direction = np.where(speed >= cfg.idle_speed, sector + 1, 0)  # NaN speed compares False -> idle

    # Sprint with hysteresis: turns on above sprint_on, off below sprint_off.
    sprint = np.zeros(speed.shape, dtype=bool)
    state = np.zeros(speed.shape[1:], dtype=bool)
    for i in range(len(speed)):
        s = np.nan_to_num(speed[i])
        state = np.where(state, s >= cfg.sprint_off, s >= cfg.sprint_on)
        sprint[i] = state
    return direction, sprint


def ball_carriers(match, cfg=LabelConfig()):
    """[T] slot of the player carrying the ball, or -1 when it is in flight, loose or dead."""
    scale = match.scale
    d = np.linalg.norm(scale.to_m(match.pos - match.ball[:, None, :2]), axis=-1)  # [T, P] metres
    d = np.where(np.isnan(d), np.inf, d)
    nearest = d.argmin(axis=1)
    near = d[np.arange(len(d)), nearest] <= cfg.carry_radius
    low = np.nan_to_num(match.ball[:, 2] * scale.mx, nan=np.inf) <= cfg.carry_max_height
    return np.where(near & low, nearest, -1)


def _is_shot(match, ta, attack_sign, cfg):
    """Does the ball released at frame ta head for the goal the kicker attacks?
    attack_sign is +1 when that goal is at +x, -1 when at -x."""
    scale = match.scale
    half_len = match.pitch_length / 2
    start = scale.to_m(match.ball[ta, :2])
    # Initial ball direction from the next two frames (a shot's path is straight at first).
    later = min(ta + 2, match.num_frames - 1)
    if later == ta or match.period[later] != match.period[ta]:
        return False
    v = scale.to_m(match.ball[later, :2]) - start
    if np.any(np.isnan(v)) or v[0] * attack_sign <= 0:
        return False
    to_line = attack_sign * half_len - start[0]
    if abs(to_line) > cfg.shot_range:
        return False
    y_at_line = start[1] + v[1] * to_line / v[0]  # extrapolate the ball's path to the goal line
    return abs(y_at_line) <= cfg.goal_width / 2 + cfg.shot_margin


def ball_actions(match, carrier=None, cfg=LabelConfig()):
    """Passes and shots as a list of (frame, slot, GRF action) at the kicker's release frame.

    A release is the kicker's last carrying frame before a different player carries the ball,
    or before the period ends. Releases towards goal are shots. Others are passes when the ball
    travels at least min_pass_distance, including passes the opponents intercept; short
    changes to an opponent are tackles and get no label.
    """
    if carrier is None:
        carrier = ball_carriers(match, cfg)
    scale = match.scale
    out = []
    carried = np.flatnonzero(carrier >= 0)
    for i, ta in enumerate(carried):
        kicker = carrier[ta]
        nxt = carried[i + 1] if i + 1 < len(carried) else None
        if nxt is not None and carrier[nxt] == kicker and match.period[nxt] == match.period[ta]:
            continue  # the same player still has it; ta is not a release
        # +1 if the kicker's team attacks +x in this Match's static orientation (home attacks +x).
        attack_sign = 1 if match.team[kicker] == HOME else -1
        if _is_shot(match, ta, attack_sign, cfg):
            out.append((int(ta), int(kicker), A["shot"]))
            continue
        if nxt is None or match.period[nxt] != match.period[ta]:
            continue  # the ball went dead or the period ended without a receiver
        dist = np.linalg.norm(scale.to_m(match.ball[nxt, :2] - match.ball[ta, :2]))
        if dist < cfg.min_pass_distance:
            continue
        receiver = carrier[nxt]
        if match.team[receiver] != match.team[kicker] and dist < 2 * cfg.min_pass_distance:
            continue  # lost in a duel rather than passed
        peak = np.nanmax(match.ball[ta:nxt + 1, 2]) * scale.mx
        if peak >= cfg.high_pass_height:
            action = A["high_pass"]
        elif dist >= cfg.long_pass_distance:
            action = A["long_pass"]
        else:
            action = A["short_pass"]
        out.append((int(ta), int(kicker), action))
    return out


def grf_actions(direction, sprint, on_pitch, events=()):
    """One GRF action per player and frame, [T, P] int8, NO_ACTION when off the pitch.

    Priority on each frame: a pass or shot; else a sprint toggle; else a change of direction
    (release_direction when stopping); else the current direction again (idle when standing).
    GRF takes one action per step, so a toggle delays a change of direction by one step.
    """
    T, P = direction.shape
    out = np.full((T, P), NO_ACTION, dtype=np.int8)
    ev = {(t, p): a for t, p, a in events}
    cur_dir = np.zeros(P, dtype=int)
    cur_sprint = np.zeros(P, dtype=bool)
    dir_actions = np.array([A["idle"]] + DIRECTION_ACTIONS)  # index with 0..8
    for t in range(T):
        for p in range(P):
            if not on_pitch[t, p]:
                cur_dir[p], cur_sprint[p] = 0, False
                continue
            if (t, p) in ev:
                out[t, p] = ev[(t, p)]
            elif sprint[t, p] != cur_sprint[p]:
                out[t, p] = A["sprint"] if sprint[t, p] else A["release_sprint"]
                cur_sprint[p] = sprint[t, p]
            elif direction[t, p] != cur_dir[p]:
                out[t, p] = A["release_direction"] if direction[t, p] == 0 else dir_actions[direction[t, p]]
                cur_dir[p] = direction[t, p]
            else:
                out[t, p] = dir_actions[cur_dir[p]]
    assert out.max() < NUM_ACTIONS
    return out


def label_match(match, vel, cfg=LabelConfig()):
    """All labels for a match: dict with direction, sprint, carrier, events and actions."""
    direction, sprint = movement_directions(vel, match.scale, cfg)
    carrier = ball_carriers(match, cfg)
    events = ball_actions(match, carrier, cfg)
    on_pitch = ~np.isnan(match.pos[..., 0])
    actions = grf_actions(direction, sprint, on_pitch, events)
    return {"direction": direction, "sprint": sprint, "carrier": carrier, "events": events, "actions": actions}
