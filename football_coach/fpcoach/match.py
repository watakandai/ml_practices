"""One match of tracking data in the project's shared format, and per-frame kinematics.

A Match stores every rostered player in a fixed slot, in GRF units, in a static orientation:
the home team attacks towards +x in both periods. Per-possession views (the team in possession
attacking +x as GRF's "left team") are made later by `attacking_frame`.
"""
from dataclasses import dataclass

import numpy as np

from .grf import GRF_STEP_SECONDS, PitchScale

HOME, AWAY = 0, 1
NO_TEAM = -1


@dataclass
class Match:
    match_id: str
    t: np.ndarray            # [T] seconds since the start of the match data, at about 10 Hz
    period: np.ndarray       # [T] int, 1 and 2 (extra time 3 and 4)
    pos: np.ndarray          # [T, P, 2] GRF units; NaN while a player is off the pitch
    ball: np.ndarray         # [T, 3] GRF units; z is metres / PitchScale.mx (like x); NaN if lost
    possession: np.ndarray   # [T] HOME, AWAY or NO_TEAM (dead ball / unknown)
    team: np.ndarray         # [P] HOME or AWAY
    is_gk: np.ndarray        # [P] bool
    player_ids: np.ndarray   # [P] str, the provider's ids
    pitch_length: float = 105.0
    pitch_width: float = 68.0

    @property
    def scale(self):
        return PitchScale(self.pitch_length, self.pitch_width)

    @property
    def num_frames(self):
        return len(self.t)

    def save(self, path):
        np.savez_compressed(path, match_id=self.match_id, t=self.t, period=self.period, pos=self.pos,
                            ball=self.ball, possession=self.possession, team=self.team, is_gk=self.is_gk,
                            player_ids=self.player_ids, pitch=np.array([self.pitch_length, self.pitch_width]))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(match_id=str(z["match_id"]), t=z["t"], period=z["period"], pos=z["pos"], ball=z["ball"],
                   possession=z["possession"], team=z["team"], is_gk=z["is_gk"], player_ids=z["player_ids"],
                   pitch_length=float(z["pitch"][0]), pitch_width=float(z["pitch"][1]))


def period_slices(period):
    """Contiguous runs of the same period, as slices. Kinematics never difference across them."""
    edges = np.flatnonzero(np.diff(period) != 0) + 1
    bounds = np.concatenate([[0], edges, [len(period)]])
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def velocities(xy, t, period):
    """GRF-style movement vectors: displacement per GRF step (0.1 s), from central differences.

    xy: [T, ..., 2]. Returns the same shape. NaN wherever a needed neighbour is missing, e.g. on
    the frames next to a player's substitution.
    """
    vel = np.full(xy.shape, np.nan)
    for s in period_slices(period):
        if s.stop - s.start < 2:
            continue
        # np.gradient uses central differences inside and one-sided ones at the run's ends.
        # Its `t` argument handles uneven spacing (dropped frames).
        vel[s] = np.gradient(xy[s], t[s], axis=0) * GRF_STEP_SECONDS
    return vel


def headings(vel, scale, min_speed_mps=0.5):
    """theta: the direction of motion in radians, in GRF's axes (0 = towards +x, +pi/2 = towards +y).

    Tracking data rarely has body orientation, so this stands in for it. Below `min_speed_mps`
    the direction is noise, so the last reliable heading is carried forward; NaN before any,
    so callers fill it after choosing an orientation. vel: [T, P, 2] -> [T, P].
    """
    v_m = scale.to_m(vel)  # angles must be measured in metres, not in GRF's anisotropic units
    theta = np.arctan2(v_m[..., 1], v_m[..., 0])
    moving = scale.speed_mps(vel) >= min_speed_mps  # False for NaN too
    out = np.zeros(theta.shape)
    last = np.full(theta.shape[1:], np.nan)
    for i in range(len(theta)):
        last = np.where(moving[i], theta[i], last)
        out[i] = last
    return out


def attacking_frame(xy, team_attacks_right):
    """Rotate positions or vectors by 180 degrees when the team of interest attacks towards -x,
    so that it attacks +x like GRF's left team. A rotation (not a mirror) keeps left and right
    footedness and the sense of the 8 movement directions intact."""
    return xy if team_attacks_right else -xy


def rotate_angle(theta, team_attacks_right):
    """The heading counterpart of `attacking_frame`: add pi and wrap back to (-pi, pi]."""
    return theta if team_attacks_right else np.angle(-np.exp(1j * theta))
