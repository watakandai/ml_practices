"""A small hand-scripted build-up with known labels, for tests and for trying the pipeline
without real data.

Home (attacking +x) keeps the ball for 10 s at 10 Hz:
- frames 0-19: the goalkeeper stands on the ball, then plays a 14 m pass to a centre-back;
- frames 30-59: the centre-back carries the ball forward at 4 m/s;
- frame 59: a 32 m pass to the centre-forward, who receives it at frame 70;
- frames 70-99: the centre-forward sprints forward with the ball at 8 m/s.
One away midfielder jogs towards the top touchline at 3 m/s; everyone else stands still.
"""
import numpy as np

from .grf import PitchScale
from .match import AWAY, HOME, Match

HZ = 10
T = 100
GK, CB, CF = 0, 1, 9          # home slots with a scripted role
AWAY_JOGGER = 11 + 5          # an away slot

# A 4-4-2 in metres (GRF orientation: +x attacks right, +y towards the bottom touchline).
_FORMATION = np.array([
    [-47, 0], [-35, -8], [-35, 8], [-32, -25], [-32, 25],
    [-10, -20], [-12, -6], [-12, 6], [-10, 20], [8, -4], [5, 6],
], dtype=float)


def _line(a, b, n):
    """n points from a to b inclusive."""
    return np.linspace(a, b, n)


def scripted_match(match_id="synthetic", length=105.0, width=68.0):
    scale = PitchScale(length, width)
    pos = np.zeros((T, 22, 2))
    pos[:, :11] = _FORMATION
    pos[:, 11:] = -_FORMATION  # the away team is the home team rotated by 180 degrees
    ball = np.zeros((T, 3))

    # Goalkeeper on the ball, then a 14 m pass to the centre-back (frames 20-26 in flight).
    gk, cb0 = _FORMATION[GK], _FORMATION[CB]
    ball[:20, :2] = gk
    ball[19:28, :2] = _line(gk, cb0, 9)
    ball[28:30, :2] = cb0
    # Centre-back carries forward at 4 m/s from frame 30 to 59.
    cb_path = _line(cb0, cb0 + [4.0 * 29 / HZ, 0], 30)
    pos[30:60, CB] = cb_path
    pos[60:, CB] = cb_path[-1]
    ball[30:60, :2] = cb_path
    # 32 m pass to the centre-forward, lofted to 1.9 m (in flight, but below a "high" pass),
    # frames 60-69 in flight.
    cf_recv = _FORMATION[CF]
    ball[59:71, :2] = _line(cb_path[-1], cf_recv, 12)
    ball[59:71, 2] = 1.9 * np.sin(np.linspace(0, np.pi, 12))
    # Centre-forward sprints forward with the ball at 8 m/s from frame 70.
    cf_path = _line(cf_recv, cf_recv + [8.0 * 29 / HZ, 0], 30)
    pos[70:, CF] = cf_path
    ball[70:, :2] = cf_path
    # One away midfielder jogs towards the top touchline (-y) at 3 m/s.
    pos[:, AWAY_JOGGER] = _line(pos[0, AWAY_JOGGER], pos[0, AWAY_JOGGER] + [0, -3.0 * (T - 1) / HZ], T)

    ball_grf = np.concatenate([scale.to_grf(ball[:, :2]), ball[:, 2:3] / scale.mx], axis=1)
    is_gk = np.zeros(22, dtype=bool)
    is_gk[[0, 11]] = True
    return Match(
        match_id=match_id, t=np.arange(T) / HZ, period=np.ones(T, dtype=int),
        pos=scale.to_grf(pos), ball=ball_grf, possession=np.full(T, HOME, dtype=np.int8),
        team=np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8), is_gk=is_gk,
        player_ids=np.array(["p%d" % i for i in range(22)]), pitch_length=length, pitch_width=width)


def mirrored(match):
    """The same match with the teams' roles swapped: away has the ball and attacks -x.
    Samples from it should equal those from the original, which tests the attacking frame."""
    swap = np.r_[11:22, 0:11]
    return Match(
        match_id=match.match_id + "_mirrored", t=match.t, period=match.period,
        pos=-match.pos[:, swap], ball=np.concatenate([-match.ball[:, :2], match.ball[:, 2:]], axis=1),
        possession=np.where(match.possession == HOME, AWAY, HOME).astype(np.int8),
        team=match.team[swap] ^ 1, is_gk=match.is_gk[swap], player_ids=match.player_ids[swap],
        pitch_length=match.pitch_length, pitch_width=match.pitch_width)
