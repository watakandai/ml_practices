"""Load PFF FC tracking data (2022 World Cup release) into a Match, through kloppy.

PFF ships, per game: `metadata/<game_id>.json`, `rosters/<game_id>.json` and
`tracking/<game_id>.jsonl.bz2`. kloppy reads them, resamples, and puts the data in a fixed
orientation; this module copies frames into Match arrays and converts to GRF units.
Other providers kloppy supports (Metrica, Sportec/IDSSE) need only a different `load_*` call,
since everything after `_from_kloppy` is provider-independent.
"""
import os

import numpy as np

from .grf import GRF_STEP_SECONDS, PitchScale
from .match import AWAY, HOME, NO_TEAM, Match


def pff_paths(root, game_id):
    return (os.path.join(root, "metadata", "%s.json" % game_id),
            os.path.join(root, "rosters", "%s.json" % game_id),
            os.path.join(root, "tracking", "%s.jsonl.bz2" % game_id))


def load_pff(root, game_id, **kw):
    """Match for one PFF game in the release's directory layout."""
    return load_pff_files(*pff_paths(root, game_id), match_id=str(game_id), **kw)


def load_pff_files(meta, roster, raw, match_id, source_hz=29.97, limit=None, periods=(1, 2)):
    """Match from explicit PFF files, resampled from `source_hz` to GRF's 10 Hz.

    Extra time is left out by default: it is a small share of the data, and in kloppy's PFF
    sample files its coordinates fall outside the pitch, so it needs checking before use."""
    from kloppy import pff

    # kloppy keeps every (1 / sample_rate)-th frame: 1/3 turns 29.97 Hz into about 10 Hz.
    sample_rate = 1 / round(source_hz * GRF_STEP_SECONDS)
    ds = pff.load_tracking(meta, roster, raw, sample_rate=sample_rate, limit=limit, only_alive=False)
    return _from_kloppy(ds, match_id, periods)


def _from_kloppy(ds, match_id, periods=None):
    """kloppy TrackingDataset -> Match. Uses Second Spectrum's coordinates (metres, origin at the
    centre spot, y up) with the home team attacking +x in both periods."""
    ds = ds.transform(to_coordinate_system="secondspectrum", to_orientation="STATIC_HOME_AWAY")
    length = ds.metadata.coordinate_system.pitch_length
    width = ds.metadata.coordinate_system.pitch_width
    scale = PitchScale(length, width)
    home, away = ds.metadata.teams  # kloppy lists the home team first

    players = list(home.players) + list(away.players)
    slot = {p.player_id: i for i, p in enumerate(players)}
    team = np.array([HOME] * len(home.players) + [AWAY] * len(away.players), dtype=np.int8)
    is_gk = np.array([_is_goalkeeper(p) for p in players])

    frames = [f for f in ds.frames if periods is None or f.period.id in periods]
    T, P = len(frames), len(players)
    pos = np.full((T, P, 2), np.nan)
    ball = np.full((T, 3), np.nan)
    possession = np.full(T, NO_TEAM, dtype=np.int8)
    t = np.zeros(T)
    period = np.zeros(T, dtype=int)
    for i, f in enumerate(frames):
        t[i] = f.timestamp.total_seconds() + 10_000 * (f.period.id - 1)  # keep periods apart in time
        period[i] = f.period.id
        for p, data in f.players_data.items():
            if data.coordinates is not None:
                pos[i, slot[p.player_id]] = (data.coordinates.x, data.coordinates.y)
        if f.ball_coordinates is not None:
            b = f.ball_coordinates
            ball[i] = (b.x, b.y, b.z if b.z is not None else 0.0)
        if f.ball_owning_team is not None:
            possession[i] = HOME if f.ball_owning_team == home else AWAY

    # Second Spectrum's y points up; GRF's points down. Flip y, then scale each axis.
    pos[..., 1] *= -1
    ball[:, 1] *= -1
    pos = scale.to_grf(pos)
    ball = np.concatenate([scale.to_grf(ball[:, :2]), ball[:, 2:3] / scale.mx], axis=1)
    if not is_gk[team == HOME].any() or not is_gk[team == AWAY].any():
        is_gk = _guess_goalkeepers(pos, team, is_gk)
    return Match(match_id=match_id, t=t, period=period, pos=pos, ball=ball, possession=possession,
                 team=team, is_gk=is_gk, player_ids=np.array([p.player_id for p in players]),
                 pitch_length=length, pitch_width=width)


def _is_goalkeeper(player):
    pos = getattr(player, "starting_position", None)
    return pos is not None and "goalkeeper" in str(pos).lower()


def _guess_goalkeepers(pos, team, is_gk):
    """Fallback when the roster has no positions: per team, the player whose mean x is deepest
    (home defends -x, away defends +x)."""
    is_gk = is_gk.copy()
    mean_x = np.nanmean(pos[..., 0], axis=0)
    for side, sign in ((HOME, 1), (AWAY, -1)):
        if is_gk[team == side].any():
            continue
        idx = np.flatnonzero((team == side) & ~np.isnan(mean_x))
        is_gk[idx[np.argmin(sign * mean_x[idx])]] = True
    return is_gk
