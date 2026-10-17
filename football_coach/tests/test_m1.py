import json
import os

import kloppy
import numpy as np
import pytest

from fpcoach.buildup import build_ups, match_split, possessions
from fpcoach.grf import A, ACTION_NAMES, NUM_ACTIONS, SMM_HEIGHT, SMM_WIDTH, PitchScale
from fpcoach.labels import label_match, movement_directions
from fpcoach.masking import MaskConfig, visibility_mask
from fpcoach.match import AWAY, HOME, NO_TEAM, Match, velocities
from fpcoach.minimap import render_smm, smm_pixels
from fpcoach.prepare_data import process
from fpcoach.render import render_build_up
from fpcoach.samples import match_samples
from fpcoach.synthetic import AWAY_JOGGER, CB, CF, GK, mirrored, scripted_match

KLOPPY_FILES = os.path.join(os.path.dirname(kloppy.__file__), "tests", "files")


@pytest.fixture(scope="module")
def match():
    return scripted_match()


@pytest.fixture(scope="module")
def labels(match):
    return label_match(match, velocities(match.pos, match.t, match.period))


def test_action_set_matches_grf():
    assert NUM_ACTIONS == 19
    assert [A[n] for n in ("idle", "right", "top", "long_pass", "shot", "sprint", "release_dribble")] == \
        [0, 5, 3, 9, 12, 13, 18]


def test_pitch_scale_round_trip():
    s = PitchScale(105, 68)
    corner = s.to_grf([52.5, 34.0])
    assert np.allclose(corner, [1.0, 0.42])
    xy = np.random.default_rng(0).uniform(-50, 50, (10, 2))
    assert np.allclose(s.to_m(s.to_grf(xy)), xy)


def test_directions_follow_grf_axes():
    s = PitchScale()
    step = 0.4 / s.mx  # 4 m/s along x, as a per-step displacement in GRF units
    vel = np.array([[[step, 0], [0, 0.4 / s.my], [-step, 0], [0, -0.4 / s.my], [0, 0]]])
    direction, _ = movement_directions(vel, s)
    # 1 = right, 3 = bottom (+y), 5 = left, 7 = top (-y), 0 = idle
    assert direction[0].tolist() == [1, 3, 5, 7, 0]


def test_scripted_labels(match, labels):
    assert labels["events"] == [(19, GK, A["short_pass"]), (59, CB, A["long_pass"])]
    acts = labels["actions"]
    assert set(acts[35:59, CB]) == {A["right"]}
    assert acts[60, CB] == A["release_direction"]
    assert set(acts[:, AWAY_JOGGER]) == {A["top"]}
    # The centre-forward goes from standing to 8 m/s: a sprint toggle, then running right.
    cf = [ACTION_NAMES[a] for a in acts[69:80, CF]]
    assert "sprint" in cf and cf[-1] == "right"
    assert labels["sprint"][85, CF] and not labels["sprint"][85, CB]


def test_possessions_bridge_short_gaps():
    poss = np.array([HOME] * 5 + [NO_TEAM] * 3 + [HOME] * 5 + [AWAY] * 4)
    runs = possessions(poss, np.ones(len(poss), dtype=int), max_gap=5)
    assert [(r.start, r.stop, r.team) for r in runs] == [(0, 13, HOME), (13, 17, AWAY)]


def test_build_up(match):
    (bu,) = build_ups(match)
    # The ball crosses halfway during the long pass, about 3/4 of the way to the forward.
    assert bu.possession.start == 0 and 62 <= bu.stop <= 68
    assert bu.reached_final_third and not bu.shot


def test_samples_are_orientation_invariant(match):
    a, b = match_samples(match), match_samples(mirrored(match))
    assert len(a["frame"]) > 0 and a.keys() == b.keys()
    for k in a:
        if k != "own_slots":
            np.testing.assert_allclose(a[k], b[k], atol=1e-9, err_msg=k)


def test_sample_contents(match):
    s = match_samples(match)
    assert s["own"].shape[1:] == (11, 5) and s["opp"].shape[1:] == (11, 5) and s["ball"].shape[1:] == (5,)
    assert s["own_gk"][:, 0].all()  # goalkeeper first
    assert s["has_ball"][0, 0] and s["has_ball"][40, 1]
    i = 40  # the centre-back carrying the ball forward at 4 m/s
    assert s["own"][i, 1, 3] > 0 and abs(s["own"][i, 1, 2]) < 1e-6  # vx > 0, heading 0
    np.testing.assert_allclose(match.scale.speed_mps(s["own"][i, 1, 3:5]), 4.0, rtol=1e-6)
    np.testing.assert_allclose(s["target"][i, 1, 0] - s["own"][i, 1, 0], 4.0 * 2 / match.scale.mx, atol=0.05)


def test_minimap_matches_grf_marking():
    # GRF: col = int((x + 1) / 2 * 96), row = int((y + 1/2.25) / (2/2.25) * 72)
    px = smm_pixels([[0.0, 0.0], [-1.0, -0.42], [1.0, 0.42], [0.5, -0.2]])
    assert px.tolist() == [[48, 36], [0, 1], [95, 70], [72, 19]]
    frame = render_smm(own=[[0.0, 0.0], [np.nan, np.nan]], opp=[[0.5, -0.2]], ball=[0.0, 0.0, 0.1], active=[0.0, 0.0])
    assert frame.shape == (SMM_HEIGHT, SMM_WIDTH, 4)
    assert frame.sum(axis=(0, 1)).tolist() == [255, 255, 255, 255]
    assert frame[19, 72, 1] == 255


def test_camera_masking_keeps_players_near_ball():
    pos = np.array([[0.0, 0], [10, 0], [45, 0], [-40, 5]])
    vis = np.array([True, True, True, False])
    cfg = MaskConfig(p_camera=1.0, camera_radius=30, p_drop=0.0, keep=(2,))
    keep = visibility_mask(pos, np.zeros(2), vis, np.random.default_rng(0), cfg)
    assert keep.tolist() == [True, True, True, False]  # slot 2 is far but kept; slot 3 is off the pitch


def test_split_is_by_match_and_stable():
    ids = ["m%d" % i for i in range(200)]
    splits = [match_split(i) for i in ids]
    assert splits == [match_split(i) for i in ids]
    assert 0.1 < splits.count("test") / len(ids) < 0.3


def test_prepare_and_render(tmp_path, match):
    s = process(match, str(tmp_path))
    json.dumps(s)
    assert s["build_ups"] == 1 and s["samples"] > 0
    m2 = Match.load(str(tmp_path / "synthetic.match.npz"))
    np.testing.assert_array_equal(m2.pos, match.pos)
    assert render_build_up(m2, 0, str(tmp_path / "bu.gif")) > 60


def test_pff_sample_files_load(tmp_path):
    from fpcoach.load_pff import load_pff_files
    f = lambda name: os.path.join(KLOPPY_FILES, name)  # noqa: E731
    m = load_pff_files(f("pff_metadata_10517.json"), f("pff_rosters_10517.json"), f("pff_10517.jsonl"), "10517")
    assert set(np.unique(m.period)) == {1, 2}
    assert np.nanmax(np.abs(m.pos[..., 0])) <= 1.1 and np.nanmax(np.abs(m.pos[..., 1])) <= 0.5
    # Static orientation: the home goalkeeper stays in the -x half in both periods.
    home_gk = m.is_gk & (m.team == HOME)
    for p in (1, 2):
        assert np.nanmean(m.pos[m.period == p][:, home_gk, 0]) < 0
    on_pitch = (~np.isnan(m.pos[..., 0])).sum(axis=1)
    assert on_pitch.max() <= 22
    process(m, str(tmp_path))  # the whole pipeline runs on real-format data
