"""Shared constants and helpers: environment construction, action names, mission parsing."""
import contextlib
import io
import re

import gymnasium as gym
import minigrid  # noqa: F401  (registers the BabyAI-* environments)
from minigrid.wrappers import RGBImgPartialObsWrapper

# MiniGrid's 7 discrete actions, in index order (minigrid.core.actions.Actions).
ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
NUM_ACTIONS = len(ACTION_NAMES)

COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
OBJECT_TYPES = ["ball", "box", "key", "door"]

# Color-object combinations whose *instructions* never appear in training data.
# Objects of these combinations can still show up as distractors in the scene.
DEFAULT_HELDOUT_COMBOS = [("red", "ball"), ("green", "key"), ("blue", "box"), ("yellow", "ball")]

_COMBO_RE = re.compile(r"\b(?:(%s) )?(%s)\b" % ("|".join(COLORS), "|".join(OBJECT_TYPES)))

# Training and test seeds come from disjoint ranges.
TRAIN_SEED_START = 0
TEST_SEED_START = 10_000_000


def make_env(level, tile_size=8):
    """BabyAI level with an egocentric RGB view of the 7x7 field of view (7 * tile_size pixels square)."""
    if not level.startswith("BabyAI-"):
        level = "BabyAI-%s" % level
    if not level.endswith("-v0"):
        level = "%s-v0" % level
    return RGBImgPartialObsWrapper(gym.make(level), tile_size=tile_size)


def quiet_reset(env, seed):
    """env.reset, with MiniGrid's 'Sampling rejected' prints suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return env.reset(seed=seed)


def mission_combos(mission):
    """(color, type) pairs named in a mission; color is None when the mission omits it."""
    return [(m.group(1), m.group(2)) for m in _COMBO_RE.finditer(mission)]


def is_heldout_mission(mission, heldout_combos=DEFAULT_HELDOUT_COMBOS):
    held = set(map(tuple, heldout_combos))
    return any(c in held for c in mission_combos(mission))


def parse_combos(spec):
    """'red-ball,green-key' -> [('red', 'ball'), ('green', 'key')]."""
    if not spec:
        return []
    return [tuple(item.split("-")) for item in spec.split(",")]
