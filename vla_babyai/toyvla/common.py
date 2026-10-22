"""Shared constants and helpers: environment construction, action names, mission parsing."""
import contextlib  # redirect_stdout, used to silence MiniGrid's prints
import io  # StringIO: an in-memory "file" that swallows those prints
import math
import re  # regular expressions, used to find "<color> <type>" phrases in missions

import gymnasium as gym  # the standard RL environment API (reset/step/render)
import minigrid  # noqa: F401  (registers the BabyAI-* environments)
# Turns MiniGrid's symbolic observation into an RGB picture of what the agent sees.
from minigrid.wrappers import RGBImgPartialObsWrapper

# MiniGrid's 7 discrete actions, in index order (minigrid.core.actions.Actions).
# Action index 0 is "left", 1 is "right", and so on; the env's step() takes these integers.
ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
NUM_ACTIONS = len(ACTION_NAMES)  # 7: the size of every policy's output

# All colors and object types BabyAI can mention in a mission.
COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
OBJECT_TYPES = ["ball", "box", "key", "door"]

# Color-object combinations whose *instructions* never appear in training data.
# Objects of these combinations can still show up as distractors in the scene.
DEFAULT_HELDOUT_COMBOS = [("red", "ball"), ("green", "key"), ("blue", "box"), ("yellow", "ball")]

# Matches an optional color followed by an object type, e.g. "red ball" or just "ball".
# The two parenthesised groups capture the color (or None) and the type.
# "|".join(COLORS) builds "red|green|..." so the regex accepts any one of them.
_COMBO_RE = re.compile(r"\b(?:(%s) )?(%s)\b" % ("|".join(COLORS), "|".join(OBJECT_TYPES)))

# Training and test seeds come from disjoint ranges.
# A seed fully determines a level's layout, so disjoint ranges guarantee unseen test layouts.
TRAIN_SEED_START = 0
# Validation levels (training combos only) for model selection, between the RL range
# (1,000,000-8,999,999, toyvla.rl) and the test range.
VAL_SEED_START = 9_000_000
TEST_SEED_START = 10_000_000


def make_env(level, tile_size=8, render_mode=None):
    """BabyAI level with an egocentric RGB view of the 7x7 field of view (7 * tile_size pixels square).
    render_mode="rgb_array" additionally lets env.render() draw the full top-down grid."""
    # Accept short names like "GoToLocal" and expand them to the registered id
    # "BabyAI-GoToLocal-v0".
    if not level.startswith("BabyAI-"):
        level = "BabyAI-%s" % level
    if not level.endswith("-v0"):
        level = "%s-v0" % level
    # gym.make builds the environment; the wrapper replaces obs["image"] with an RGB array
    # of shape (7 * tile_size, 7 * tile_size, 3), i.e. 56x56x3 for the default tile_size=8.
    return RGBImgPartialObsWrapper(gym.make(level, render_mode=render_mode), tile_size=tile_size)


def quiet_reset(env, seed):
    """env.reset, with MiniGrid's 'Sampling rejected' prints suppressed."""
    # While generating a level, MiniGrid prints a message every time it discards a layout.
    # redirect_stdout sends those prints into a throwaway StringIO instead of the terminal.
    with contextlib.redirect_stdout(io.StringIO()):
        # reset(seed=...) builds the level for that seed and returns (observation, info).
        return env.reset(seed=seed)


def mission_combos(mission):
    """(color, type) pairs named in a mission; color is None when the mission omits it."""
    # finditer yields one match per "<color> <type>" phrase; group(1) is the color
    # (None if absent) and group(2) the type.
    return [(m.group(1), m.group(2)) for m in _COMBO_RE.finditer(mission)]


def is_heldout_mission(mission, heldout_combos=DEFAULT_HELDOUT_COMBOS):
    # A set of tuples makes the "is this pair held out?" lookup fast. map(tuple, ...) also
    # turns lists loaded from JSON (["red", "ball"]) into hashable tuples.
    held = set(map(tuple, heldout_combos))
    # True if any object phrase in the mission is one of the held-out pairs.
    return any(c in held for c in mission_combos(mission))


def parse_combos(spec):
    """'red-ball,green-key' -> [('red', 'ball'), ('green', 'key')]."""
    if not spec:  # empty string or None: no held-out combos
        return []
    # Split on commas into "red-ball" items, then each item on "-" into (color, type).
    return [tuple(item.split("-")) for item in spec.split(",")]


def faced_object(u):
    """The object in the cell directly in front of the agent, or None. u is the raw MiniGrid env."""
    # front_pos = (x, y) of the cell the agent faces; grid.get returns whatever is there
    # (None for empty floor). Walls are not objects a mission can name.
    obj = u.grid.get(*u.front_pos)
    return obj if obj is not None and obj.type in OBJECT_TYPES else None


def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval for a success rate; better than p +- 1.96 SE near 0 or 1."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)  # clamp float rounding at 0 and 1
