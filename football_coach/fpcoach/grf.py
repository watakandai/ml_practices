"""Google Research Football (GRF) constants and the conversion from real pitches to GRF's frame.

Everything in this project lives in GRF's coordinates and action set, so a model trained on
real tracking data can be fine-tuned in GRF without changing its inputs or outputs.
"""
import numpy as np

# GRF's default action set (gfootball.env.football_action_set.action_set_v1), in index order.
# The environment's step() takes these integers.
ACTION_NAMES = [
    "idle", "left", "top_left", "top", "top_right", "right", "bottom_right", "bottom", "bottom_left",
    "long_pass", "high_pass", "short_pass", "shot", "sprint", "release_direction", "release_sprint",
    "sliding", "dribble", "release_dribble",
]
NUM_ACTIONS = len(ACTION_NAMES)  # 19
A = {name: i for i, name in enumerate(ACTION_NAMES)}  # name -> index, e.g. A["shot"] == 12

# The 8 movement actions, ordered by angle. GRF's y axis points down the screen ("bottom" is +y),
# so angle 0 is "right", +90 degrees is "bottom" and -90 degrees is "top".
DIRECTION_ACTIONS = [A["right"], A["bottom_right"], A["bottom"], A["bottom_left"],
                     A["left"], A["top_left"], A["top"], A["top_right"]]

# Action relabelling for a 180-degree rotation of the pitch (see match.attacking_frame):
# each direction swaps with its opposite, everything else is unchanged.
ROTATE_180 = np.arange(NUM_ACTIONS)
for _i, _a in enumerate(DIRECTION_ACTIONS):
    ROTATE_180[_a] = DIRECTION_ACTIONS[(_i + 4) % 8]

# GRF's pitch: x in [-1, 1] from the left goal line to the right one, y in [-0.42, 0.42] from the
# top touchline to the bottom one (scenario_builder.py). The left team attacks towards x = +1.
GRF_X_MAX = 1.0
GRF_Y_MAX = 0.42

# GRF runs at 10 steps per second of game time, so "*_direction" observations are displacements
# per 0.1 s. Real data is resampled to this rate. To re-check against the engine in M2.
GRF_STEP_SECONDS = 0.1

# GRF's "super mini-map" (SMM) observation (gfootball.env.observation_preprocessing).
# Note the minimap's y range is +-1/2.25 (about 0.444), slightly wider than the pitch's 0.42.
SMM_WIDTH = 96
SMM_HEIGHT = 72
SMM_LAYERS = ["left_team", "right_team", "ball", "active"]
SMM_X_RANGE = (-1.0, 1.0)
SMM_Y_RANGE = (-1.0 / 2.25, 1.0 / 2.25)
SMM_MARKER = 255


class PitchScale:
    """Converts between metres and GRF units for one real pitch.

    Real pitches are about 105 x 68 m (aspect 1.54) and GRF's is 2 x 0.84 (aspect 2.38), so the two
    axes scale differently. Metres here use GRF's orientation: origin at the centre spot, +x towards
    the right goal, +y towards the bottom touchline.
    """

    def __init__(self, length=105.0, width=68.0):
        self.length = float(length)
        self.width = float(width)
        # Metres per GRF unit along each axis.
        self.mx = self.length / 2 / GRF_X_MAX
        self.my = self.width / 2 / GRF_Y_MAX

    def to_grf(self, xy_m):
        """(..., 2) metres -> (..., 2) GRF units. Works for positions and for displacements."""
        xy_m = np.asarray(xy_m, dtype=np.float64)
        return np.stack([xy_m[..., 0] / self.mx, xy_m[..., 1] / self.my], axis=-1)

    def to_m(self, xy_grf):
        """(..., 2) GRF units -> (..., 2) metres."""
        xy_grf = np.asarray(xy_grf, dtype=np.float64)
        return np.stack([xy_grf[..., 0] * self.mx, xy_grf[..., 1] * self.my], axis=-1)

    def speed_mps(self, vel_grf):
        """(..., 2) GRF displacement per step -> (...,) speed in m/s."""
        return np.linalg.norm(self.to_m(vel_grf), axis=-1) / GRF_STEP_SECONDS
