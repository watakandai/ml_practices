"""Hide players during training so the model copes with the partial views of broadcast video."""
from dataclasses import dataclass

import numpy as np


@dataclass
class MaskConfig:
    p_camera: float = 0.5        # chance a sample uses the camera rule
    camera_radius: float = 30.0  # m; the camera rule keeps players within this of the ball
    p_drop: float = 0.1          # chance each remaining player is dropped at random
    keep: tuple = ()             # slots that are never hidden (e.g. the player being labelled)


def visibility_mask(pos_m, ball_m, visible, rng, cfg=MaskConfig()):
    """[P] bool: which of the `visible` players stay visible.

    pos_m: [P, 2] player positions in metres; ball_m: [2]; visible: [P] bool (on the pitch).
    The camera rule approximates a broadcast frame centred on the ball: a disc, which is cruder
    than the real trapezoid but keeps the same structure (near the ball visible, far side not).
    """
    keep = visible.copy()
    if rng.random() < cfg.p_camera and not np.any(np.isnan(ball_m)):
        dist = np.linalg.norm(pos_m - ball_m, axis=-1)
        keep &= np.nan_to_num(dist, nan=np.inf) <= cfg.camera_radius
    keep &= rng.random(len(keep)) >= cfg.p_drop
    for s in cfg.keep:
        keep[s] = visible[s]
    return keep
