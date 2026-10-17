"""Render states as GRF's super mini-map (SMM), so real and simulated frames look alike."""
import numpy as np

from .grf import SMM_HEIGHT, SMM_MARKER, SMM_WIDTH, SMM_X_RANGE, SMM_Y_RANGE


def smm_pixels(xy, width=SMM_WIDTH, height=SMM_HEIGHT):
    """GRF units [N, 2] -> integer pixel (col, row) [N, 2], exactly as GRF's mark_points does:
    truncate towards zero, then clip to the frame."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    col = ((xy[:, 0] - SMM_X_RANGE[0]) / (SMM_X_RANGE[1] - SMM_X_RANGE[0]) * width).astype(int)
    row = ((xy[:, 1] - SMM_Y_RANGE[0]) / (SMM_Y_RANGE[1] - SMM_Y_RANGE[0]) * height).astype(int)
    return np.stack([np.clip(col, 0, width - 1), np.clip(row, 0, height - 1)], axis=1)


def _mark(channel, xy):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xy = xy[~np.isnan(xy).any(axis=1)]  # hidden or off-pitch players have no marker
    if len(xy):
        cols_rows = smm_pixels(xy, channel.shape[1], channel.shape[0])
        channel[cols_rows[:, 1], cols_rows[:, 0]] = SMM_MARKER


def render_smm(own, opp, ball, active=None, width=SMM_WIDTH, height=SMM_HEIGHT):
    """[H, W, 4] uint8 minimap with GRF's layers: left team (own), right team (opp), ball, active.

    own, opp: [N, 2] positions in the attacking frame (NaN rows are skipped); ball: [2] or [3];
    active: [2] position of the controlled player, or None for an empty active layer.
    """
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    _mark(frame[..., 0], own)
    _mark(frame[..., 1], opp)
    _mark(frame[..., 2], np.asarray(ball)[:2])
    if active is not None:
        _mark(frame[..., 3], active)
    return frame
