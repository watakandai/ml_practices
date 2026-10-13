"""Loading the demo splits written by collect_demos."""
import json
import os

import numpy as np


class EpisodeData:
    """One split: all frames concatenated, with ep_start[i]:ep_start[i+1] marking episode i."""

    def __init__(self, path):
        d = np.load(path)
        self.frames = d["frames"]          # (N, H, W, 3) uint8
        self.actions = d["actions"].astype(np.int64)  # (N,)
        self.ep_start = d["ep_start"]      # (E + 1,)
        self.seeds = d["seeds"]            # (E,)
        self.missions = [str(m) for m in d["missions"]]  # (E,)
        self.lengths = np.diff(self.ep_start)
        # Episode index and time step of every frame.
        self.step_episode = np.repeat(np.arange(len(self.seeds)), self.lengths)
        self.step_t = np.arange(len(self.actions)) - self.ep_start[self.step_episode]

    def __len__(self):
        return len(self.seeds)

    def episode(self, i):
        s, e = self.ep_start[i], self.ep_start[i + 1]
        return self.frames[s:e], self.actions[s:e], self.missions[i]

    def history(self, step, k):
        """Actions taken before `step` within its episode, oldest first, at most k of them."""
        t = self.step_t[step]
        return self.actions[step - min(t, k):step].tolist()

    def bot_lengths(self):
        """Seed -> bot episode length, the reference for the episode-length metric."""
        return dict(zip(self.seeds.tolist(), self.lengths.tolist()))


def load_split(data_dir, split):
    return EpisodeData(os.path.join(data_dir, split + ".npz"))


def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)
