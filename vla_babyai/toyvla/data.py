"""Loading the demo splits written by collect_demos."""
import json
import os

import numpy as np


class EpisodeData:
    """One split: all frames concatenated, with ep_start[i]:ep_start[i+1] marking episode i.

    Storing every episode back to back in one big array (instead of a list of arrays)
    makes the file compact and lets training index any step directly by its position.
    """

    def __init__(self, path):
        d = np.load(path)  # an .npz file is a zip of named numpy arrays
        self.frames = d["frames"]          # (N, H, W, 3) uint8: N = total steps over all episodes
        self.actions = d["actions"].astype(np.int64)  # (N,) the bot's action at each step; int64 for PyTorch
        self.ep_start = d["ep_start"]      # (E + 1,) where each episode starts; the last entry is N
        self.seeds = d["seeds"]            # (E,) the seed that generated each episode's level
        self.missions = [str(m) for m in d["missions"]]  # (E,) instruction text, as plain Python strings
        # Episode lengths are the gaps between consecutive start positions.
        self.lengths = np.diff(self.ep_start)
        # Episode index and time step of every frame.
        # np.repeat writes episode index i lengths[i] times, e.g. lengths [2, 3] -> [0, 0, 1, 1, 1].
        self.step_episode = np.repeat(np.arange(len(self.seeds)), self.lengths)
        # Position within its own episode = global position - start of that episode.
        self.step_t = np.arange(len(self.actions)) - self.ep_start[self.step_episode]

    def __len__(self):
        # len(data) is the number of episodes, not steps.
        return len(self.seeds)

    def episode(self, i):
        # Slice episode i out of the concatenated arrays.
        s, e = self.ep_start[i], self.ep_start[i + 1]
        return self.frames[s:e], self.actions[s:e], self.missions[i]

    def history(self, step, k):
        """Actions taken before `step` within its episode, oldest first, at most k of them."""
        t = self.step_t[step]  # how many steps of this episode came before `step`
        # Go back min(t, k) steps, so the window never crosses into the previous episode.
        return self.actions[step - min(t, k):step].tolist()

    def bot_lengths(self):
        """Seed -> bot episode length, the reference for the episode-length metric."""
        # zip pairs each seed with its length; dict() turns the pairs into a lookup table.
        return dict(zip(self.seeds.tolist(), self.lengths.tolist()))


def load_split(data_dir, split):
    # e.g. load_split("data/GoToLocal", "train") reads data/GoToLocal/train.npz
    return EpisodeData(os.path.join(data_dir, split + ".npz"))


def load_meta(data_dir):
    # meta.json records the level name, tile size and held-out combos used to build the data.
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)
