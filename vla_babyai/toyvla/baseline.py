"""M3: the BabyAI-style CNN + GRU + FiLM policy, trained from scratch with behavior cloning.

The architecture follows the BabyAI paper (Chevalier-Boisvert et al., 2019), with the
symbolic-grid input swapped for the same RGB frames the VLA sees:
  instruction -> word embeddings -> GRU -> FiLM parameters
  frame -> conv stem -> FiLM-conditioned residual conv blocks -> max pool
  -> GRU memory over time -> actor MLP over the 7 actions

  python -m toyvla.baseline --data data/GoToLocal --out runs/baseline/GoToLocal
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .common import NUM_ACTIONS
from .data import load_split
from .evaluate import evaluate_all


class Vocab:
    PAD, UNK = 0, 1

    def __init__(self, words):
        self.words = ["<pad>", "<unk>"] + sorted(set(words))
        self.index = {w: i for i, w in enumerate(self.words)}

    @classmethod
    def from_missions(cls, missions):
        return cls(w for m in missions for w in m.lower().split())

    def encode(self, missions):
        """List of strings -> (B, L) padded LongTensor and lengths."""
        ids = [[self.index.get(w, self.UNK) for w in m.lower().split()] for m in missions]
        lengths = torch.tensor([len(x) for x in ids])
        out = torch.full((len(ids), int(lengths.max())), self.PAD, dtype=torch.long)
        for i, x in enumerate(ids):
            out[i, :len(x)] = torch.tensor(x)
        return out, lengths


class FiLMBlock(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.film = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x, cond):
        gamma, beta = self.film(cond)[:, :, None, None].chunk(2, dim=1)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return x + F.relu(gamma * h + beta)


class BabyAIModel(nn.Module):
    def __init__(self, vocab_size, channels=64, instr_dim=128, memory_dim=128, n_film=2):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, instr_dim, padding_idx=Vocab.PAD)
        self.instr_rnn = nn.GRU(instr_dim, instr_dim, batch_first=True)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.films = nn.ModuleList(FiLMBlock(channels, instr_dim) for _ in range(n_film))
        self.memory = nn.GRU(channels, memory_dim, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(memory_dim, 64), nn.Tanh(), nn.Linear(64, NUM_ACTIONS))

    def encode_instr(self, tokens, lengths):
        packed = pack_padded_sequence(self.word_emb(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.instr_rnn(packed)
        return h[0]  # (B, instr_dim)

    def encode_frames(self, frames, instr):
        """frames (N, H, W, 3) uint8, instr (N, instr_dim) -> (N, channels)."""
        x = self.stem(frames.permute(0, 3, 1, 2).float() / 255.0)
        for film in self.films:
            x = film(x, instr)
        return x.amax(dim=(2, 3))

    def forward_step(self, frames, instr, hidden):
        """One time step for a batch of running episodes. hidden: (1, B, memory_dim)."""
        out, hidden = self.memory(self.encode_frames(frames, instr)[:, None], hidden)
        return self.actor(out[:, 0]), hidden


def episode_batch(data, idx, vocab, device):
    """Pad a batch of episodes. Returns frames (B, T, H, W, 3), actions (B, T) with -100 padding, lengths, tokens."""
    lengths = torch.tensor([int(data.lengths[i]) for i in idx])
    T = int(lengths.max())
    H, W = data.frames.shape[1:3]
    frames = torch.zeros(len(idx), T, H, W, 3, dtype=torch.uint8)
    actions = torch.full((len(idx), T), -100, dtype=torch.long)
    for b, i in enumerate(idx):
        f, a, _ = data.episode(i)
        frames[b, :len(a)] = torch.from_numpy(f)
        actions[b, :len(a)] = torch.from_numpy(a)
    tokens, tok_len = vocab.encode([data.missions[i] for i in idx])
    return frames.to(device), actions.to(device), lengths, tokens.to(device), tok_len


def bc_loss(model, frames, actions, lengths, tokens, tok_len):
    B, T = actions.shape
    instr = model.encode_instr(tokens, tok_len)
    valid = actions >= 0
    # Run the CNN only on real (non-padding) frames.
    feats = torch.zeros(B, T, model.memory.input_size, device=frames.device)
    feats[valid] = model.encode_frames(frames[valid], instr[:, None].expand(B, T, -1)[valid])
    packed = pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
    out, _ = pad_packed_sequence(model.memory(packed)[0], batch_first=True, total_length=T)
    logits = model.actor(out)
    loss = F.cross_entropy(logits.reshape(-1, NUM_ACTIONS), actions.reshape(-1), ignore_index=-100)
    acc = (logits.argmax(-1) == actions)[valid].float().mean()
    return loss, acc


class BaselinePolicy:
    def __init__(self, model, vocab, device):
        self.model, self.vocab, self.device = model.eval(), vocab, device

    def reset(self, missions):
        tokens, lengths = self.vocab.encode(missions)
        self.instr = self.model.encode_instr(tokens.to(self.device), lengths)
        self.hidden = torch.zeros(1, len(missions), self.model.memory.hidden_size, device=self.device)

    def act(self, images, idx):
        frames = torch.from_numpy(images).to(self.device)
        logits, h = self.model.forward_step(frames, self.instr[idx], self.hidden[:, idx])
        self.hidden[:, idx] = h
        return logits.argmax(-1).tolist()

    @classmethod
    def load(cls, ckpt, device):
        state = torch.load(ckpt, map_location=device)
        vocab = Vocab(state["vocab"][2:])
        model = BabyAIModel(len(vocab.words), **state["config"]).to(device)
        model.load_state_dict(state["model"])
        return cls(model, vocab, device)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/GoToLocal")
    p.add_argument("--out", default="runs/baseline/GoToLocal")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64, help="episodes per batch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--eval-episodes", type=int, default=200, help="per split, after training")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    train = load_split(args.data, "train")
    vocab = Vocab.from_missions(train.missions)
    config = {"channels": args.channels}
    model = BabyAIModel(len(vocab.words), **config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("train: %d episodes, %d steps; %.2fM params"
          % (len(train), len(train.actions), sum(p.numel() for p in model.parameters()) / 1e6))

    for epoch in range(args.epochs):
        model.train()
        t0, losses, accs = time.time(), [], []
        order = rng.permutation(len(train))
        for b in range(0, len(order), args.batch_size):
            loss, acc = bc_loss(model, *episode_batch(train, order[b:b + args.batch_size], vocab, device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            accs.append(acc.item())
        print("epoch %d  loss %.4f  action acc %.3f  (%.0fs)" % (epoch, np.mean(losses), np.mean(accs), time.time() - t0))

    ckpt = os.path.join(args.out, "model.pt")
    torch.save({"model": model.state_dict(), "vocab": vocab.words, "config": config}, ckpt)
    print("saved %s; evaluating" % ckpt)
    results = evaluate_all(BaselinePolicy(model, vocab, device), args.data, args.eval_episodes)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
