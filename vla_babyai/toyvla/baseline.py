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
import torch.nn as nn  # layers: Conv2d, GRU, Linear, ...
import torch.nn.functional as F  # stateless functions: relu, cross_entropy, ...
# Helpers that let a GRU skip padding when sequences in a batch have different lengths.
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .common import NUM_ACTIONS
from .data import load_split
from .evaluate import evaluate_all


class Vocab:
    """Maps mission words to integer ids, the form an embedding layer needs."""
    PAD, UNK = 0, 1  # id 0 fills short missions up to the batch length; id 1 = unknown word

    def __init__(self, words):
        # Fixed order: the two special tokens first, then every distinct word alphabetically.
        self.words = ["<pad>", "<unk>"] + sorted(set(words))
        self.index = {w: i for i, w in enumerate(self.words)}  # word -> id

    @classmethod
    def from_missions(cls, missions):
        # Every lower-cased word that appears in any training mission.
        return cls(w for m in missions for w in m.lower().split())

    def encode(self, missions):
        """List of strings -> (B, L) padded LongTensor and lengths."""
        # Words never seen in training map to UNK instead of raising an error.
        ids = [[self.index.get(w, self.UNK) for w in m.lower().split()] for m in missions]
        lengths = torch.tensor([len(x) for x in ids])
        # A (batch, longest mission) table filled with PAD, then each row overwritten
        # with its mission's ids from the left.
        out = torch.full((len(ids), int(lengths.max())), self.PAD, dtype=torch.long)
        for i, x in enumerate(ids):
            out[i, :len(x)] = torch.tensor(x)
        return out, lengths


class FiLMBlock(nn.Module):
    """A residual conv block whose features are scaled and shifted by the instruction.

    FiLM (Feature-wise Linear Modulation): the instruction produces one scale (gamma) and
    one shift (beta) per channel, so "red ball" can amplify channels that respond to red
    and round shapes. This is how language steers the vision features.
    """

    def __init__(self, channels, cond_dim):
        super().__init__()  # required: sets up nn.Module's parameter bookkeeping
        # Two 3x3 convolutions; padding=1 keeps the spatial size unchanged.
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)  # normalizes each channel; stabilizes training
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        # Instruction vector -> gamma and beta for every channel (hence 2 * channels outputs).
        self.film = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x, cond):
        # x: (B, C, H, W) image features; cond: (B, cond_dim) instruction vectors.
        # [:, :, None, None] adds two size-1 axes so (B, 2C) broadcasts over H and W;
        # chunk(2) splits the 2C channels into gamma (B, C, 1, 1) and beta (B, C, 1, 1).
        gamma, beta = self.film(cond)[:, :, None, None].chunk(2, dim=1)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        # Modulate, then add the input back (the residual connection eases training).
        return x + F.relu(gamma * h + beta)


class BabyAIModel(nn.Module):
    def __init__(self, vocab_size, channels=64, instr_dim=128, memory_dim=128, n_film=2):
        super().__init__()
        # Word id -> learned 128-dim vector. padding_idx keeps PAD's vector at zero.
        self.word_emb = nn.Embedding(vocab_size, instr_dim, padding_idx=Vocab.PAD)
        # Reads the word vectors in order; its final hidden state summarizes the mission.
        # batch_first=True means tensors are shaped (batch, time, features).
        self.instr_rnn = nn.GRU(instr_dim, instr_dim, batch_first=True)
        # Image stem: two conv layers, each followed by a 2x2 max-pool that halves height
        # and width, so a 56x56 frame becomes a 14x14 grid of `channels` features.
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2),
        )
        # ModuleList registers the blocks so their weights are trained and saved.
        self.films = nn.ModuleList(FiLMBlock(channels, instr_dim) for _ in range(n_film))
        # Memory across time steps: the agent sees only part of the room, so it must
        # remember what it saw and did earlier in the episode.
        self.memory = nn.GRU(channels, memory_dim, batch_first=True)
        # Small MLP from the memory state to one score (logit) per action.
        self.actor = nn.Sequential(nn.Linear(memory_dim, 64), nn.Tanh(), nn.Linear(64, NUM_ACTIONS))

    def encode_instr(self, tokens, lengths):
        # Packing tells the GRU each mission's true length so it stops before the padding.
        packed = pack_padded_sequence(self.word_emb(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.instr_rnn(packed)  # h: final hidden state, shape (1 layer, B, instr_dim)
        return h[0]  # (B, instr_dim)

    def encode_frames(self, frames, instr):
        """frames (N, H, W, 3) uint8, instr (N, instr_dim) -> (N, channels)."""
        # PyTorch convolutions expect (N, channels, H, W), so move the color axis forward,
        # and scale pixel values from 0..255 to 0..1.
        x = self.stem(frames.permute(0, 3, 1, 2).float() / 255.0)
        for film in self.films:
            x = film(x, instr)  # language-conditioned processing
        # Global max-pool: keep each channel's strongest response anywhere in the image.
        return x.amax(dim=(2, 3))

    def forward_step(self, frames, instr, hidden):
        """One time step for a batch of running episodes. hidden: (1, B, memory_dim)."""
        # [:, None] adds a time axis of length 1, because the GRU expects (B, T, features).
        out, hidden = self.memory(self.encode_frames(frames, instr)[:, None], hidden)
        # out[:, 0] drops the time axis again: (B, memory_dim) -> actor -> (B, 7) logits.
        return self.actor(out[:, 0]), hidden


def episode_batch(data, idx, vocab, device):
    """Pad a batch of episodes. Returns frames (B, T, H, W, 3), actions (B, T) with -100 padding, lengths, tokens."""
    lengths = torch.tensor([int(data.lengths[i]) for i in idx])
    T = int(lengths.max())  # every episode in the batch is padded to the longest one
    H, W = data.frames.shape[1:3]
    frames = torch.zeros(len(idx), T, H, W, 3, dtype=torch.uint8)
    # -100 marks padding steps; cross_entropy below ignores targets equal to -100.
    actions = torch.full((len(idx), T), -100, dtype=torch.long)
    for b, i in enumerate(idx):
        f, a, _ = data.episode(i)
        frames[b, :len(a)] = torch.from_numpy(f)  # copy the real steps in from the left
        actions[b, :len(a)] = torch.from_numpy(a)
    tokens, tok_len = vocab.encode([data.missions[i] for i in idx])
    return frames.to(device), actions.to(device), lengths, tokens.to(device), tok_len


def bc_loss(model, frames, actions, lengths, tokens, tok_len):
    """Behavior cloning: cross-entropy between the model's action scores and the bot's actions."""
    B, T = actions.shape
    instr = model.encode_instr(tokens, tok_len)  # (B, instr_dim), one vector per episode
    valid = actions >= 0  # (B, T) True on real steps, False on padding
    # Run the CNN only on real (non-padding) frames.
    feats = torch.zeros(B, T, model.memory.input_size, device=frames.device)
    # frames[valid] picks out only the real frames as one flat batch. The instruction is
    # repeated across time (expand) so every frame gets its own episode's mission vector.
    feats[valid] = model.encode_frames(frames[valid], instr[:, None].expand(B, T, -1)[valid])
    # Run the memory GRU over each episode's frames in order, skipping the padding.
    packed = pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
    out, _ = pad_packed_sequence(model.memory(packed)[0], batch_first=True, total_length=T)
    logits = model.actor(out)  # (B, T, 7): action scores at every step
    # Flatten to (B*T, 7) vs (B*T,) targets; padding targets (-100) don't count.
    loss = F.cross_entropy(logits.reshape(-1, NUM_ACTIONS), actions.reshape(-1), ignore_index=-100)
    # Fraction of real steps where the top-scoring action matches the bot's.
    acc = (logits.argmax(-1) == actions)[valid].float().mean()
    return loss, acc


class BaselinePolicy:
    """Adapter that lets evaluate.py drive the model one step at a time."""

    def __init__(self, model, vocab, device):
        # eval() switches BatchNorm to its stored running statistics for inference.
        self.model, self.vocab, self.device = model.eval(), vocab, device

    def reset(self, missions):
        tokens, lengths = self.vocab.encode(missions)
        # Encode each mission once per episode; it doesn't change during the episode.
        self.instr = self.model.encode_instr(tokens.to(self.device), lengths)
        # A fresh, empty memory for every episode.
        self.hidden = torch.zeros(1, len(missions), self.model.memory.hidden_size, device=self.device)

    def act(self, images, idx):
        # idx lists the episodes still running; only their instruction and memory are used.
        frames = torch.from_numpy(images).to(self.device)
        logits, h = self.model.forward_step(frames, self.instr[idx], self.hidden[:, idx])
        self.hidden[:, idx] = h  # store the updated memory for the next step
        return logits.argmax(-1).tolist()  # greedy: the highest-scoring action

    @classmethod
    def load(cls, ckpt, device):
        state = torch.load(ckpt, map_location=device)  # map_location: load onto CPU or GPU
        # The saved word list starts with "<pad>" and "<unk>"; Vocab adds those back itself.
        vocab = Vocab(state["vocab"][2:])
        # Rebuild the same architecture, then fill in the trained weights.
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
    p.add_argument("--eval-max-steps", type=int, help="cut evaluation episodes at this many steps (failures)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Fix the random seeds so a rerun gives the same weights and data order.
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    train = load_split(args.data, "train")
    vocab = Vocab.from_missions(train.missions)
    config = {"channels": args.channels}  # saved with the weights so load() can rebuild the model
    model = BabyAIModel(len(vocab.words), **config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("train: %d episodes, %d steps; %.2fM params"
          % (len(train), len(train.actions), sum(p.numel() for p in model.parameters()) / 1e6))

    for epoch in range(args.epochs):  # one epoch = one pass over every training episode
        model.train()  # BatchNorm uses batch statistics while training
        t0, losses, accs = time.time(), [], []
        order = rng.permutation(len(train))  # a new random episode order each epoch
        for b in range(0, len(order), args.batch_size):
            # The * unpacks episode_batch's 5 return values into bc_loss's arguments.
            loss, acc = bc_loss(model, *episode_batch(train, order[b:b + args.batch_size], vocab, device))
            opt.zero_grad()  # clear the gradients left over from the previous batch
            loss.backward()  # backpropagation: compute d(loss)/d(weight) for every weight
            # If the gradients' overall size exceeds 1.0, scale them down (guards against spikes).
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()  # update the weights along the gradients
            losses.append(loss.item())  # .item() turns a 1-element tensor into a Python float
            accs.append(acc.item())
        print("epoch %d  loss %.4f  action acc %.3f  (%.0fs)" % (epoch, np.mean(losses), np.mean(accs), time.time() - t0))

    ckpt = os.path.join(args.out, "model.pt")
    # state_dict() holds all weights; the vocab and config are needed to rebuild the model.
    torch.save({"model": model.state_dict(), "vocab": vocab.words, "config": config}, ckpt)
    print("saved %s; evaluating" % ckpt)
    results = evaluate_all(BaselinePolicy(model, vocab, device), args.data, args.eval_episodes,
                           max_steps=args.eval_max_steps)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
