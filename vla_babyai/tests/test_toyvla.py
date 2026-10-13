import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from tiny_vla import build_tiny_vla  # noqa: E402

from toyvla import baseline, train_vla  # noqa: E402
from toyvla.collect_demos import collect, rollout_bot, save_split  # noqa: E402
from toyvla.common import (DEFAULT_HELDOUT_COMBOS, NUM_ACTIONS, TEST_SEED_START, is_heldout_mission,  # noqa: E402
                           make_env, mission_combos)
from toyvla.data import load_split  # noqa: E402
from toyvla.evaluate import evaluate_all, run_episodes  # noqa: E402
from toyvla.vla import VLA, VLAPolicy  # noqa: E402

LEVEL = "GoToLocal"


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data") / LEVEL
    os.makedirs(d)
    combos = DEFAULT_HELDOUT_COMBOS
    train = collect(LEVEL, 0, {"train": (False, 60)}, combos, 8, 2)
    test = collect(LEVEL, TEST_SEED_START, {"test_seeds": (False, 8), "test_combos": (True, 8)}, combos, 8, 2)
    for name, eps in {**train, **test}.items():
        save_split(str(d / (name + ".npz")), eps)
    with open(d / "meta.json", "w") as f:
        json.dump({"level": LEVEL, "tile_size": 8, "heldout_combos": combos}, f)
    return str(d)


@pytest.fixture(scope="session")
def tiny_path(tmp_path_factory):
    return build_tiny_vla(str(tmp_path_factory.mktemp("tiny")))


class ScriptedPolicy:
    """Replays the bot's recorded actions (episode i of the split for env i), so evaluation
    must report full success at exactly the bot's episode lengths."""

    def __init__(self, data):
        self.data = data

    def reset(self, missions):
        assert missions == self.data.missions[:len(missions)]
        self.t = [0] * len(missions)

    def act(self, images, idx):
        out = []
        for i in idx:
            out.append(int(self.data.episode(i)[1][self.t[i]]))
            self.t[i] += 1
        return out


def test_mission_combos():
    assert mission_combos("go to the red ball") == [("red", "ball")]
    assert mission_combos("pick up a key") == [(None, "key")]
    assert is_heldout_mission("go to a green key and go to the box")
    assert not is_heldout_mission("go to the green ball")


def test_splits(data_dir):
    train = load_split(data_dir, "train")
    combos = load_split(data_dir, "test_combos")
    seeds = load_split(data_dir, "test_seeds")
    assert not any(is_heldout_mission(m) for m in train.missions + seeds.missions)
    assert all(is_heldout_mission(m) for m in combos.missions)
    assert set(train.seeds).isdisjoint(set(seeds.seeds) | set(combos.seeds))
    assert train.frames.shape[1:] == (56, 56, 3) and len(train.frames) == len(train.actions)
    # History is within-episode and oldest first.
    s = train.ep_start[int(np.argmax(train.lengths >= 3))]
    assert train.history(s, 4) == []
    assert train.history(s + 2, 4) == train.actions[s:s + 2].tolist()


def test_bot_replay_evaluates_to_full_success(data_dir):
    data = load_split(data_dir, "test_seeds")
    policy = ScriptedPolicy(data)
    success, length = run_episodes(policy, LEVEL, data.seeds.tolist(), batch_size=len(data))
    assert success.all()
    assert (length == data.lengths).all()


def test_rollout_is_deterministic_per_seed():
    env = make_env(LEVEL)
    a, b = rollout_bot(env, 123), rollout_bot(env, 123)
    assert a[2] == b[2] and (a[1] == b[1]).all()


def test_baseline_learns_and_evaluates(data_dir):
    torch.manual_seed(0)
    data = load_split(data_dir, "train")
    vocab = baseline.Vocab.from_missions(data.missions)
    model = baseline.BabyAIModel(len(vocab.words), channels=16)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    batch = baseline.episode_batch(data, list(range(16)), vocab, "cpu")
    first = baseline.bc_loss(model, *batch)[0].item()
    for _ in range(40):
        loss, _ = baseline.bc_loss(model, *batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5
    results = evaluate_all(baseline.BaselinePolicy(model, vocab, "cpu"), data_dir, episodes=4, batch_size=3)
    assert set(results) == {"seen_seeds", "heldout_seeds", "heldout_combos"}


def test_baseline_step_matches_sequence(data_dir):
    """The step-by-step policy must reproduce the training-time sequence logits."""
    data = load_split(data_dir, "train")
    vocab = baseline.Vocab.from_missions(data.missions)
    model = baseline.BabyAIModel(len(vocab.words), channels=16).eval()
    frames, actions, mission = data.episode(0)
    with torch.no_grad():
        policy = baseline.BaselinePolicy(model, vocab, "cpu")
        policy.reset([mission])
        step_logits = []
        for f in frames:
            logits, policy.hidden = model.forward_step(torch.from_numpy(f[None]), policy.instr, policy.hidden)
            step_logits.append(logits[0])
        instr = model.encode_instr(*vocab.encode([mission]))
        feats = model.encode_frames(torch.from_numpy(frames), instr.expand(len(frames), -1))
        seq_logits = model.actor(model.memory(feats[None])[0])[0]
    assert torch.allclose(torch.stack(step_logits), seq_logits, atol=1e-5)


def test_vla_action_tokens_and_prompt(tiny_path):
    vla = VLA.from_pretrained(tiny_path, history=2, image_size=64)
    assert len(set(vla.action_ids.tolist())) == NUM_ACTIONS
    prompt = vla.encode.prompt("go to the red ball", [0, 1, 2])
    assert "Recent actions: right forward" in prompt and prompt.endswith("Assistant:")
    assert "Recent actions" not in VLA.from_pretrained(tiny_path, history=0, image_size=64).encode.prompt("x", [1])


@pytest.mark.parametrize("image_size", [64, 32])
def test_vla_overfits_batch(tiny_path, data_dir, image_size):
    torch.manual_seed(0)
    vla = VLA.from_pretrained(tiny_path, image_size=image_size)
    data = load_split(data_dir, "train")
    ds = train_vla.StepDataset(data, 4)
    inputs, actions = train_vla.Collate(vla.encode)([ds[i] for i in range(16)])
    # Left padding puts every row's prediction slot at the last position.
    assert (inputs["attention_mask"][:, -1] == 1).all()
    opt = torch.optim.AdamW(vla.model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    first, _ = train_vla.forward_backward(vla, inputs, actions, scaler, "cpu")
    opt.zero_grad()
    for _ in range(60):
        loss, acc = train_vla.forward_backward(vla, inputs, actions, scaler, "cpu")
        train_vla.optimizer_step(vla, opt, scaler)
    assert loss < first * 0.3 and acc == 1.0


def test_vla_policy_and_save_load(tiny_path, data_dir, tmp_path):
    vla = VLA.from_pretrained(tiny_path, image_size=64)
    policy = VLAPolicy(vla, "cpu")
    seeds = load_split(data_dir, "test_seeds").seeds[:3].tolist()
    success, length = run_episodes(policy, LEVEL, seeds, batch_size=2)
    assert len(success) == 3 and (length >= 1).all()
    assert all(0 <= a < NUM_ACTIONS for h in policy.histories for a in h)

    vla.save(str(tmp_path / "ckpt"))
    loaded = VLAPolicy.load(str(tmp_path / "ckpt"), "cpu")
    frames = load_split(data_dir, "train").frames[:2]
    inputs = vla.encode(frames, ["go to the red ball"] * 2, [[1], []])
    with torch.no_grad():
        assert torch.allclose(vla.logits(inputs), loaded.vla.logits(inputs), atol=1e-5)


def test_vla_lora_trains_only_adapters(tiny_path):
    vla = VLA.from_pretrained(tiny_path, image_size=64, lora_rank=4)
    names = [n for n, p in vla.model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)


def test_train_vla_cli(tiny_path, data_dir, tmp_path, monkeypatch):
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "train_vla", "--model", tiny_path, "--data", data_dir, "--out", str(out), "--image-size", "64",
        "--micro-batch", "4", "--grad-accum", "2", "--max-steps", "4", "--warmup", "1", "--log-every", "2",
        "--eval-every", "2", "--eval-episodes", "2", "--final-eval-episodes", "2", "--workers", "0",
        "--lora-rank", "4"])
    train_vla.main()
    results = json.load(open(out / "results.json"))
    assert set(results) == {"seen_seeds", "heldout_seeds", "heldout_combos"}
    assert os.path.exists(out / "final" / "vla_config.json")
