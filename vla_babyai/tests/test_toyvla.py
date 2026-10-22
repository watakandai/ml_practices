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
from toyvla.common import (DEFAULT_HELDOUT_COMBOS, NUM_ACTIONS, TEST_SEED_START, VAL_SEED_START,  # noqa: E402
                           is_heldout_mission, make_env, mission_combos, wilson_ci)
from toyvla.data import load_split  # noqa: E402
from toyvla.evaluate import evaluate_all, evaluate_split, run_episodes  # noqa: E402
from toyvla.vla import NULL_MISSION, VLA, VLAPolicy  # noqa: E402

LEVEL = "GoToLocal"


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data") / LEVEL
    os.makedirs(d)
    combos = DEFAULT_HELDOUT_COMBOS
    train = collect(LEVEL, 0, {"train": (False, 60)}, combos, 8, 2)
    val = collect(LEVEL, VAL_SEED_START, {"val": (False, 6)}, combos, 8, 2)
    test = collect(LEVEL, TEST_SEED_START, {"test_seeds": (False, 8), "test_combos": (True, 8)}, combos, 8, 2)
    for name, eps in {**train, **val, **test}.items():
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
    success, length, *_ = run_episodes(policy, LEVEL, data.seeds.tolist(), batch_size=len(data))
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
    success, length, *_ = run_episodes(policy, LEVEL, seeds, batch_size=2)
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


def test_init_optimizer_state_allocates_without_changing_weights():
    model = torch.nn.Linear(8, 4)
    before = [p.clone() for p in model.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    train_vla.init_optimizer_state(opt)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all("exp_avg_sq" in opt.state[p] for p in model.parameters())
    assert all(p.grad is None for p in model.parameters())
    assert opt.param_groups[0]["lr"] == 1e-3


def test_train_vla_resumes_from_checkpoint(tiny_path, data_dir, tmp_path, monkeypatch):
    out = tmp_path / "run"
    os.makedirs(out)
    vla = VLA.from_pretrained(tiny_path, image_size=64)
    opt = torch.optim.AdamW(vla.model.parameters(), lr=1e-4)
    sched = train_vla.get_cosine_schedule_with_warmup(opt, 1, 4)
    # Same setting as train_vla's own scaler, whose saved state it must load.
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    train_vla.init_optimizer_state(opt)
    for _ in range(3):
        opt.step()
        sched.step()
    train_vla.save_checkpoint(str(out / "last.pt"), 3, vla, opt, sched, scaler)

    monkeypatch.setattr(sys, "argv", [
        "train_vla", "--model", tiny_path, "--data", data_dir, "--out", str(out), "--image-size", "64",
        "--micro-batch", "2", "--grad-accum", "1", "--max-steps", "4", "--warmup", "1", "--log-every", "1",
        "--eval-every", "0", "--final-eval-episodes", "1", "--workers", "0", "--resume"])
    train_vla.main()
    steps = [json.loads(line)["step"] for line in open(out / "log.jsonl") if '"step"' in line]
    assert steps == [4]  # steps 1-3 came from the checkpoint
    assert not os.path.exists(out / "last.pt")


def test_diagnose_bot_replay_and_render(data_dir, tmp_path):
    from toyvla import diagnose, render
    data = load_split(data_dir, "test_combos")
    records = diagnose.run(diagnose.ReplayPolicy(data), LEVEL, 8, data.seeds.tolist(), data.bot_lengths(), 3)
    s = diagnose.summarize(records)
    assert s["success_rate"] == 1.0 and s["length_ratio_successes"]["mean"] == 1.0
    assert all(r["faced"][-1] == "%s %s" % tuple(r["target"]) for r in records if r["target"][0])
    assert all(len(r["wrong"]) == len(r["faced"]) - 1 for r in records)

    env = make_env(LEVEL, 8, render_mode="rgb_array")
    frames, success = render.rollout(diagnose.ReplayPolicy(data), env, int(data.seeds[0]), int(data.lengths[0]))
    assert success and len(frames) == data.lengths[0] + 1
    render.save_gif(frames, str(tmp_path / "ep.gif"))
    assert os.path.getsize(tmp_path / "ep.gif") > 0


def test_diagnose_shuffle_missions(data_dir):
    from toyvla import diagnose
    data = load_split(data_dir, "test_seeds")

    class Recorder(diagnose.ReplayPolicy):
        def reset(self, missions):
            self.given = list(missions)
            super().reset(missions)

    policy = Recorder(data)
    records = diagnose.run(policy, LEVEL, 8, data.seeds.tolist(), data.bot_lengths(), 4, shuffle_missions=True)
    assert [r["given_mission"] for r in records[-len(policy.given):]] == policy.given
    assert any(r["given_mission"] != r["mission"] for r in records)
    assert all(r["success"] for r in records)  # replayed bot actions don't depend on the mission


class ConstantPolicy:
    def reset(self, missions):
        pass

    def act(self, images, idx):
        return [0] * len(idx)  # turn left forever: never succeeds


def test_eval_max_steps_cuts_episodes(data_dir):
    seeds = load_split(data_dir, "test_seeds").seeds[:3].tolist()
    success, length, *_ = run_episodes(ConstantPolicy(), LEVEL, seeds, max_steps=5)
    assert not success.any() and (length == 5).all()


def test_rl_group_advantages():
    from toyvla.rl import group_advantages
    adv = group_advantages(np.array([1.0, 0.0, 0.5, 0.5]), group=2)
    assert np.allclose(adv[:2], [1, -1], atol=1e-4) and np.isnan(adv[2:]).all()


def test_rl_rollout_and_update(tiny_path, data_dir):
    from toyvla import rl
    torch.manual_seed(0)
    vla = VLA.from_pretrained(tiny_path, image_size=64)
    group = 3
    envs = [make_env(LEVEL) for _ in range(2 * group)]
    seeds = rl.sample_seeds(LEVEL, 2, np.random.default_rng(0), DEFAULT_HELDOUT_COMBOS)
    assert all(s >= rl.RL_SEED_START for s in seeds)
    reward, length, success, missions, steps = rl.rollout(vla, envs, seeds, group, "cpu", chunk=4)
    assert len(steps["ep"]) == length.sum() and missions[0] == missions[group - 1]
    assert not any(is_heldout_mission(m) for m in missions)
    assert all(lp <= 0 for lp in steps["logp"])

    # The PPO ratio is 1 on the rollout policy, so the clipped loss reduces to -mean(advantage).
    idx = list(range(4))
    inputs = vla.encode(np.stack([steps["frame"][i] for i in idx]), [missions[steps["ep"][i]] for i in idx],
                        [steps["hist"][i] for i in idx])
    adv = torch.tensor([1.0, -1.0, 0.5, 2.0])
    loss, stats = rl.ppo_loss(vla, inputs, torch.tensor([steps["action"][i] for i in idx]),
                              torch.tensor([steps["logp"][i] for i in idx]), adv)
    assert abs(loss.item() + adv.mean().item()) < 1e-4 and stats["clip_frac"] == 0

    args = type("A", (), {"micro_batch": 2, "grad_accum": 2, "ppo_epochs": 1, "temperature": 1.0, "clip": 0.2,
                          "max_grad_norm": 1.0})()
    fake_adv = np.where(np.arange(len(reward)) % 2 == 0, 1.0, -1.0)
    before = [p.clone() for p in vla.model.parameters()]
    opt = torch.optim.AdamW(vla.model.parameters(), lr=1e-3)
    out = rl.update(vla, opt, torch.amp.GradScaler("cuda", enabled=False), missions, steps, fake_adv, args, "cpu",
                    np.random.default_rng(0))
    assert out["train_steps"] == len(steps["ep"]) and out["optimizer_steps"] >= 1
    assert any(not torch.equal(a, b) for a, b in zip(before, vla.model.parameters()))


def test_rl_cli(tiny_path, data_dir, tmp_path, monkeypatch):
    from toyvla import rl
    init = tmp_path / "init"
    VLA.from_pretrained(tiny_path, image_size=64).save(str(init))
    out = tmp_path / "rl"
    monkeypatch.setattr(sys, "argv", [
        "rl", "--init", str(init), "--data", data_dir, "--out", str(out), "--iters", "2", "--seeds-per-iter", "2",
        "--group", "2", "--micro-batch", "2", "--grad-accum", "2", "--eval-every", "1", "--eval-episodes", "2",
        "--final-eval-episodes", "2", "--eval-max-steps", "10", "--save-every", "1"])
    rl.main()
    assert set(json.load(open(out / "results.json"))) == {"seen_seeds", "heldout_seeds", "heldout_combos"}
    assert not os.path.exists(out / "last.pt")


def test_shuffle_missions_rotates_and_marks_same_target(data_dir):
    data = load_split(data_dir, "test_seeds")
    seeds = data.seeds[:4].tolist()
    given = []

    class Recorder(ConstantPolicy):
        def reset(self, missions):
            given.extend(missions)

    _, _, same, _ = run_episodes(Recorder(), LEVEL, seeds, max_steps=2, shuffle_missions=True)
    true = data.missions[:4]
    assert given == true[1:] + true[:1]
    assert same.tolist() == [mission_combos(g) == mission_combos(m) for g, m in zip(given, true)]


def test_random_init_and_train_episodes(tiny_path, data_dir):
    pre = VLA.from_pretrained(tiny_path, image_size=64)
    rnd = VLA.from_pretrained(tiny_path, image_size=64, random_init=True)
    assert rnd.action_ids.tolist() == pre.action_ids.tolist()  # same tokenizer and action head
    assert any(not torch.equal(a, b) for a, b in zip(pre.model.parameters(), rnd.model.parameters()))
    train = load_split(data_dir, "train")
    assert len(train_vla.StepDataset(train, 4, episodes=5)) == train.ep_start[5]
    assert len(train_vla.StepDataset(train, 4)) == len(train.actions)


def test_direct_success_matches_diagnose(data_dir):
    from toyvla import diagnose
    data = load_split(data_dir, "test_combos")
    r = evaluate_split(ScriptedPolicy(data), LEVEL, data.seeds.tolist(), data.bot_lengths(), batch_size=len(data))
    s = diagnose.summarize(diagnose.run(diagnose.ReplayPolicy(data), LEVEL, 8, data.seeds.tolist(),
                                        data.bot_lengths(), len(data)))
    assert r["success_rate"] == 1.0 and r["direct_success_rate"] == s["direct_successes"]
    lo, hi = r["direct_success_ci95"]
    assert lo <= r["direct_success_rate"] <= hi
    # Turning left forever faces whatever is next to the agent, never reaching a target in GoToLocal.
    _, _, _, wrong = run_episodes(ConstantPolicy(), LEVEL, data.seeds[:3].tolist(), max_steps=8)
    assert (wrong >= 0).all()


def test_wilson_and_summarize():
    from toyvla.summarize import across_runs, summarize
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi and abs((hi - lo) - 0.19) < 0.01
    assert wilson_ci(0, 10)[0] == 0.0
    runs = [{"heldout_combos": {"success_rate": v, "length_ratio": 1.0}} for v in (0.5, 0.6, 0.7)]
    s = summarize(runs)["heldout_combos"]["success_rate"]
    assert abs(s["mean"] - 0.6) < 1e-9 and abs(s["ci95"] - 4.303 * 0.1 / 3 ** 0.5) < 1e-6
    assert np.isnan(across_runs([0.5])[1])


def test_train_vla_selects_on_validation(tiny_path, data_dir, tmp_path, monkeypatch):
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "train_vla", "--model", tiny_path, "--data", data_dir, "--out", str(out), "--image-size", "64",
        "--micro-batch", "2", "--grad-accum", "1", "--max-steps", "4", "--warmup", "1", "--log-every", "2",
        "--eval-every", "2", "--eval-episodes", "2", "--val-steps", "8", "--final-eval-episodes", "2",
        "--eval-max-steps", "5", "--workers", "0"])
    train_vla.main()
    evals = [json.loads(line)["eval_val"] for line in open(out / "log.jsonl") if "eval_val" in line]
    assert [e["step"] for e in evals] == [2, 4] and all(e["val_loss"] > 0 for e in evals)
    sel = json.load(open(out / "selection.json"))
    best_step = min(evals, key=lambda e: e["val_loss"])["step"]
    assert sel["step"] == best_step and sel["selected"] == ("final" if best_step == 4 else "best")
    assert os.path.exists(out / "best") == (best_step != 4)
    assert "direct_success_rate" in json.load(open(out / "results.json"))["heldout_combos"]


def test_collate_mission_dropout(data_dir):
    seen = []
    collate = train_vla.Collate(lambda frames, missions, histories: seen.append(missions), mission_dropout=0.5)
    data = load_split(data_dir, "train")
    steps = train_vla.StepDataset(data, 4)
    items = [steps[i] for i in range(len(steps))]
    assert len(items) >= 200
    torch.manual_seed(0)
    collate(items)
    dropped = [m == NULL_MISSION for m in seen[0]]
    assert 0.4 < np.mean(dropped) < 0.6
    assert all(m == it[1] for m, it, d in zip(seen[0], items, dropped) if not d)
    train_vla.Collate(lambda f, m, h: seen.append(m))(items)
    assert seen[1] == [it[1] for it in items]


def test_vla_policy_guidance(tiny_path, data_dir):
    vla = VLA.from_pretrained(tiny_path, image_size=64)
    frames = load_split(data_dir, "train").frames[:4]
    missions = ["go to the red ball", "go to a key", "go to the blue box", "go to the grey key"]
    plain = VLAPolicy(vla, "cpu")
    cond = plain.action_logits(frames, missions, [[]] * 4)
    uncond = plain.action_logits(frames, [NULL_MISSION] * 4, [[]] * 4)
    for w, expected in [(1.0, cond), (0.0, uncond), (3.0, uncond + 3.0 * (cond - uncond))]:
        policy = VLAPolicy(vla, "cpu", guidance=w)
        policy.reset(missions)
        assert policy.act(frames, list(range(4))) == expected.argmax(-1).tolist()


def test_train_vla_mission_dropout_and_guidance(tiny_path, data_dir, tmp_path, monkeypatch):
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "train_vla", "--model", tiny_path, "--data", data_dir, "--out", str(out), "--image-size", "64",
        "--micro-batch", "2", "--grad-accum", "1", "--max-steps", "2", "--warmup", "1", "--eval-every", "0",
        "--final-eval-episodes", "2", "--eval-max-steps", "5", "--workers", "0",
        "--mission-dropout", "0.5", "--final-guidance", "2"])
    train_vla.main()
    for name in ["results.json", "results_guidance2.json"]:
        assert set(json.load(open(out / name))) == {"seen_seeds", "heldout_seeds", "heldout_combos"}
