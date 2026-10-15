"""M7: RL fine-tuning of an SFT VLA with group-relative PPO (GRPO without the KL term).

Each iteration:
  1. Draw `--seeds-per-iter` fresh levels from a seed range used by neither the demos nor the test
     splits, skipping any whose mission names a held-out combo (so held-out combos stay unseen).
  2. Roll out `--group` sampled episodes per level with the current policy. The policy is the
     softmax over the 7 action-token logits, i.e. exactly the distribution SFT trained and greedy
     evaluation takes the argmax of.
  3. Reward = BabyAI's own episode reward, 1 - 0.9 * steps / max_steps on success and 0 on failure,
     so shorter successes score higher. Advantage = (reward - group mean) / group std; levels
     where every episode got the same reward carry no signal and are dropped.
  4. `--ppo-epochs` passes of the PPO clipped objective over every step of the kept episodes,
     averaged per step, with the rollout log-probabilities as the old policy.

  python -m toyvla.rl --init runs/vla256/GoToLocal/final --data data/GoToLocal --out runs/vla256_rl/GoToLocal
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .common import is_heldout_mission, make_env, quiet_reset
from .data import load_meta, load_split
from .evaluate import evaluate_all, evaluate_split
from .train_vla import autocast, init_optimizer_state, optimizer_step
from .vla import VLA, VLAPolicy

# Fresh levels for RL: above the ~10k-50k training seeds the demos use, below the test range (10M).
RL_SEED_START = 1_000_000


def sample_seeds(level, n, rng, heldout_combos, tile_size=8):
    """n seeds from the RL range whose missions name no held-out combo."""
    env = make_env(level, tile_size)
    seeds = []
    while len(seeds) < n:
        s = int(rng.integers(RL_SEED_START, RL_SEED_START + 8_000_000))
        if not is_heldout_mission(quiet_reset(env, s)[0]["mission"], heldout_combos):
            seeds.append(s)
    env.close()
    return seeds


@torch.no_grad()
def rollout(vla, envs, seeds, group, device, temperature=1.0, chunk=32):
    """`group` sampled episodes per seed (envs[j] runs seed j // group). Returns per-episode rewards,
    lengths and successes, and per-step lists (episode index, frame, mission, history, action, log-prob)."""
    vla.model.eval()
    n = len(seeds) * group
    obs = [quiet_reset(envs[j], seeds[j // group])[0] for j in range(n)]
    missions = [o["mission"] for o in obs]
    hist = [[] for _ in range(n)]
    reward, length, success = np.zeros(n), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
    steps = {"ep": [], "frame": [], "hist": [], "action": [], "logp": []}
    running = list(range(n))
    while running:
        actions, logps = [], []
        for c in range(0, len(running), chunk):
            idx = running[c:c + chunk]
            inputs = vla.encode(np.stack([obs[j]["image"] for j in idx]), [missions[j] for j in idx],
                                [hist[j] for j in idx])
            with autocast(device):
                logits = vla.constrain(vla.logits(inputs.to(device))) / temperature
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            actions += a.tolist()
            logps += dist.log_prob(a).tolist()
        still = []
        for j, a, lp in zip(running, actions, logps):
            steps["ep"].append(j)
            steps["frame"].append(obs[j]["image"])
            steps["hist"].append(list(hist[j]))
            steps["action"].append(a)
            steps["logp"].append(lp)
            hist[j].append(a)
            obs[j], r, terminated, truncated, _ = envs[j].step(a)
            length[j] += 1
            if terminated or truncated:
                reward[j], success[j] = r, r > 0
            else:
                still.append(j)
        running = still
    return reward, length, success, missions, steps


def group_advantages(reward, group, eps=1e-6):
    """(reward - group mean) / group std per seed; NaN marks groups with no reward spread."""
    r = reward.reshape(-1, group)
    std = r.std(axis=1, keepdims=True)
    adv = (r - r.mean(axis=1, keepdims=True)) / (std + eps)
    adv[np.broadcast_to(std < eps, adv.shape)] = np.nan
    return adv.reshape(-1)


def ppo_loss(vla, inputs, actions, old_logp, adv, temperature=1.0, clip=0.2):
    """Clipped surrogate, mean over steps. Also returns the approximate KL to the rollout policy,
    the clipped fraction and the policy entropy, for logging."""
    logits = vla.constrain(vla.logits(inputs)) / temperature
    logp_all = torch.log_softmax(logits, -1)
    logp = logp_all.gather(1, actions[:, None])[:, 0]
    ratio = torch.exp(logp - old_logp)
    loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    with torch.no_grad():
        stats = {"approx_kl": ((ratio - 1) - (logp - old_logp)).mean().item(),
                 "clip_frac": ((ratio - 1).abs() > clip).float().mean().item(),
                 "entropy": -(logp_all.exp() * logp_all).sum(-1).mean().item()}
    return loss, stats


def update(vla, opt, scaler, missions, steps, adv, args, device, rng):
    """PPO epochs over every step of the episodes with a finite advantage. Returns logging stats."""
    keep = [i for i, j in enumerate(steps["ep"]) if math.isfinite(adv[j])]
    if not keep:
        return {"train_steps": 0}
    vla.model.train()
    stats, nonfinite, updates = [], 0, 0
    mb = args.micro_batch
    per_update = mb * args.grad_accum
    for _ in range(args.ppo_epochs):
        order = rng.permutation(keep)
        for u in range(0, len(order), per_update):
            chunk = order[u:u + per_update]
            for c in range(0, len(chunk), mb):
                idx = chunk[c:c + mb]
                inputs = vla.encode(np.stack([steps["frame"][i] for i in idx]),
                                    [missions[steps["ep"][i]] for i in idx], [steps["hist"][i] for i in idx])
                actions = torch.tensor([steps["action"][i] for i in idx], device=device)
                old_logp = torch.tensor([steps["logp"][i] for i in idx], device=device)
                a = torch.tensor([adv[steps["ep"][i]] for i in idx], device=device, dtype=torch.float32)
                with autocast(device):
                    loss, s = ppo_loss(vla, inputs.to(device), actions, old_logp, a, args.temperature, args.clip)
                if not math.isfinite(loss.item()):
                    nonfinite += 1
                # Weight each micro-batch by its share of the update so the loss is a per-step mean.
                scaler.scale(loss * len(idx) / len(chunk)).backward()
                stats.append(s)
            grad_norm = optimizer_step(vla, opt, scaler, args.max_grad_norm)
            updates += 1
    out = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    out.update(train_steps=len(keep), optimizer_steps=updates, grad_norm=grad_norm, nonfinite_losses=nonfinite)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", required=True, help="SFT checkpoint directory (a VLA saved by train_vla)")
    p.add_argument("--data", default="data/GoToLocal", help="for the level, held-out combos and evaluation")
    p.add_argument("--out", default="runs/vla256_rl/GoToLocal")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seeds-per-iter", type=int, default=8)
    p.add_argument("--group", type=int, default=8, help="sampled episodes per seed")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--warmup", type=int, default=5, help="iterations of linear LR warmup")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=64, help="micro-batches per optimizer step")
    p.add_argument("--eval-every", type=int, default=25, help="iterations; 0 disables")
    p.add_argument("--eval-episodes", type=int, default=100, help="held-out seeds and combos during training")
    p.add_argument("--final-eval-episodes", type=int, default=500)
    p.add_argument("--eval-max-steps", type=int)
    p.add_argument("--save-every", type=int, default=10, help="iterations between resumable checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    meta = load_meta(args.data)
    level, tile = meta["level"], meta["tile_size"]
    heldout_combos = [tuple(c) for c in meta["heldout_combos"]]

    with open(os.path.join(args.init, "vla_config.json")) as f:
        vla = VLA.from_pretrained(args.init, **json.load(f))
    vla.model.to(device)
    trainable = [p for p in vla.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / max(1, args.warmup)))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    init_optimizer_state(opt)

    ckpt_path = os.path.join(args.out, "last.pt")
    start = 1
    if args.resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        vla.model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start = state["iter"] + 1
        print("resumed from %s at iteration %d" % (ckpt_path, start))

    envs = [make_env(level, tile) for _ in range(args.seeds_per_iter * args.group)]
    heldout = {"heldout_seeds": load_split(args.data, "test_seeds"),
               "heldout_combos": load_split(args.data, "test_combos")}
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    t0 = time.time()
    for it in range(start, args.iters + 1):
        rng = np.random.default_rng([args.seed, it])  # per-iteration streams, so resuming replays nothing
        torch.manual_seed(args.seed * 100_003 + it)
        seeds = sample_seeds(level, args.seeds_per_iter, rng, heldout_combos, tile)
        t_roll = time.time()
        reward, length, success, missions, steps = rollout(vla, envs, seeds, args.group, device, args.temperature)
        t_roll = time.time() - t_roll
        adv = group_advantages(reward, args.group)
        stats = update(vla, opt, scaler, missions, steps, adv, args, device, rng)
        sched.step()
        rec = {"iter": it, "reward": float(reward.mean()), "success": float(success.mean()),
               "length": float(length.mean()), "groups_with_signal": int(np.isfinite(adv).sum() // args.group),
               "lr": sched.get_last_lr()[0], "rollout_sec": t_roll, "sec": time.time() - t0, **stats}
        if device == "cuda":
            rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
            rec["loss_scale"] = scaler.get_scale()
        print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n")
        log.flush()

        if args.eval_every and it % args.eval_every == 0 and it < args.iters:
            policy = VLAPolicy(vla, device)
            for name, data in heldout.items():
                r = evaluate_split(policy, level, data.seeds[:args.eval_episodes].tolist(), data.bot_lengths(),
                                   tile_size=tile, max_steps=args.eval_max_steps)
                r["iter"] = it
                print("eval", name, json.dumps(r))
                log.write(json.dumps({"eval_" + name: r}) + "\n")
            log.flush()

        if args.save_every and it % args.save_every == 0 and it < args.iters:
            tmp = ckpt_path + ".tmp"
            torch.save({"iter": it, "model": vla.model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict()}, tmp)
            os.replace(tmp, ckpt_path)

    final = os.path.join(args.out, "final")
    vla.save(final)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("saved %s; evaluating" % final)
    results = evaluate_all(VLAPolicy(vla, device), args.data, args.final_eval_episodes,
                           max_steps=args.eval_max_steps)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
