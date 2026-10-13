"""M4: supervised fine-tuning (behavior cloning) of SmolVLM into a BabyAI VLA.

fp32 master weights + fp16 autocast + GradScaler (the RTX 2080 has no bf16), SDPA attention,
gradient accumulation to an effective batch of micro_batch * grad_accum.

  # SmolVLM-256M, full fine-tune
  python -m toyvla.train_vla --data data/GoToLocal --out runs/vla256/GoToLocal
  # SmolVLM-500M + LoRA
  python -m toyvla.train_vla --model HuggingFaceTB/SmolVLM-500M-Instruct --lora-rank 16 --lr 2e-4 \\
      --out runs/vla500lora/GoToLocal
"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from .data import load_split
from .evaluate import evaluate_all, evaluate_split
from .vla import DEFAULT_MODEL, VLA, VLAPolicy


class StepDataset(torch.utils.data.Dataset):
    """Every (frame, mission, recent actions) -> action pair in a split."""

    def __init__(self, data, history):
        self.data, self.history = data, history

    def __len__(self):
        return len(self.data.actions)

    def __getitem__(self, i):
        d = self.data
        return d.frames[i], d.missions[d.step_episode[i]], d.history(i, self.history), int(d.actions[i])


class Collate:
    def __init__(self, encode):
        self.encode = encode

    def __call__(self, items):
        frames, missions, histories, actions = zip(*items)
        return self.encode(np.stack(frames), list(missions), list(histories)), torch.tensor(actions)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda")


def forward_backward(vla, inputs, actions, scaler, device, grad_accum=1):
    with autocast(device):
        loss, acc = vla.loss(inputs.to(device), actions.to(device))
    scaler.scale(loss / grad_accum).backward()
    return loss.item(), acc.item()


def optimizer_step(vla, opt, scaler, max_grad_norm=1.0):
    scaler.unscale_(opt)
    params = [p for p in vla.model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    scaler.step(opt)  # skipped automatically if fp16 gradients overflowed
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return grad_norm.item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default="data/GoToLocal")
    p.add_argument("--out", default="runs/vla256/GoToLocal")
    p.add_argument("--history", type=int, default=4, help="past actions in the prompt")
    p.add_argument("--image-size", type=int, default=512, help="frame resolution fed to the vision encoder")
    p.add_argument("--lora-rank", type=int, default=0, help="0 = full fine-tune")
    p.add_argument("--freeze-vision", action="store_true")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=2000, help="optimizer steps")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=500, help="optimizer steps; 0 disables")
    p.add_argument("--eval-episodes", type=int, default=100, help="held-out seeds during training")
    p.add_argument("--final-eval-episodes", type=int, default=500, help="per split, after training")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    vla = VLA.from_pretrained(args.model, history=args.history, image_size=args.image_size,
                              lora_rank=args.lora_rank, freeze_vision=args.freeze_vision)
    vla.model.to(device)
    if args.grad_checkpointing:
        vla.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in vla.model.parameters() if p.requires_grad]
    print("trainable params: %.1fM / %.1fM; %d image tokens per frame"
          % (sum(p.numel() for p in trainable) / 1e6, sum(p.numel() for p in vla.model.parameters()) / 1e6,
             vla.processor.image_seq_len))

    train = load_split(args.data, "train")
    loader = torch.utils.data.DataLoader(
        StepDataset(train, args.history), batch_size=args.micro_batch, shuffle=True, drop_last=True,
        num_workers=args.workers, collate_fn=Collate(vla.encode), persistent_workers=args.workers > 0)
    batches = itertools.chain.from_iterable(itertools.repeat(loader))
    print("train: %d steps; %.1f epochs" % (len(train.actions),
                                            args.max_steps * args.micro_batch * args.grad_accum / len(train.actions)))

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    heldout = load_split(args.data, "test_seeds")
    meta_level = json.load(open(os.path.join(args.data, "meta.json")))
    log = open(os.path.join(args.out, "log.jsonl"), "a")

    vla.model.train()
    t0, losses, accs, nonfinite = time.time(), [], [], 0
    for step in range(1, args.max_steps + 1):
        for _ in range(args.grad_accum):
            inputs, actions = next(batches)
            loss, acc = forward_backward(vla, inputs, actions, scaler, device, args.grad_accum)
            if not math.isfinite(loss):
                nonfinite += 1
            losses.append(loss)
            accs.append(acc)
        grad_norm = optimizer_step(vla, opt, scaler)
        sched.step()

        if step % args.log_every == 0:
            rec = {"step": step, "loss": float(np.nanmean(losses)), "action_acc": float(np.mean(accs)),
                   "grad_norm": grad_norm, "lr": sched.get_last_lr()[0], "nonfinite_losses": nonfinite,
                   "sec": time.time() - t0}
            if device == "cuda":
                rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
                rec["loss_scale"] = scaler.get_scale()
            print(json.dumps(rec))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            losses, accs = [], []

        if args.eval_every and step % args.eval_every == 0 and step < args.max_steps:
            r = evaluate_split(VLAPolicy(vla, device), meta_level["level"],
                               heldout.seeds[:args.eval_episodes].tolist(), heldout.bot_lengths(),
                               tile_size=meta_level["tile_size"])
            r["step"] = step
            print("eval heldout_seeds", json.dumps(r))
            log.write(json.dumps({"eval_heldout_seeds": r}) + "\n")
            vla.model.train()

    final = os.path.join(args.out, "final")
    vla.save(final)
    print("saved %s; evaluating" % final)
    if args.lora_rank:  # save() merged the adapters; reload the plain model for evaluation
        policy = VLAPolicy.load(final, device)
    else:
        policy = VLAPolicy(vla, device)
    results = evaluate_all(policy, args.data, args.final_eval_episodes)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
