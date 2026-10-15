"""M4: supervised fine-tuning (behavior cloning) of SmolVLM into a BabyAI VLA.

fp32 master weights + fp16 autocast + GradScaler (Turing GPUs such as the RTX 2070 have no bf16), SDPA attention,
gradient accumulation to an effective batch of micro_batch * grad_accum (default 4 * 16 = 64).

  # SmolVLM-256M, full fine-tune
  python -m toyvla.train_vla --data data/GoToLocal --out runs/vla256/GoToLocal
  # SmolVLM-500M + LoRA
  python -m toyvla.train_vla --model HuggingFaceTB/SmolVLM-500M-Instruct --lora-rank 16 --lr 2e-4 \\
      --out runs/vla500lora/GoToLocal

Glossary:
  micro-batch    the examples that go through the GPU together in one forward/backward pass
  accumulation   summing the gradients of several micro-batches before one weight update,
                 to train with a larger batch than fits in memory at once
  fp16 autocast  run most math in 16-bit floats (half the memory, faster on the GPU) while the
                 weights themselves stay in 32-bit
  GradScaler     multiplies the loss by a large factor before backprop so tiny fp16 gradients
                 don't round to zero, then divides it back out before the update
"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch
# Learning-rate schedule: ramp up linearly during warmup, then decay along a cosine curve to 0.
from transformers import get_cosine_schedule_with_warmup

from .data import load_meta, load_split
from .evaluate import evaluate_all, evaluate_split
from .vla import DEFAULT_MODEL, VLA, VLAPolicy


class StepDataset(torch.utils.data.Dataset):
    """Every (frame, mission, recent actions) -> action pair in a split."""

    def __init__(self, data, history):
        self.data, self.history = data, history

    def __len__(self):
        # One training example per step (not per episode): ~54k for 10k GoToLocal episodes.
        return len(self.data.actions)

    def __getitem__(self, i):
        # The DataLoader calls this with random indices i to assemble each micro-batch.
        d = self.data
        # (frame at step i, its episode's mission, the actions before it, the bot's action)
        return d.frames[i], d.missions[d.step_episode[i]], d.history(i, self.history), int(d.actions[i])


class Collate:
    """Combines a list of examples into one batch of model inputs.

    A class rather than a function so it can be pickled and sent to DataLoader worker processes.
    """

    def __init__(self, encode):
        self.encode = encode  # a PromptEncoder

    def __call__(self, items):
        # zip(*items) turns [(f1, m1, h1, a1), (f2, m2, h2, a2), ...] into
        # (f1, f2, ...), (m1, m2, ...), (h1, h2, ...), (a1, a2, ...).
        frames, missions, histories, actions = zip(*items)
        return self.encode(np.stack(frames), list(missions), list(histories)), torch.tensor(actions)


def autocast(device):
    # A context manager: code inside `with autocast(device):` runs in fp16 on the GPU.
    # On CPU it is disabled and everything stays fp32.
    return torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda")


def forward_backward(vla, inputs, actions, scaler, device, grad_accum=1):
    with autocast(device):
        loss, acc = vla.loss(inputs.to(device), actions.to(device))  # forward pass
    # Divide by grad_accum so that summing gradients over grad_accum micro-batches gives the
    # average, as if it were one big batch. scaler.scale multiplies by the loss scale first;
    # backward() then adds this micro-batch's gradients onto each weight's .grad.
    scaler.scale(loss / grad_accum).backward()
    return loss.item(), acc.item()


@torch.no_grad()
def init_optimizer_state(opt):
    """Allocate AdamW's moment buffers now (2x the trainable weights in fp32) instead of at the first
    applied update. GradScaler usually skips the first updates while it lowers the loss scale, so
    without this the memory jump, and any out-of-memory error, arrives a few steps into training.
    The step runs at lr 0 on zero gradients, so it changes no weights."""
    lrs = [g["lr"] for g in opt.param_groups]  # remember each group's learning rate
    for g in opt.param_groups:
        g["lr"] = 0.0  # a zero learning rate makes this step a no-op on the weights
        for p in g["params"]:
            p.grad = torch.zeros_like(p)  # step() skips parameters without a gradient
    opt.step()  # AdamW creates its moment buffers (exp_avg, exp_avg_sq) on its first step
    for g, lr in zip(opt.param_groups, lrs):
        g["lr"] = lr  # restore
    opt.zero_grad(set_to_none=True)  # drop the dummy gradients, freeing their memory


def optimizer_step(vla, opt, scaler, max_grad_norm=1.0):
    # Divide the accumulated gradients by the loss scale, back to their true size, so that
    # clipping below compares real gradient magnitudes.
    scaler.unscale_(opt)
    params = [p for p in vla.model.parameters() if p.requires_grad]
    # If the gradients' overall L2 norm exceeds max_grad_norm, shrink them all proportionally.
    # Returns the norm before clipping, which we log.
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    scaler.step(opt)  # skipped automatically if fp16 gradients overflowed
    # Adjust the loss scale: halve it after an overflow, grow it after a run of clean steps.
    scaler.update()
    # Reset gradients for the next accumulation cycle; None (instead of zeros) frees the memory.
    opt.zero_grad(set_to_none=True)
    return grad_norm.item()


def save_checkpoint(path, step, vla, opt, sched, scaler):
    """Everything needed to resume: weights, AdamW moments, LR schedule and loss scale (~3 GB for
    the 256M model). Written to a temp file first so a crash mid-write keeps the previous one."""
    tmp = path + ".tmp"
    torch.save({"step": step, "model": vla.model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict()}, tmp)
    # os.replace swaps the new file in with a single rename, so last.pt is never half-written.
    os.replace(tmp, path)


def load_checkpoint(path, vla, opt, sched, scaler, device):
    # weights_only=False: the file holds optimizer and scheduler state, not just tensors.
    state = torch.load(path, map_location=device, weights_only=False)
    vla.model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    scaler.load_state_dict(state["scaler"])
    return state["step"]  # the last completed optimizer step


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default="data/GoToLocal")
    p.add_argument("--out", default="runs/vla256/GoToLocal")
    p.add_argument("--history", type=int, default=4, help="past actions in the prompt")
    p.add_argument("--image-size", type=int, default=512, help="frame resolution fed to the vision encoder")
    p.add_argument("--lora-rank", type=int, default=0, help="0 = full fine-tune")
    p.add_argument("--freeze-vision", action="store_true")
    # Gradient checkpointing: recompute activations during backward instead of storing them;
    # saves memory at the cost of extra compute.
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=2000, help="optimizer steps")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=500, help="optimizer steps; 0 disables")
    p.add_argument("--eval-episodes", type=int, default=100, help="held-out seeds during training")
    p.add_argument("--final-eval-episodes", type=int, default=500, help="per split, after training")
    p.add_argument("--eval-max-steps", type=int, help="cut evaluation episodes at this many steps (failures)")
    p.add_argument("--save-every", type=int, default=100,
                   help="optimizer steps between resumable checkpoints (out/last.pt); 0 disables")
    p.add_argument("--resume", action="store_true", help="continue from out/last.pt if it exists")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)  # reproducible data order
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    # Save the exact settings of this run next to its results.
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Model ---
    vla = VLA.from_pretrained(args.model, history=args.history, image_size=args.image_size,
                              lora_rank=args.lora_rank, freeze_vision=args.freeze_vision)
    vla.model.to(device)  # move the weights onto the GPU
    if args.grad_checkpointing:
        vla.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Only parameters with requires_grad are trained (all of them for a full fine-tune; just
    # the LoRA adapters with --lora-rank).
    trainable = [p for p in vla.model.parameters() if p.requires_grad]
    print("trainable params: %.1fM / %.1fM; %d image tokens per frame"
          % (sum(p.numel() for p in trainable) / 1e6, sum(p.numel() for p in vla.model.parameters()) / 1e6,
             vla.processor.image_seq_len))

    # --- Data ---
    train = load_split(args.data, "train")
    # The DataLoader shuffles the steps, groups them into micro-batches, and runs Collate
    # (image resizing, tokenizing) in `workers` background processes while the GPU trains.
    # drop_last skips the final incomplete batch; persistent_workers keeps the workers alive
    # between epochs.
    loader = torch.utils.data.DataLoader(
        StepDataset(train, args.history), batch_size=args.micro_batch, shuffle=True, drop_last=True,
        num_workers=args.workers, collate_fn=Collate(vla.encode), persistent_workers=args.workers > 0)
    # An endless stream of batches: when one pass (epoch) ends, start another, reshuffled.
    batches = itertools.chain.from_iterable(itertools.repeat(loader))
    print("train: %d steps; %.1f epochs" % (len(train.actions),
                                            args.max_steps * args.micro_batch * args.grad_accum / len(train.actions)))

    # --- Optimizer ---
    # AdamW: Adam with decoupled weight decay; keeps two running averages per weight.
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.max_steps)
    # Loss scaling is only needed for fp16 on the GPU; on CPU the scaler does nothing.
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    init_optimizer_state(opt)
    heldout = load_split(args.data, "test_seeds")  # for the periodic success-rate check
    meta = load_meta(args.data)
    log = open(os.path.join(args.out, "log.jsonl"), "a")  # "a" = append: a resumed run keeps the old lines
    ckpt_path = os.path.join(args.out, "last.pt")
    start = 1
    if args.resume and os.path.exists(ckpt_path):
        start = load_checkpoint(ckpt_path, vla, opt, sched, scaler, device) + 1
        print("resumed from %s at step %d" % (ckpt_path, start))
        torch.manual_seed(args.seed + start)  # fresh data order for the remaining steps

    # --- Training loop ---
    vla.model.train()  # training mode: enables dropout
    t0, losses, accs, nonfinite = time.time(), [], [], 0
    for step in range(start, args.max_steps + 1):  # one iteration = one weight update
        for _ in range(args.grad_accum):  # accumulate gradients over grad_accum micro-batches
            inputs, actions = next(batches)
            loss, acc = forward_backward(vla, inputs, actions, scaler, device, args.grad_accum)
            if not math.isfinite(loss):  # NaN/inf loss = an fp16 overflow in the forward pass
                nonfinite += 1
            losses.append(loss)
            accs.append(acc)
        grad_norm = optimizer_step(vla, opt, scaler)  # apply the update
        sched.step()  # move the learning rate along its schedule

        if step % args.log_every == 0:
            # Averages over the micro-batches since the last log line. nanmean ignores
            # any NaN losses so one overflow doesn't hide the trend.
            rec = {"step": step, "loss": float(np.nanmean(losses)), "action_acc": float(np.mean(accs)),
                   "grad_norm": grad_norm, "lr": sched.get_last_lr()[0], "nonfinite_losses": nonfinite,
                   "sec": time.time() - t0}
            if device == "cuda":
                rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30  # bytes -> GiB
                rec["loss_scale"] = scaler.get_scale()  # should settle; repeated halving signals trouble
            print(json.dumps(rec))
            log.write(json.dumps(rec) + "\n")
            log.flush()  # write to disk now, so the log survives a crash
            losses, accs = [], []

        # Every eval_every updates, roll the current model out on held-out seeds.
        if args.eval_every and step % args.eval_every == 0 and step < args.max_steps:
            r = evaluate_split(VLAPolicy(vla, device), meta["level"],
                               heldout.seeds[:args.eval_episodes].tolist(), heldout.bot_lengths(),
                               tile_size=meta["tile_size"], max_steps=args.eval_max_steps)
            r["step"] = step
            print("eval heldout_seeds", json.dumps(r))
            log.write(json.dumps({"eval_heldout_seeds": r}) + "\n")
            log.flush()
            vla.model.train()  # VLAPolicy switched the model to eval mode; switch back

        if args.save_every and step % args.save_every == 0 and step < args.max_steps:
            save_checkpoint(ckpt_path, step, vla, opt, sched, scaler)

    # --- Save and evaluate ---
    final = os.path.join(args.out, "final")
    vla.save(final)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # the final model supersedes it
    print("saved %s; evaluating" % final)
    if args.lora_rank:  # save() merged the adapters; reload the plain model for evaluation
        policy = VLAPolicy.load(final, device)
    else:
        policy = VLAPolicy(vla, device)
    results = evaluate_all(policy, args.data, args.final_eval_episodes, max_steps=args.eval_max_steps)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
