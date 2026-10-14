"""M1: check the GPU and load SmolVLM, then record peak VRAM per training step at growing micro-batch sizes.

Uses exactly the training setup of train_vla (fp32 master weights, fp16 autocast, GradScaler,
SDPA attention), with random BabyAI frames, and stops at the first out-of-memory batch size.

  python -m toyvla.setup_check
  python -m toyvla.setup_check --model HuggingFaceTB/SmolVLM-500M-Instruct --lora-rank 16
"""
import argparse
import json
import os

import numpy as np
import torch

from .common import NUM_ACTIONS, make_env, quiet_reset
from .train_vla import forward_backward, optimizer_step
from .vla import DEFAULT_MODEL, VLA


def sample_inputs(n, level="GoToLocal", history=4, seed=0):
    env = make_env(level)
    rng = np.random.default_rng(seed)
    frames, missions, histories = [], [], []
    for i in range(n):
        obs, _ = quiet_reset(env, seed + i)
        frames.append(obs["image"])
        missions.append(obs["mission"])
        histories.append(rng.integers(0, NUM_ACTIONS, history).tolist())
    return np.stack(frames), missions, histories, torch.from_numpy(rng.integers(0, NUM_ACTIONS, n))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument("--freeze-vision", action="store_true")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    p.add_argument("--out", help="write the measurements as JSON here")
    args = p.parse_args()
    if args.out:  # create the output directory up front, before any slow work
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device found; M1 measures GPU memory.")
    props = torch.cuda.get_device_properties(0)
    print("GPU: %s, %.1f GB, compute capability %d.%d, bf16 supported: %s"
          % (props.name, props.total_memory / 2**30, props.major, props.minor, torch.cuda.is_bf16_supported()))

    vla = VLA.from_pretrained(args.model, history=args.history, image_size=args.image_size,
                              lora_rank=args.lora_rank, freeze_vision=args.freeze_vision)
    vla.model.to("cuda").train()
    if args.grad_checkpointing:
        vla.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW([p for p in vla.model.parameters() if p.requires_grad], lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    weights_gb = torch.cuda.memory_allocated() / 2**30
    print("model loaded: %.2f GB of weights; %d image tokens per frame" % (weights_gb, vla.processor.image_seq_len))

    results = {"gpu": props.name, "total_gb": props.total_memory / 2**30, "weights_gb": weights_gb,
               "config": vars(args), "steps": []}
    for bs in map(int, args.batch_sizes.split(",")):
        frames, missions, histories, actions = sample_inputs(bs, history=args.history)
        inputs = vla.encode(frames, missions, histories)
        torch.cuda.reset_peak_memory_stats()
        try:
            loss, _ = forward_backward(vla, inputs, actions, scaler, "cuda")
            optimizer_step(vla, opt, scaler)
        except torch.cuda.OutOfMemoryError:
            print("batch %2d: out of memory" % bs)
            results["steps"].append({"batch": bs, "oom": True})
            break
        peak = torch.cuda.max_memory_allocated() / 2**30
        print("batch %2d: seq len %d, loss %.3f, peak VRAM %.2f GB" % (bs, inputs["input_ids"].shape[1], loss, peak))
        results["steps"].append({"batch": bs, "seq_len": inputs["input_ids"].shape[1], "loss": loss, "peak_gb": peak})
        opt.zero_grad(set_to_none=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
