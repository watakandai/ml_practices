# Toy VLA on BabyAI

This project fine-tunes a small pretrained vision-language model (SmolVLM) into a
vision-language-action (VLA) policy for BabyAI. The policy reads the egocentric RGB frame,
the mission string and its recent actions, then emits one of MiniGrid's 7 action words.
The recipe follows RT-2 and OpenVLA at toy scale and is sized for a single RTX 2070
(8 GB, no bf16, no FlashAttention-2).

It implements the project plan *Toy VLA on BabyAI — Project Plan*. Milestones and status:

| Milestone | Code | Status |
|---|---|---|
| M1 Setup check | `toyvla/setup_check.py` | Done on the RTX 2070: SmolVLM-256M full fine-tune peaks at 5.2 GB at micro-batch 8; 16 runs out of memory |
| M2 Data | `toyvla/collect_demos.py`, `toyvla/data.py` | Done (GoToLocal, 10k episodes) |
| M3 Baseline | `toyvla/baseline.py` | Done; results below (CPU run) |
| M4 VLA SFT | `toyvla/vla.py`, `toyvla/train_vla.py` | Written and tested on a tiny random Idefics3; needs the GPU and SmolVLM weights |
| M5 Ablations | `--history`, `--image-size` flags on `train_vla` | Knobs in place |
| M6 Harder levels | `--level PickupLoc` / `GoToSeq` on every script | Knobs in place |
| M7 RL fine-tuning | — | Not started |

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
cd vla_babyai
uv sync                                                    # creates .venv; torch from PyTorch's cu126 index
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"  # expect True 12.6
uv run pytest                                              # ~1 min; builds a tiny random SmolVLM-architecture model offline
git add uv.lock                                            # commit the lock produced on the GPU machine
```

`pyproject.toml` pulls torch and torchvision from the CUDA 12.6 wheel index on Linux. The
default PyPI torch is a CUDA 13 build, and the 2070's driver 535 (CUDA 12.2) cannot run CUDA 13.
CUDA 12.x wheels do run on it, through minor-version compatibility. After a driver upgrade to
570 or newer, switching the index URL to `cu128` also works.

Run every command from `vla_babyai/`, prefixed with `uv run` (e.g. `uv run python -m toyvla.collect_demos`). Data goes to `data/` and checkpoints to `runs/`;
both are git-ignored.

## Pipeline

Commands below are shown without the `uv run` prefix.

```bash
# M1: GPU info, then peak VRAM of one real training step at micro-batch 1, 2, 4, ... until OOM
python -m toyvla.setup_check --out runs/m1_256m.json
python -m toyvla.setup_check --model HuggingFaceTB/SmolVLM-500M-Instruct --lora-rank 16 --out runs/m1_500m_lora.json

# M2: bot demos + held-out splits (~1 min for 10k GoToLocal episodes)
python -m toyvla.collect_demos --level GoToLocal --episodes 10000

# M3: from-scratch baseline (trains, then evaluates all three splits)
python -m toyvla.baseline --data data/GoToLocal --out runs/baseline/GoToLocal

# M4: VLA SFT. SmolVLM-256M full fine-tune, then SmolVLM-500M + LoRA
python -m toyvla.train_vla --data data/GoToLocal --out runs/vla256/GoToLocal
python -m toyvla.train_vla --model HuggingFaceTB/SmolVLM-500M-Instruct --lora-rank 16 --lr 2e-4 \
    --data data/GoToLocal --out runs/vla500lora/GoToLocal

# Re-evaluate any checkpoint
python -m toyvla.evaluate --policy vla --ckpt runs/vla256/GoToLocal/final --data data/GoToLocal
python -m toyvla.evaluate --policy baseline --ckpt runs/baseline/GoToLocal/model.pt --data data/GoToLocal

# M5 ablations, e.g.
python -m toyvla.train_vla --history 0 --out runs/vla256_h0/GoToLocal
python -m toyvla.train_vla --image-size 256 --out runs/vla256_r256/GoToLocal
```

## Design notes

**Data (M2).** The data comes from the BabyAI bot (`minigrid.utils.baby_ai_bot.BabyAIBot`)
on `RGBImgPartialObsWrapper` observations: the agent's 7×7 view at 8 px per cell, so 56×56 RGB.
Only successful bot episodes are kept. There are three splits:

- `train`: seeds from 0 upward. Missions naming a held-out color–object combo are skipped.
- `test_seeds`: seeds from 10,000,000 upward, using training combos only.
- `test_combos`: seeds from that same test range whose mission names a held-out combo.
  The defaults are red ball, green key, blue box and yellow ball (`--heldout-combos`).
  These objects can still appear as distractors in training scenes; only instructions
  that name them are held out.

Each split stores frames, actions, episode boundaries, seeds and missions. A seed fully
determines the level, so evaluation re-creates each episode from its seed. The stored bot
episode length is the reference for the length metric.

**Evaluation.** Each table column is a success rate on one split: training seeds ("seen"),
`test_seeds` and `test_combos`. `length_ratio` is the mean of policy steps ÷ bot steps over
successful episodes; a value above 1 means the policy wanders. Environments run in parallel
batches, so the VLA forward pass is batched too.

**Baseline (M3).** The model follows the BabyAI paper: a GRU instruction encoder, a conv stem,
two FiLM-conditioned residual conv blocks, max-pooling, a GRU memory over time and an MLP
actor. It is trained with behavior cloning on whole episodes and sees the same RGB frames as
the VLA.

**VLA (M4)** (`toyvla/vla.py`):

- **Prompt.** SmolVLM's chat template, holding the frame, `Mission: …`, the last
  `--history` actions as words (`Recent actions: left forward`) and the list of valid
  actions. It ends at `Assistant:`.
- **Action head.** The language-model head is the action head. The target is the first
  token of ` <action word>`. The code checks at load time that the 7 first tokens are
  distinct.
- **Loss.** Cross-entropy over the full vocabulary, at the last prompt position only.
  Left padding puts that position at index −1 for every row, and the model returns
  logits only there (`logits_to_keep=1`).
- **Inference.** Argmax over the 7 action-token logits only, i.e. constrained decoding.
  These 7 logits are also the categorical policy that M7's PPO/GRPO will need.
- **Images.** No image splitting: each frame is upscaled with nearest neighbour to
  `--image-size` (default 512, giving 64 image tokens for SmolVLM-256M) and encoded as
  one tile. `--image-size 256` gives 16 tokens. That is the main lever on sequence length.
- **Precision and memory.**
  - fp32 master weights, fp16 autocast and `GradScaler` (Turing has no bf16).
  - SDPA attention.
  - Micro-batch 8 × 8 accumulation steps = an effective batch of 64.
  - `--grad-checkpointing` and `--freeze-vision` when memory is tight.
  - The training log records the count of non-finite losses and the loss scale, to catch
    fp16 overflow.
- **LoRA.** `--lora-rank r` puts adapters on every attention and MLP projection. At the end
  they are merged into the weights, so a checkpoint always loads as a plain model.

## Results — BabyAI-GoToLocal

Success rate (episode length ÷ bot length) over 500 episodes per split.

| Policy | Seen seeds | Held-out seeds | Held-out color–object combos |
|---|---|---|---|
| BabyAI CNN+GRU+FiLM, from scratch | 0.998 (1.05) | 0.998 (1.06) | 0.536 (3.15) |
| SmolVLM-256M, SFT | | | |
| SmolVLM-500M + LoRA, SFT | | | |
| Best SFT model + RL | | | |

The baseline row comes from a CPU run: 15 epochs, 64 channels, 10k demos, about 45 minutes, final
training action accuracy 0.989. It generalizes across seeds but not to unseen instruction
combinations. On held-out combos it succeeds about half the time and takes about 3× the bot's
steps when it does. That gap is what the VLA's language pretraining should close.
