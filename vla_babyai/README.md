# Toy VLA on BabyAI

This project fine-tunes a small pretrained vision-language model (SmolVLM) into a
vision-language-action (VLA) policy for BabyAI. The policy reads the egocentric RGB frame,
the mission string and its recent actions, then emits one of MiniGrid's 7 action words.
The recipe follows RT-2 and OpenVLA at toy scale and is sized for a single RTX 2070
(8 GB, no bf16, no FlashAttention-2).

It implements the project plan *Toy VLA on BabyAI — Project Plan*. Milestones and status:

| Milestone | Code | Status |
|---|---|---|
| M1 Setup check | `toyvla/setup_check.py` | Done on the RTX 2070: full fine-tune of SmolVLM-256M at micro-batch 4 peaks at 6.2 GB |
| M2 Data | `toyvla/collect_demos.py`, `toyvla/data.py` | Done (GoToLocal, 10k episodes) |
| M3 Baseline | `toyvla/baseline.py` | Done; results below (CPU run) |
| M4 VLA SFT | `toyvla/vla.py`, `toyvla/train_vla.py` | SmolVLM-256M done on the RTX 2070; results below |
| M5 Ablations | `--history`, `--image-size` flags on `train_vla` | Done: history 0 and image size 256; results below |
| M6 Harder levels | `--level PickupLoc` / `GoToSeq` on every script | Done: baseline and SmolVLM-256M SFT on both; results below |
| M7 RL fine-tuning | `toyvla/rl.py` | Done: group-relative PPO from the 256M SFT model; results below |

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

# M5 ablations
python -m toyvla.train_vla --history 0 --out runs/vla256_h0/GoToLocal
python -m toyvla.train_vla --image-size 256 --out runs/vla256_r256/GoToLocal

# M6 harder levels. GoToSeq episodes are ~67 bot steps, so 4k demos (~2.5 GB of frames) instead of 10k,
# and evaluation is cut at 512 steps (the level allows 2304) over 200 episodes per split.
python -m toyvla.collect_demos --level PickupLoc --episodes 10000
python -m toyvla.collect_demos --level GoToSeq --episodes 4000
python -m toyvla.baseline --data data/PickupLoc --out runs/baseline/PickupLoc --epochs 15 --eval-episodes 500
python -m toyvla.baseline --data data/GoToSeq --out runs/baseline/GoToSeq --epochs 15 --batch-size 16 \
    --eval-episodes 200 --eval-max-steps 512
python -m toyvla.train_vla --data data/PickupLoc --out runs/vla256/PickupLoc
python -m toyvla.train_vla --data data/GoToSeq --out runs/vla256/GoToSeq --final-eval-episodes 200 --eval-max-steps 512

# M7: RL fine-tuning of the SFT model
python -m toyvla.rl --init runs/vla256/GoToLocal/final --data data/GoToLocal --out runs/vla256_rl/GoToLocal
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
  These 7 logits are also the categorical policy that M7's RL fine-tunes.
- **Images.** No image splitting: each frame is upscaled with nearest neighbour to
  `--image-size` (default 512, giving 64 image tokens for SmolVLM-256M) and encoded as
  one tile. `--image-size 256` gives 16 tokens. That is the main lever on sequence length.
- **Precision and memory.**
  - fp32 master weights, fp16 autocast and `GradScaler` (Turing has no bf16).
  - SDPA attention.
  - Micro-batch 4 × 16 accumulation steps = an effective batch of 64.
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
| SmolVLM-256M, SFT | 0.992 (1.05) | 0.992 (1.06) | 0.652 (4.96) |
| SmolVLM-500M + LoRA, SFT | 0.980 (1.26) | 0.962 (1.28) | 0.680 (4.45) |
| SmolVLM-256M SFT + RL (M7) | 0.994 (1.00) | 0.990 (1.00) | 0.626 (4.18) |

The baseline row comes from a CPU run: 15 epochs, 64 channels, 10k demos, about 45 minutes, final
training action accuracy 0.989. It generalizes across seeds but not to unseen instruction
combinations. On held-out combos it succeeds about half the time and takes about 3× the bot's
steps when it does. That gap is what the VLA's language pretraining should close.

The SmolVLM-256M row: 2000 updates at an effective batch of 64 (about 2.4 epochs), on the
RTX 2070 in 2 h 7 min, peak VRAM 6.2 GB, with no fp16 overflow. Final training loss was 0.058
and action accuracy 0.983. On held-out combos it succeeds more often than the baseline
(65.2% vs 53.6%; about ±2 points of standard error at 500 episodes). But it takes about 5× the
bot's steps when it does, so part of that success comes from wandering until it happens to
face the right object rather than heading straight for it. Both models are at about 99% on
seeds they did not train on.

The SmolVLM-500M + LoRA row: rank 16, lr 2e-4, otherwise the same settings. Its 68.0% on held-out
combos is not clearly better than 256M's 65.2%: the gap is about one standard error of the
difference (±3 points). It is clearly worse on seeds it trained on or that are held out (about
97% vs 99%) and takes about 1.3× the bot's steps there instead of 1.05×. So LoRA at this rank
and learning rate fits the training distribution less closely than full fine-tuning of the
smaller model.

## M5: ablations on GoToLocal

SmolVLM-256M, full fine-tune, same settings as the SFT row above (2000 updates, effective batch
64), changing one knob at a time. 500 episodes per split.

| Variant | Final train loss / action acc | Seen seeds | Held-out seeds | Held-out combos | Train time |
|---|---|---|---|---|---|
| History 4, 512 px (64 image tokens), the SFT row above | 0.058 / 0.983 | 0.992 (1.05) | 0.992 (1.06) | 0.652 (4.96) | 2 h 7 min |
| History 0 | 0.289 / 0.895 | 0.858 (1.16) | 0.872 (1.12) | 0.064 (1.47) | 2 h 21 min |
| Image size 256 (16 image tokens) | 0.414 / 0.808 | 0.714 (2.93) | 0.726 (2.88) | 0.434 (3.95) | 1 h 28 min |

Both knobs matter, and history matters most for generalization.

- **History 0.** With no past actions, held-out combo success falls from 65% to 6%. The policy
  sees only the current frame, so it cannot tell that it has just turned past something or has
  been spinning in place. A likely reading, not yet checked with rollouts: on an unseen combo it
  falls into a loop and times out. The short length ratio (1.47) on the few successes says that it only succeeds when it is
  already nearly facing the target. Even on training seeds it loses 13 points. The bot's next
  action is not a function of the frame alone, and training accuracy stops at 0.90.
- **Image size 256.** Training accuracy stops at 0.81, and episodes run about 3× the bot's
  length even on seen seeds. The vision encoder still sees each cell across several 16 px
  patches, but the connector's pixel shuffle then packs the 7×7 view into a 4×4 grid of tokens,
  so each token mixes about two cells. At 512 px it is 8×8, finer than one token per cell, which
  is the likely reason 64 tokens work and 16 do not. The 256 px run is only 1.4× faster
  per step, because the language model and the fixed per-step overheads dominate.

## M6: harder levels

`PickupLoc` ("pick up the red ball") has the same 8×8 single room as GoToLocal, but the agent
must face the object and then pick it up. `GoToSeq` chains two to four GoTo instructions with
"and", "then" and "after you" across a 3×3 grid of rooms with doors (22×22 cells); the bot needs
about 67 steps (90 on held-out combos). GoToSeq uses 4k demos and a 512-step evaluation cutoff
over 200 episodes per split (see the commands above). Held-out combos are the same four; a
GoToSeq mission is held out if any of its parts names one.

| Level | Policy | Seen seeds | Held-out seeds | Held-out combos |
|---|---|---|---|---|
| PickupLoc | BabyAI CNN+GRU+FiLM, from scratch | 0.966 (1.05) | 0.942 (1.03) | 0.290 (1.68) |
| PickupLoc | SmolVLM-256M, SFT | 0.976 (1.05) | 0.974 (1.09) | 0.962 (1.63) |
| GoToSeq | BabyAI CNN+GRU+FiLM, from scratch | 0.600 (3.34) | 0.550 (3.65) | 0.265 (4.17) |
| GoToSeq | SmolVLM-256M, SFT | 0.450 (8.03) | 0.420 (3.90) | 0.220 (4.78) |

The baselines ran on the CPU (15 epochs; PickupLoc 44 min, GoToSeq 3.5 h at 16 episodes per
batch to fit in RAM).

The VLA rows use the GoToLocal SFT settings (2000 updates at an effective batch of 64): PickupLoc
took 2 h 9 min (final training loss 0.046, action accuracy 0.984), GoToSeq 2 h 34 min (0.26,
0.90). Peak VRAM was 6.2–6.3 GB, with no fp16 overflow.

- **PickupLoc: the VLA generalizes to unseen combos almost fully.** It succeeds on 96% of held-out
  combos, against the baseline's 29% and its own 65% on GoToLocal. Standard error at 500
  episodes is under 1 point for the VLA and about 2 for the baseline, so the gap is real. Picking
  up a wrong object does not end the episode here, so the length ratio of 1.6 likely includes
  some drop-and-retry. Why pickup grounds unseen combos so much better than GoTo is not yet
  diagnosed; `toyvla.diagnose` only handles GoTo levels.
- **GoToSeq: both policies fail often, and the VLA does worse in-distribution.** The VLA succeeds
  on 45% of seen seeds against the baseline's 60%, and on 42% of held-out seeds against 55%
  (about ±5 points of standard error at 200 episodes). On held-out combos the two are level
  (22% vs 26.5%). A GoToSeq mission needs memory of which sub-goals are done. The baseline has a
  GRU over the whole episode; the VLA sees only the current frame and its last 4 actions, so it
  cannot know where it is in the sequence. Its training action accuracy stops at 0.90, as does
  the history-0 ablation's on GoToLocal, which also lacks information it needs. A longer action
  history, past frames in the prompt, or a sub-goal counter would be the next things to try.

The length ratios here are biased low for the weaker policies, because the 512-step cutoff turns
the longest wandering episodes into failures.

## M7: RL fine-tuning

`toyvla/rl.py` fine-tunes the SmolVLM-256M SFT model with a group-relative PPO (GRPO without
its KL term):

- **Policy.** The softmax over the 7 action-token logits, the same distribution that SFT trains
  and greedy evaluation takes the argmax of. Rollouts sample from it at temperature 1.
- **Levels.** Each iteration draws 8 fresh seeds from a range that neither the demos nor the
  test splits use (from 1,000,000), skipping missions that name a held-out combo. RL therefore
  never sees the held-out instructions either.
- **Reward and advantage.** BabyAI's own reward, 1 − 0.9 × steps / max_steps on success and 0 on
  failure, so a shorter success scores higher. 8 episodes per seed; advantage = (reward − group
  mean) / group std. Seeds where all 8 episodes got the same reward are dropped.
- **Update.** 2 epochs of the PPO clipped objective (clip 0.2) over every step of the kept
  episodes, averaged per step, with the rollout log-probabilities as the old policy. lr 2e-6,
  256 steps per optimizer update, fp16 autocast with `GradScaler` as in SFT.

The run: 100 iterations (6,400 episodes) in 45 min on the RTX 2070, peak VRAM 6.2 GB, no fp16
overflow. Results are the "SFT + RL" row of the GoToLocal table.

- **On training-distribution levels RL did what the reward asks.** Episodes got shorter: the
  length ratio on seen and held-out seeds went from 1.05 to 1.00, so the policy now matches the
  bot's step count, while success stayed at 99%.
- **On held-out combos it did nothing.** Success went from 65.2% to 62.6%, inside the noise
  (about ±3 points for the difference), and the length ratio stayed at about 4.2. The training
  signal was thin: the SFT model already succeeds on almost every training mission, so only
  about 2 of the 8 seeds per iteration had any reward spread, all of it about path length. RL on
  the training instructions shortens paths there, but it does not teach the policy anything
  about grounding instructions it has never seen.

To move held-out combos, RL would need either a weaker starting point with room to improve on
training missions, or rollouts on the held-out instructions themselves, which would end the
held-out evaluation.

## Rollouts

`toyvla.render` writes one GIF per episode. Each shows the top-down level, with the agent's
field of view highlighted, next to the 56×56 frame the policy actually receives. The mission
is at the top, the chosen action at the bottom, and the last frame states the outcome and the
bot's step count for the same seed.

```bash
python -m toyvla.render --policy vla --ckpt runs/vla256/GoToLocal/final --split heldout_combos --episodes 4
python -m toyvla.render --policy vla --ckpt runs/vla256/GoToLocal/final --split heldout_combos --select fail
```

The baseline on held-out color–object combos (GIFs written to `media/<policy>_<level>/`):

| "go to a blue box": success, 7 steps vs the bot's 3 | A failure: timeout after 64 steps |
|---|---|
| ![baseline success](media/baseline_GoToLocal/heldout_combos_seed10000004_success.gif) | ![baseline failure](media/baseline_GoToLocal/heldout_combos_seed10000000_fail.gif) |

## Diagnosing the wandering

`toyvla.diagnose` replays each evaluation episode and records every object the agent faces.
A GoTo episode ends in success the moment the agent faces a matching object; facing any other
object does nothing. Each wrong object faced is classed by its relation to the instruction:
same color, same type, or neither. `--policy bot` replays the recorded demos as the reference.
Even the bot faces a non-target object before its target in about 30% of episodes, because it
turns past objects on the way.

```bash
python -m toyvla.diagnose --policy bot
python -m toyvla.diagnose --policy baseline --ckpt runs/baseline/GoToLocal/model.pt
python -m toyvla.diagnose --policy vla --ckpt runs/vla256/GoToLocal/final
python -m toyvla.diagnose --policy vla --ckpt runs/vla256/GoToLocal/final --shuffle-missions
```

Results on held-out color–object combos, 500 episodes. "Solved directly" is the success
rate times the share of successes with no wrong object faced first.

| | Bot | Baseline | SmolVLM-256M |
|---|---|---|---|
| Success | 100% | 54% | 65% |
| Successes with no wrong object faced first | 68% | 49% | 27% |
| Episodes solved directly | 68% | 26% | 18% |
| Wrong objects faced per success | 0.41 | 0.94 | 1.94 |
| Wrong objects faced per failure | — | 3.19 | 6.82 |
| Failures that never faced an object | — | 20% | 5% |
| Success, target in view at start / hidden | 100% / 100% | 58% / 49% | 68% / 62% |
| Wrong objects with the target's color / type | 10% / 35% | 12% / 41% | 24% / 26% |

On held-out seeds both models look like the bot: about 68–69% of successes are direct, with
about 0.4 wrong objects per success.

The VLA's higher success on unseen instructions comes from persistence, not better
grounding. It solves fewer episodes directly than the baseline (18% vs 26%). Instead it keeps
visiting objects until it faces a match, while the baseline more often gets stuck. The
wrong objects it visits share the instruction's color twice as often as the bot's (24% vs
10%), so it picks up the color word but not the object type.

**Language ablation** (`--shuffle-missions`). Each policy is given another episode's
mission, while the environment still rewards the true one. The table shows success on
episodes whose given mission names a different target:

| | True mission | Another target's mission |
|---|---|---|
| Baseline, held-out seeds | 100% | 25% |
| Baseline, held-out combos | 54% | 36% |
| SmolVLM-256M, held-out seeds | 99% | |
| SmolVLM-256M, held-out combos | 65% | |

On held-out seeds, whose instructions use combinations seen in training, the baseline clearly
follows the instruction. On held-out combos,
36 of its 54 points survive a wrong instruction, so they come from roaming into the target
rather than from reading the mission.

Success on held-out combos by target:

| Held-out combo | Episodes | Baseline success | SmolVLM-256M success |
|---|---|---|---|
| yellow ball | 127 | 89% | 59% |
| blue box | 126 | 67% | 79% |
| green key | 116 | 29% | 67% |
| red ball | 131 | 28% | 56% |

