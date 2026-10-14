"""toyvla: fine-tuning a small vision-language model into a BabyAI policy.

Run modules from the vla_babyai/ folder, e.g. `python -m toyvla.collect_demos`. In pipeline order:
  common         environment construction, action names, held-out combos
  collect_demos  M2: record bot demonstrations
  data           load the recorded demonstrations
  baseline       M3: CNN + GRU + FiLM policy trained from scratch
  vla            the SmolVLM-based policy: prompts, loss, constrained decoding
  train_vla      M4: fine-tune the VLA on the demonstrations
  setup_check    M1: measure GPU memory for one training step
  evaluate       success rate and episode length on the three splits
  diagnose       how policies succeed or fail (wandering vs. heading for the target)
  render         GIFs of rollouts
"""
import os

# Lets the CUDA caching allocator grow segments instead of fragmenting the 8 GB card.
# Read at the first CUDA allocation, so it must be set before any tensor reaches the GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
