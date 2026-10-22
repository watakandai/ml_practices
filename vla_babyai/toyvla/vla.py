"""The VLA: a SmolVLM backbone whose language-model head doubles as the action head.

Input  = chat prompt with [current frame] + mission + the last `history` actions as words.
Output = the next token after "Assistant:", which is trained to be one of the 7 action words.

Only the logits at the final prompt position matter: training is cross-entropy against the
first token of the action word (all other positions are masked out), and inference takes the
argmax over the 7 action tokens only (constrained decoding).

Background: a language model reads a sequence of tokens (word pieces) and, at every position,
outputs one score ("logit") per vocabulary entry for what the next token should be. A
vision-language model (VLM) like SmolVLM also turns an image into a few dozen "image tokens"
placed inside that sequence. So asking "what token comes after 'Assistant:'?" and training the
answer to be an action word turns the VLM into a policy, with no new output layer needed.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image  # image library; used to resize frames

from .common import ACTION_NAMES

# Hugging Face Hub name of the default backbone (downloaded on first use).
DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
# Names of the linear layers inside each transformer block that LoRA adapts:
# the attention projections (q, k, v, o) and the MLP projections (gate, up, down).
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def action_token_ids(tokenizer):
    """First token of " <action>" for each action. Must be distinct so one token decides the action."""
    # The leading space matters: after "Assistant:" the model writes " forward", not "forward",
    # and the two tokenize differently. A word may split into several tokens ("pickup" might
    # become " pick" + "up"); we keep only the first, which is enough if all 7 differ.
    ids = [tokenizer(" " + name, add_special_tokens=False).input_ids[0] for name in ACTION_NAMES]
    if len(set(ids)) != len(ids):  # two actions starting with the same token would be ambiguous
        raise ValueError("action words share a first token: %s" % dict(zip(ACTION_NAMES, ids)))
    return ids


class PromptEncoder:
    """Turns (frame, mission, recent actions) into processor tensors. Holds no model, so it is cheap
    to send to DataLoader workers."""

    def __init__(self, processor, history=4, image_size=512):
        # processor = tokenizer (text -> token ids) + image processor (image -> pixel tensors).
        self.processor, self.history, self.image_size = processor, history, image_size

    def prompt(self, mission, past_actions):
        text = "Mission: %s\n" % mission
        if self.history:
            # The last `history` actions as words, oldest first, e.g. "left forward forward".
            # They stand in for memory: the model sees one frame and has no state of its own.
            recent = " ".join(ACTION_NAMES[a] for a in past_actions[-self.history:])
            text += "Recent actions: %s\n" % (recent or "none")
        text += "Next action? One of: %s." % ", ".join(ACTION_NAMES)
        # A chat message whose content is an image placeholder followed by the text.
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        # Formats it the way SmolVLM was instruction-tuned ("User:<image>...<end_of_utterance>")
        # and, with add_generation_prompt, appends "Assistant:" so the next token is the answer.
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def __call__(self, frames, missions, histories):
        """frames (B, H, W, 3) uint8 -> processor tensors for B prompts."""
        # Upscale the pixel-art frame with nearest neighbour so cell edges stay sharp.
        # Each prompt gets a list of images (here exactly one), hence the inner [ ].
        images = [[Image.fromarray(f).resize((self.image_size, self.image_size), Image.NEAREST)] for f in frames]
        texts = [self.prompt(m, h) for m, h in zip(missions, histories)]
        # Tokenizes the texts, expands each image placeholder into its image tokens, normalizes
        # the pixels, and pads the prompts to equal length. Returns input_ids, attention_mask
        # (1 = real token, 0 = padding) and pixel_values, all as PyTorch tensors ("pt").
        return self.processor(text=texts, images=images, return_tensors="pt", padding=True)


class VLA:
    def __init__(self, model, processor, history=4, image_size=512):
        self.model, self.processor = model, processor
        self.history, self.image_size = history, image_size
        # Vocabulary ids of the 7 action words, in ACTION_NAMES order.
        self.action_ids = torch.tensor(action_token_ids(processor.tokenizer))
        # Pad on the left so every prompt ends at the last position: then logits[:, -1] is the
        # "next token" prediction for every row of the batch.
        processor.tokenizer.padding_side = "left"  # the last position is the prediction slot for every row
        ip = processor.image_processor
        ip.do_image_splitting = False  # one tile per frame instead of a grid of tiles + global view
        ip.size = {"longest_edge": image_size}  # resize target
        ip.max_image_size = {"longest_edge": image_size}
        # Image tokens per frame after the vision encoder's patching and the connector's pixel shuffle.
        # E.g. 512px / 16px patches = 32x32 patches; pixel shuffle merges 4x4 of them, leaving
        # 8x8 = 64 image tokens. Fewer tokens = shorter sequences = less memory.
        vcfg = model.config.vision_config
        processor.image_seq_len = (image_size // vcfg.patch_size // model.config.scale_factor) ** 2
        self.encode = PromptEncoder(processor, history, image_size)

    @classmethod
    def from_pretrained(cls, name=DEFAULT_MODEL, history=4, image_size=512, lora_rank=0,
                        freeze_vision=False, dtype=torch.float32, random_init=False):
        # Imported here so modules that never build a VLA don't pay transformers' import time.
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        # "name" is a Hub id or a local folder (a saved checkpoint); both work the same way.
        processor = AutoProcessor.from_pretrained(name)
        # fp32 master weights by default; training runs under fp16 autocast (Turing has no bf16).
        # attn_implementation="sdpa": PyTorch's built-in fused attention (FlashAttention-2
        # needs a newer GPU than the RTX 2070).
        if random_init:
            # Same architecture and tokenizer, freshly initialized weights: the pretraining ablation.
            model = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(name), dtype=dtype,
                                                            attn_implementation="sdpa")
        else:
            model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
        if freeze_vision:
            # requires_grad False = no gradients, so the optimizer never changes these weights
            # and no memory is spent on their gradients or optimizer state.
            for p in model.model.vision_model.parameters():
                p.requires_grad_(False)
        if lora_rank:
            # LoRA: freeze every original weight and learn a small low-rank correction
            # (two thin matrices of rank r) next to each targeted layer. Far fewer trainable
            # parameters, so the 500M model fits in 8 GB.
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05,
                                                     target_modules=LORA_TARGETS))
        return cls(model, processor, history, image_size)

    def logits(self, inputs):
        """Full-vocabulary logits at the last prompt position, (B, vocab)."""
        # **inputs passes input_ids, attention_mask, pixel_values, ... as keyword arguments.
        # logits_to_keep=1 computes scores only for the last position, which is all we use;
        # this saves memory, since the vocabulary has ~49k entries per position.
        out = self.model(**inputs, logits_to_keep=1)
        # [:, -1] takes the last (only) position; .float() casts fp16 to fp32 for a stable loss.
        return out.logits[:, -1].float()

    def loss(self, inputs, actions):
        """Cross-entropy on the action token only. actions: (B,) action indices."""
        logits = self.logits(inputs)  # (B, vocab)
        # Convert action indices (0..6) into vocabulary ids, the classes cross_entropy expects.
        targets = self.action_ids.to(logits.device)[actions]
        # Cross-entropy over the whole vocabulary: pushes probability onto the right action
        # word and away from every other token.
        loss = F.cross_entropy(logits, targets)
        # Accuracy as evaluation measures it: is the right action the best of the 7?
        acc = (self.constrain(logits).argmax(-1) == actions).float().mean()
        return loss, acc

    def constrain(self, logits):
        """(B, vocab) -> (B, 7) logits over the valid action tokens."""
        # Keep only the 7 action-word columns: the model can never output anything else.
        return logits[:, self.action_ids.to(logits.device)]

    def save(self, path):
        model = self.model
        if hasattr(model, "merge_and_unload"):  # LoRA: fold adapters into the weights
            # Adds each low-rank correction into its base weight, giving a plain model that
            # loads without peft.
            model = model.merge_and_unload()
        model.save_pretrained(path)  # weights + config.json
        self.processor.save_pretrained(path)  # tokenizer + image settings
        # Our own settings, needed to rebuild the same prompts at load time.
        with open(os.path.join(path, "vla_config.json"), "w") as f:
            json.dump({"history": self.history, "image_size": self.image_size}, f)


class VLAPolicy:
    """Evaluation wrapper: batched, constrained greedy decoding with fp16 autocast on GPU."""

    def __init__(self, vla, device):
        self.vla, self.device = vla, device
        vla.model.eval()  # inference mode: disables dropout

    def reset(self, missions):
        self.missions = list(missions)
        # Each episode keeps its own list of past actions, fed back in the prompt.
        self.histories = [[] for _ in missions]

    @torch.no_grad()  # no gradients needed at inference: faster, less memory
    def act(self, images, idx):
        # Prompts for the episodes still running (idx), each with its own mission and history.
        inputs = self.vla.encode(images, [self.missions[i] for i in idx], [self.histories[i] for i in idx])
        # autocast runs matrix multiplies in fp16 on the GPU (about 2x faster, half the memory).
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            logits = self.vla.logits(inputs.to(self.device))
        # Greedy, constrained decoding: the best of the 7 action tokens.
        actions = self.vla.constrain(logits).argmax(-1).tolist()
        for i, a in zip(idx, actions):
            self.histories[i].append(a)  # remember it for the next prompt
        return actions

    @classmethod
    def load(cls, path, device):
        with open(os.path.join(path, "vla_config.json")) as f:
            cfg = json.load(f)  # {"history": ..., "image_size": ...}
        # A saved checkpoint is a normal Hugging Face folder, so from_pretrained loads it.
        vla = VLA.from_pretrained(path, **cfg)
        vla.model.to(device)
        return cls(vla, device)
