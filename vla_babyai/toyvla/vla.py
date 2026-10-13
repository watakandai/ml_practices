"""The VLA: a SmolVLM backbone whose language-model head doubles as the action head.

Input  = chat prompt with [current frame] + mission + the last `history` actions as words.
Output = the next token after "Assistant:", which is trained to be one of the 7 action words.

Only the logits at the final prompt position matter: training is cross-entropy against the
first token of the action word (all other positions are masked out), and inference takes the
argmax over the 7 action tokens only (constrained decoding).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .common import ACTION_NAMES

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def action_token_ids(tokenizer):
    """First token of " <action>" for each action. Must be distinct so one token decides the action."""
    ids = [tokenizer(" " + name, add_special_tokens=False).input_ids[0] for name in ACTION_NAMES]
    if len(set(ids)) != len(ids):
        raise ValueError("action words share a first token: %s" % dict(zip(ACTION_NAMES, ids)))
    return ids


class PromptEncoder:
    """Turns (frame, mission, recent actions) into processor tensors. Holds no model, so it is cheap
    to send to DataLoader workers."""

    def __init__(self, processor, history=4, image_size=512):
        self.processor, self.history, self.image_size = processor, history, image_size

    def prompt(self, mission, past_actions):
        text = "Mission: %s\n" % mission
        if self.history:
            recent = " ".join(ACTION_NAMES[a] for a in past_actions[-self.history:])
            text += "Recent actions: %s\n" % (recent or "none")
        text += "Next action? One of: %s." % ", ".join(ACTION_NAMES)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def __call__(self, frames, missions, histories):
        """frames (B, H, W, 3) uint8 -> processor tensors for B prompts."""
        # Upscale the pixel-art frame with nearest neighbour so cell edges stay sharp.
        images = [[Image.fromarray(f).resize((self.image_size, self.image_size), Image.NEAREST)] for f in frames]
        texts = [self.prompt(m, h) for m, h in zip(missions, histories)]
        return self.processor(text=texts, images=images, return_tensors="pt", padding=True)


class VLA:
    def __init__(self, model, processor, history=4, image_size=512):
        self.model, self.processor = model, processor
        self.history, self.image_size = history, image_size
        self.action_ids = torch.tensor(action_token_ids(processor.tokenizer))
        processor.tokenizer.padding_side = "left"  # the last position is the prediction slot for every row
        ip = processor.image_processor
        ip.do_image_splitting = False  # one tile per frame instead of a grid of tiles + global view
        ip.size = {"longest_edge": image_size}
        ip.max_image_size = {"longest_edge": image_size}
        # Image tokens per frame after the vision encoder's patching and the connector's pixel shuffle.
        vcfg = model.config.vision_config
        processor.image_seq_len = (image_size // vcfg.patch_size // model.config.scale_factor) ** 2
        self.encode = PromptEncoder(processor, history, image_size)

    @classmethod
    def from_pretrained(cls, name=DEFAULT_MODEL, history=4, image_size=512, lora_rank=0,
                        freeze_vision=False, dtype=torch.float32):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(name)
        # fp32 master weights by default; training runs under fp16 autocast (Turing has no bf16).
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
        if freeze_vision:
            for p in model.model.vision_model.parameters():
                p.requires_grad_(False)
        if lora_rank:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05,
                                                     target_modules=LORA_TARGETS))
        return cls(model, processor, history, image_size)

    def logits(self, inputs):
        """Full-vocabulary logits at the last prompt position, (B, vocab)."""
        out = self.model(**inputs, logits_to_keep=1)
        return out.logits[:, -1].float()

    def loss(self, inputs, actions):
        """Cross-entropy on the action token only. actions: (B,) action indices."""
        logits = self.logits(inputs)
        targets = self.action_ids.to(logits.device)[actions]
        loss = F.cross_entropy(logits, targets)
        acc = (self.constrain(logits).argmax(-1) == actions).float().mean()
        return loss, acc

    def constrain(self, logits):
        """(B, vocab) -> (B, 7) logits over the valid action tokens."""
        return logits[:, self.action_ids.to(logits.device)]

    def save(self, path):
        model = self.model
        if hasattr(model, "merge_and_unload"):  # LoRA: fold adapters into the weights
            model = model.merge_and_unload()
        model.save_pretrained(path)
        self.processor.save_pretrained(path)
        with open(os.path.join(path, "vla_config.json"), "w") as f:
            json.dump({"history": self.history, "image_size": self.image_size}, f)


class VLAPolicy:
    """Evaluation wrapper: batched, constrained greedy decoding with fp16 autocast on GPU."""

    def __init__(self, vla, device):
        self.vla, self.device = vla, device
        vla.model.eval()

    def reset(self, missions):
        self.missions = list(missions)
        self.histories = [[] for _ in missions]

    @torch.no_grad()
    def act(self, images, idx):
        inputs = self.vla.encode(images, [self.missions[i] for i in idx], [self.histories[i] for i in idx])
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            logits = self.vla.logits(inputs.to(self.device))
        actions = self.vla.constrain(logits).argmax(-1).tolist()
        for i, a in zip(idx, actions):
            self.histories[i].append(a)
        return actions

    @classmethod
    def load(cls, path, device):
        with open(os.path.join(path, "vla_config.json")) as f:
            cfg = json.load(f)
        vla = VLA.from_pretrained(path, **cfg)
        vla.model.to(device)
        return cls(vla, device)

