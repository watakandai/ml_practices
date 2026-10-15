"""A tiny, randomly initialised Idefics3 (SmolVLM's architecture) with a word-level tokenizer,
saved to disk so tests exercise the real from_pretrained / processor / chat-template path offline."""
import os

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (Idefics3Config, Idefics3ForConditionalGeneration, Idefics3ImageProcessor,
                          Idefics3Processor, PreTrainedTokenizerFast)

from toyvla.common import ACTION_NAMES, COLORS, OBJECT_TYPES

SPECIAL = ["<pad>", "<unk>", "<|im_start|>", "<end_of_utterance>", "<image>", "<fake_token_around_image>",
           "<global-img>"]
CHAT_TEMPLATE = (
    "<|im_start|>{% for message in messages %}{{ message['role'] | capitalize }}"
    "{% if message['content'][0]['type'] == 'image' %}{{ ':' }}{% else %}{{ ': ' }}{% endif %}"
    "{% for line in message['content'] %}{% if line['type'] == 'text' %}{{ line['text'] }}"
    "{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n"
    "{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}")
WORDS = (ACTION_NAMES + COLORS + OBJECT_TYPES +
         "User Assistant Mission Recent Next action actions none One of go to the a pick up put next open "
         "and then after you : , . ?".split())


def build_tiny_vla(path, image_size=64):
    if os.path.exists(os.path.join(path, "config.json")):
        return path
    vocab = {t: i for i, t in enumerate(SPECIAL + sorted(set(WORDS)))}
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>",
                                        additional_special_tokens=SPECIAL[2:])
    image_processor = Idefics3ImageProcessor(do_image_splitting=False, size={"longest_edge": image_size},
                                             max_image_size={"longest_edge": image_size})
    processor = Idefics3Processor(image_processor, tokenizer, image_seq_len=4, chat_template=CHAT_TEMPLATE)

    config = Idefics3Config(
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                           image_size=image_size, patch_size=16),
        text_config=dict(model_type="llama", hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=len(vocab),
                         pad_token_id=vocab["<pad>"]),
        scale_factor=2, image_token_id=vocab["<image>"], pad_token_id=vocab["<pad>"])
    model = Idefics3ForConditionalGeneration(config)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    return path
