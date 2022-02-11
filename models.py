"""
Import transformer model and tokenizer.

- Downloads model and tokenizer into model_dir for reuse.

USAGE:
    from models import model, tokenizer

Required Packages:
    pip install pyyaml
    pip install torch
    pip install transformers
    pip install sentencepiece

"""
import yaml
import csv

from transformers import MT5Config, MT5Tokenizer, MT5ForConditionalGeneration


config = None
with open('config.yaml') as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)

model_dir = config['MODEL_DIR']
special_tokens = config['SPECIAL_TOKENS']
vocab_size = config['VOCAB_SIZE']
num_layers = config['NUM_LAYERS']
num_heads = config['NUM_HEADS']

try:
    tokenizer = MT5Tokenizer.from_pretrained(model_dir)
except:
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    tokenizer.add_special_tokens({'additional_special_tokens' : special_tokens})
    tokenizer.save_pretrained(model_dir)

config = MT5Config(
    vocab_size=vocab_size,
    num_layers=num_layers,
    num_heads=num_heads
)

try:
    model = MT5ForConditionalGeneration.from_pretrained(model_dir, config=config)
except:
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small", config=config)
    model.save_pretrained(model_dir)

