#%%
from day3_setup import generate
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Callable
import torch
import sys
from pathlib import Path
from aisb_utils import report

CACHE_DIR = "/tmp/cache-tokenizer"


def load_tokenizer(model_name: str, cache_dir: str = CACHE_DIR) -> AutoTokenizer:
    """Load a HuggingFace tokenizer (provided helper)."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(model_name: str, cache_dir: str = CACHE_DIR):
    """Load a model and tokenizer for generation (provided helper)."""
    tokenizer = load_tokenizer(model_name, cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, 
        cache_dir=cache_dir, trust_remote_code=True,
    )
    return model, tokenizer


def send_unguarded(query: str) -> str:
    """Send a query to the model with no guardrails and return the response."""
    # TODO: Send the query to the model with no guardrails.
    # Build a messages list with a single user message, then call
    # generate() to get the model's response.
    messages = [{"role": "user", "content": query}]
    model, tokenizer = load_model("Qwen/Qwen3-0.6B")
    outputs = generate(model, tokenizer, messages)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    with open("model_response.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
    return output_text
from section3_test import test_send_unguarded

test_send_unguarded(send_unguarded)