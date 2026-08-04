from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Callable
import torch
import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report

CACHE_DIR = "/tmp/cache-tokenizer"


def load_tokenizer(model_name: str, cache_dir: str = CACHE_DIR) -> AutoTokenizer:
    """Load a HuggingFace tokenizer (provided helper)."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(model_name: str, cache_dir: str = CACHE_DIR):
    """Load a model and tokenizer for generation (provided helper)."""
    tokenizer = load_tokenizer(model_name, cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    return model, tokenizer


def generate_response(question: str, model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Format a question as a chat prompt, generate a response, and return
    the decoded output."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [{"role": "system", "content": "Your are helpful assistant"}, {"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, continue_final_message=False, return_tensors="pt", return_dict=True
    )

    model = AutoModelForCausalLM.from_pretrained(model_name)

    outputs = model.generate(**inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(result)
    # TODO: Load the model, format the question as a chat prompt,
    # tokenize it, generate a response, and decode the output.
    # Key steps: apply_chat_template -> tokenize -> model.generate -> decode
    # Use add_generation_prompt=True so the model knows to respond.
    # Check the HuggingFace docs for model.generate().
    return result


print(generate_response("I'm trying to decide whether to take another bootcamp."))
from section1_test import test_generate_response


test_generate_response(generate_response)
