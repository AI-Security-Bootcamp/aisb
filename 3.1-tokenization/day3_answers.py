# %%
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
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir=cache_dir, trust_remote_code=True,
    )
    return model, tokenizer


# %%
def generate_response(question: str, model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Format a question as a chat prompt, generate a response, and return
    the decoded output."""
    model, tokenizer = load_model(model_name)
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(chat_prompt, max_new_tokens=1024, do_sample=False)

    response = tokenizer.decode(outputs[0][chat_prompt.shape[1]:], skip_special_tokens=True)
    return response


print(generate_response("I'm trying to decide whether to take another bootcamp."))
from section1_test import test_generate_response


test_generate_response(generate_response)


# %%
def generate_continue_message(question: str, model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Generate with continue_final_message=True to see the infinite-loop
    behaviour."""
    # TODO: Same as 1.1, but change the template parameters so the
    # model *continues* the user's message instead of starting a new
    # assistant turn. Check the hint above if you're unsure which
    # parameters to change. Cap max_new_tokens=256.
    model, tokenizer = load_model(model_name)
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=False,
        return_tensors="pt",
        continue_final_message=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(chat_prompt, max_new_tokens=256, do_sample=False)

    response = tokenizer.decode(outputs[0][chat_prompt.shape[1]:], skip_special_tokens=True)
    return response


print(generate_continue_message("I'm trying to decide whether to take another bootcamp."))
# %%
def compare_thinking_models(
    questions: list[str],
    model_names: list[str] = ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B"],
) -> None:
    """Generate and print responses for each (model, question) pair."""
    # TODO: For each model and question, generate a response
    # (same pipeline as 1.1) and print the result. Compare the
    # outputs between the thinking and non-thinking model.
    for model_name in model_names:
        print(f"Model: {model_name}")
        for question in questions:
            response = generate_response(question, model_name)
            print(f"Question: {question}")
            print(f"Response: {response}\n")


compare_thinking_models([
    "What is the capital of Japan?",
    "What is the distance between London and Edinburgh?",
])
# %%
