import sys
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# day3_setup lives alongside this section; keep it importable.
_day3_setup_dir = Path(__file__).resolve().parent
if str(_day3_setup_dir) not in sys.path:
    sys.path.insert(0, str(_day3_setup_dir))


import torch
from day3_setup import (
    model,
    tokenizer,
    user_msg,
    system_msg,
    strip_thinking,
    show,
    show_verdict,
    CLASSIFIER_SYSTEM_PROMPT,
)
import os
from aisb_utils import report

BENIGN_QUERY = "How do I bake sourdough bread?"


print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


os.environ["HF_HOME"] = "/workspace/model-cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/model-cache"
CACHE = os.getenv("TRANSFORMERS_CACHE")

# The only model this section uses (Qwen/Qwen3-4B) is loaded by `day3_setup`
# on import, which caches it under HF_HOME. No extra manual downloads are needed.
print("Models are loaded on import by day3_setup (Qwen/Qwen3-4B).")


from day3_setup import generate


def send_unguarded(query: str) -> str:
    """Send a query to the model with no guardrails and return the response."""
    # TODO: Send the query to the model with no guardrails.
    # Build a messages list with a single user message, then call
    # generate() to get the model's response.
    messages = [{"role": "user", "prompt": query}]
    res = generate(model=model, tokenizer=tokenizer, messages=messages)
    return res


from section3_test import test_send_unguarded


test_send_unguarded(send_unguarded)
