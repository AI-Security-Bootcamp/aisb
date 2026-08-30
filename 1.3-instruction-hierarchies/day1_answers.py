# %%
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils import report
from aisb_utils.env import load_dotenv

load_dotenv()

# OpenRouter client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)


# %%

PREFILL_MODEL = "openai/gpt-4.1-mini"


def complete_with_prefill(
    user_message: str,
    prefill: str,
    model: str = PREFILL_MODEL,
    max_tokens: int = 50,
    ) -> str:
    """Send a chat completion request with an assistant prefill.

    The prefill is injected as the beginning of the assistant's response,
    and the model continues from there.

    Returns the full response including the prefill.
    """
    # TODO: Build a messages list with the user message followed by an
    # assistant message containing the prefill. Call the API and return
    # the prefill + the model's continuation.
    messages: list[dict] = [
    {"role": "user", "content": user_message},
    {"role": "assistant", "content": prefill},
    ]


    comp_response = openrouter_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens        
    )
    print(comp_response.choices[0].message.content)
    return prefill + comp_response.choices[0].message.content


# Compare with and without prefill
question = "What is the capital of France?"
normal = (
    openrouter_client.chat.completions.create(
        model=PREFILL_MODEL,
        messages=[{"role": "user", "content": question}],
        max_tokens=50,
    )
    .choices[0]
    .message.content
)

prefilled = complete_with_prefill(question, "I'LL ANSWER IN ALL CAPS, ")

print(f"Normal:    {normal}")
print(f"Prefilled: {prefilled}")
from section3_test import test_complete_with_prefill


test_complete_with_prefill(complete_with_prefill)

# %%
