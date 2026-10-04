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



LOGPROBS_MODEL = "openai/gpt-4.1-mini"  # Not all models support logprobs


def get_completion_with_logprobs(
    prompt: str,
    model: str = LOGPROBS_MODEL,
    max_tokens: int = 50,
    top_logprobs: int = 5,
) -> list[list[tuple[str, float]]]:
    """Get a completion with logprobs from the API.

    Returns for each generated token a list of (token, logprob) pairs
    for the top alternatives at that position.
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt}
    ]
    global_pairs = []
    resp = openrouter_client.chat.completions.create(model=LOGPROBS_MODEL, messages=messages, logprobs=True, top_logprobs=top_logprobs, max_tokens=max_tokens)
    if resp.choices[0].logprobs.content:
        for probe in resp.choices[0].logprobs.content:
            pairs = []
            for top_probe in probe.top_logprobs:
                log_pair = top_probe.token, top_probe.logprob
                pairs.append(log_pair)
            global_pairs.append(pairs)
    # TODO: Call the completions API on `openrouer_client` with logprobs=True and top_logprobs.
    # Parse the response into the format described in the docstring.
    # Hint: you will need choice.logprobs.content
    print("hey", global_pairs)
    return global_pairs


# Get logprobs for a simple prompt
token_pairs = get_completion_with_logprobs("My favorite joke")
completion = "".join(alts[0][0] for alts in token_pairs)
print(f"Completion: {completion}")
for alts in token_pairs[:10]:
    alt_str = ", ".join(f"{tok}({math.exp(lp):.1%})" for tok, lp in alts[:3])
    print(f"  - {alt_str}")
from section2_test import test_get_completion_with_logprobs


test_get_completion_with_logprobs(get_completion_with_logprobs)

# %%
