
# W1D1 - Section 2: Log Probabilities

This section connects the distribution an LLM computes at each generation step
to the information exposed by an inference API. You will request token-level
log probabilities, inspect the response structure, and reason about how that
extra observability changes an attacker's options.

## Table of Contents

- [Content & Learning Objectives](#content--learning-objectives)
- [Logprobs — What the Model Actually Computes](#logprobs-—-what-the-model-actually-computes)
    - [Exercise 2.1: Logprobs](#exercise-21-logprobs)

## Content & Learning Objectives

> **Learning Objectives**
> - Request and parse token-level log probabilities from a chat API
> - Relate logits, log probabilities, probabilities, and token rankings
> - Explain how output-distribution access can support extraction and adversarial optimization


```python


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
```

## Logprobs — What the Model Actually Computes

An LLM produces a **probability distribution over tokens** at each step. The API can return these as `logprobs`. This is the raw output before sampling — and it reveals information that the final text doesn't.


### Exercise 2.1: Logprobs

> **Difficulty**: 🔴🔴⚪⚪⚪
> **Importance**: 🔵🔵🔵⚪⚪

Use the [completions](https://developers.openai.com/api/reference/resources/completions/methods/create) API to make a request with `logprobs=True` and examine what comes back.


```python


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
    # TODO: Call the completions API on `openrouter_client` with logprobs=True and top_logprobs.
    # Parse the response into the format described in the docstring.
    # Hint: you will need choice.logprobs.content
    pass


# Get logprobs for a simple prompt
token_pairs = get_completion_with_logprobs("My favorite joke")
completion = "".join(alts[0][0] for alts in token_pairs)
print(f"Completion: {completion}")
for alts in token_pairs[:10]:
    alt_str = ", ".join(f"{tok}({math.exp(lp):.1%})" for tok, lp in alts[:3])
    print(f"  - {alt_str}")
from section2_test import test_get_completion_with_logprobs


test_get_completion_with_logprobs(get_completion_with_logprobs)
```

**Question: How can logprobs be misused by an attacker?**
<details>
<summary>Answer</summary><blockquote>

Logprobs leak information about the model's internal state beyond what sampled tokens alone reveal. The key threats include:
- **Adversarial prompt optimization** — logprobs provide a differentiable-like signal that attackers can use to iteratively refine jailbreak prompts. Instead of random guessing, they measure which token substitutions increase the probability of harmful completions, effectively using logprobs as a black-box gradient.
- **Model distillation/stealing** — logprobs expose the model's full probability distribution (or top-k), which provides far richer training signal than sampled tokens alone. An attacker can train a smaller "student" model on these soft labels, efficiently cloning the target model's behavior at a fraction of the original training cost.

Logprobs can also aid **system prompt extraction** and **model fingerprinting** (probability distributions can identify the model version or provider). This is why some providers restrict or disable logprobs access.
</blockquote></details>
