
# W1D1 - Section 3: Instruction Hierarchies and Assistant Prefill

This optional section connects instruction priority to the flat token sequence
processed by an LLM. You will use assistant prefills to observe how a partial
assistant turn can steer the model's continuation.

## Table of Contents

- [Content & Learning Objectives](#content--learning-objectives)
- [Instruction Hierarchies and Assistant Prefill (Optional)](#instruction-hierarchies-and-assistant-prefill-optional)
    - [Exercise 3.1: Assistant prefill (Optional)](#exercise-31-assistant-prefill-optional)
- [Summary](#summary)
    - [Key Takeaways](#key-takeaways)
    - [Further Reading](#further-reading)

## Content & Learning Objectives

> **Learning Objectives**
> - Relate system, developer, user, and tool content to principals with different authority
> - Explain why an assistant prefill changes the continuation distribution
> - Design and compare harmless prefills without repeating the API wrapper from scratch


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

## Instruction Hierarchies and Assistant Prefill (Optional)

AI labs are aware that input to an LLM is a flat string of tokens rather than something structured, and they are motivated to fix this to reduce jailbreaks, misuse, and the impact of indirect prompt injections. We already mentioned one layer: sanitization of control tokens at the API level.

Another layer is training models to respect [instruction hierarchies](https://arxiv.org/abs/2404.13208) that define how models should behave when instructions of different priorities conflict — e.g., giving more weight to the system prompt than to user messages or third-party content from tool calls.

A related attack vector is "prefilling" tokens in the assistant turn, effectively putting words in the model's mouth. By injecting a compliant prefix like `"Sure, here is how to do it:"`, an attacker exploits the model's self-consistency to steer it past safety training. This technique is sometimes called [sockpuppeting](https://www.trendmicro.com/vinfo/gb/security/news/cybercrime-and-digital-threats/sockpuppeting-how-a-single-line-can-bypass-llm-safety-guardrails). Some providers have stopped supporting prefill to prevent this effective jailbreaking technique (Anthropic dropped support for it starting with Claude Opus 4.6).

### Exercise 3.1: Assistant prefill (Optional)

> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵⚪⚪⚪

Let's try prefilling in practice. Not all providers support it — OpenRouter does for most models. Implement `complete_with_prefill` to make the model respond in all caps without the user asking for it.


```python

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
    pass


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
```

## Summary

Today you explored what happens under the hood of LLM inference APIs:

1. **Conversation Serialization** — Multi-turn conversations are serialized into flat token sequences using chat templates (ChatML, Mistral, etc.). The structured API is a convenience layer — the model sees one stream of tokens
2. **Token-level boundaries** — Special control tokens (`<|im_start|>`, `<|im_end|>`) mark message boundaries, but the serialization pipeline can be tricked into treating injected text as control tokens
3. **Logprobs** — The model produces a full probability distribution at each step; the API can expose this via logprobs, revealing information that sampled text alone doesn't

### Key Takeaways

- LLM "security boundaries" (system vs user prompt) are **conventions, not hard barriers** — the model sees a single token stream with no enforced separation
- **Control token injection** works at the tokenizer level (e.g., `apply_chat_template`) but production API stacks tokenize each part separately to prevent it
- **Logprobs** are the raw language model output before sampling. They are a powerful signal that can be exploited for adversarial prompt optimization, model stealing, and system prompt extraction

### Further Reading

**Tokenization and LLM APIs**
- [OpenAI API Reference: Chat Completions](https://developers.openai.com/api/reference/resources/chat-completions)
- [Dive into Deep Learning: Text Tokenization](https://d2l.ai/chapter_recurrent-neural-networks/text-sequence.html#tokenization)
- [HuggingFace: Common Chat Template Formats](https://huggingface.co/learn/llm-course/chapter11/2#common-template-formats)

**Instruction hierarchies**
- [The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions](https://arxiv.org/abs/2404.13208)
- Claude's constitution [refers to "principals"](https://www.anthropic.com/constitution#what-constitutes-genuine-helpfulness) when instructing whose instructions to give weight to and who it should act on behalf of
- [Control Illusion: The Failure of Instruction Hierarchies in Large Language Models](https://arxiv.org/abs/2502.15851)

