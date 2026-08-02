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


from transformers import AutoTokenizer

# A conversation with all message types
SAMPLE_CONVERSATION: list[dict] = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions about weather.",
    },
    {
        "role": "user",
        "content": "What's the weather in London?"
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": '{"temp_c": 15, "condition": "cloudy"}',
    },
    {"role": "assistant", "content": "It's 15°C and cloudy in London."},
]


def serialize_conversation_chatml(messages: list[dict]) -> str:
    """Serialize a conversation to ChatML format.

    ChatML wraps each message in special tokens:
        <|im_start|>{role}\\n{content}<|im_end|>\\n

    For assistant messages with tool_calls (and no text content), serialize
    the entire tool_calls list as a single JSON array (use `json.dumps(..., indent=2)`)
    in place of content.

    For tool messages, include the tool_call_id in the role tag:
        <|im_start|>tool(tool_call_id={id})\\n{content}<|im_end|>\\n

    Returns the full serialized string (without a final generation prompt).
    """
    parts: list[str] = []
    for msg in messages:
        role: str = msg["role"]
        content: str | None = msg.get("content")
        tool_calls: list[dict] | None = msg.get("tool_calls")

        # Each branch only decides the role tag and the body; the ChatML
        # wrapper itself is applied once, below.
        if role == "tool":
            # Tool results carry the id of the call they answer.
            tag, body = f"tool(tool_call_id={msg['tool_call_id']})", content
        elif role == "assistant" and not content and tool_calls:
            # A tool-calling turn has no text, so the calls become the body.
            tag, body = role, json.dumps(tool_calls, indent=2)
        else:
            tag, body = role, content

        parts.append(f"<|im_start|>{tag}\n{body}<|im_end|>\n")

    return "".join(parts)


serialized = serialize_conversation_chatml(SAMPLE_CONVERSATION)
print("=== Serialized conversation (ChatML) ===")
print(serialized)
from section1_test import test_serialization


test_serialization(serialize_conversation_chatml)
# %%
