# %%

import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

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

from transformers import AutoTokenizer
from typing import TypedDict, NotRequired
import json


class ToolCallFunction(TypedDict):
    name: str
    arguments: dict


class ToolCall(TypedDict):
    id: str
    type: str
    function: ToolCallFunction


class Message(TypedDict):
    role: str
    content: str | None
    tool_call_id: NotRequired[str]
    tool_calls: NotRequired[list[dict]]


# A conversation with all message types
# TODO: this needed to be moved to tests
SAMPLE_CONVERSATION: list[Message] = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions about weather.",
    },
    {"role": "user", "content": "What's the weather in London?"},
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


def serialize_conversation_chatml(messages: list[Message]) -> str:
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
    converted_messages = []
    for message in messages:
        content = message.get("content")

        if tool_calls := message.get("tool_calls", []):
            content = json.dumps(tool_calls, indent=2)

        if message.get("role") == "tool":
            id = message.get("tool_call_id")
            converted_message = f"<|im_start|>tool(tool_call_id={id})\n{content}<|im_end|>\n"
        else:
            converted_message = f"<|im_start|>{message.get('role')}\n{content}<|im_end|>\n"

        converted_messages.append(converted_message)

    # TODO: Serialize the conversation into ChatML format.
    # Each message becomes: <|im_start|>{role}\n{content}<|im_end|>\n
    # Handle tool_calls and tool messages as described in the docstring.
    return "".join(converted_messages)


serialized = serialize_conversation_chatml(SAMPLE_CONVERSATION)
print("=== Serialized conversation (ChatML) ===")
print(serialized)
from section1_test import test_serialization


test_serialization(serialize_conversation_chatml)

# %%
