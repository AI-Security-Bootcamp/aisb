# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from aisb_utils import report
from aisb_utils.env import load_dotenv
import tiktoken
from transformers import AutoTokenizer



@report
def test_serialization(solution: Callable[[list[dict]], str]):
    conversation: list[dict] = [
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
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "London"}',
                    },
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
    result = solution(conversation)
    assert "<|im_start|>system" in result
    assert "<|im_start|>user" in result
    assert "<|im_start|>assistant" in result
    assert "tool_call_id=call_abc123" in result
    assert "<|im_end|>" in result
    assert "get_weather" in result
    print("  Serialization looks correct!")
