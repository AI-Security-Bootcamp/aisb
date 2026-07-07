# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Callable
import torch
import sys
from pathlib import Path
from aisb_utils import report



# requires: GPU — this test loads Qwen3-0.6B and runs a real generation.
@report
def test_generate_response(solution: Callable[[str], str]):
    result = solution("What is the capital of Japan?")
    assert isinstance(result, str), "generate_response must return a string"
    assert len(result.strip()) > 0, "generate_response must return a non-empty string"
    # Qwen3-0.6B is a thinking model, so the decoded output should contain a
    # <think> tag from the assistant's chain-of-thought.
    assert "<think>" in result, "Expected a <think> tag in the thinking model's output"
    print("  All tests passed!")
