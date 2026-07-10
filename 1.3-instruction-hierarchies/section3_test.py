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



# NOTE: This test makes a live API call — requires the OPENROUTER_API_KEY env var and is non-deterministic.
@report
def test_complete_with_prefill(solution: Callable[..., str]):
    result = solution("What is the capital of France?", "I'LL ANSWER IN ALL CAPS, ")
    assert result.startswith("I'LL ANSWER IN ALL CAPS, "), (
        f"Response should start with the prefill, got: {result[:40]}"
    )
    continuation = result[len("I'LL ANSWER IN ALL CAPS, ") :]
    upper_ratio = sum(c.isupper() for c in continuation) / max(
        sum(c.isalpha() for c in continuation), 1
    )
    assert upper_ratio > 0.5, (
        f"Expected mostly uppercase continuation, got {upper_ratio:.0%} uppercase: {continuation[:60]}"
    )
    print("  All tests passed!")
