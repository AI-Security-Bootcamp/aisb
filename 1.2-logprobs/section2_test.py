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



@report
def test_get_completion_with_logprobs(
    solution: Callable[..., list[list[tuple[str, float]]]],
):
    result = solution("Hello", max_tokens=16, top_logprobs=3)

    assert isinstance(result, list) and len(result) > 0, (
        f"Expected a non-empty list (one entry per token), got {type(result).__name__}"
    )
    for i, alts in enumerate(result):
        assert isinstance(alts, list) and len(alts) > 0, (
            f"Position {i}: expected a non-empty list of alternatives, got {type(alts).__name__}"
        )
        assert len(alts) <= 3, (
            f"Position {i}: expected at most top_logprobs=3 alternatives, got {len(alts)}"
        )
        for token, logprob in alts:
            assert isinstance(token, str), (
                f"Position {i}: token should be a str, got {type(token).__name__}"
            )
            assert isinstance(logprob, float) and logprob <= 0, (
                f"Position {i}: logprob should be a non-positive float, got {logprob}"
            )

    print("  All tests passed!")
