"""Test codebook reasoning using the existing summaries, without classifying.

Reads: simple_outputs/01-summaries.json and the codebook function in step 01.
Writes: reasoning_codebook/01-codebook.json; usage appends to simple_outputs/costs.json.
The active codebook, classifications, and viewer are preserved.
Every run makes fresh paid calls. Set OPENROUTER_API_KEY or enter it when asked.
"""

import getpass
import json
import os
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    # Load the same function without running step 01's summarization loop.
    step = runpy.run_path(str(HERE / "steps/01-summarize.py"))
    summaries = json.loads((HERE / "simple_outputs/01-summaries.json").read_text())
    key = os.environ.get("OPENROUTER_API_KEY") or getpass.getpass("OpenRouter key (hidden): ")
    codebook = step["make_codebook"](summaries, key)
    output = HERE / "reasoning_codebook"
    output.mkdir(exist_ok=True)
    (output / "01-codebook.json").write_text(json.dumps(codebook, indent=2) + "\n")


if __name__ == "__main__":
    main()
