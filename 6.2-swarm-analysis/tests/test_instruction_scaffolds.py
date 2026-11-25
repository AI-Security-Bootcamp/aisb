"""Exercise the generated, commented helper calls without making model requests."""

import ast
import contextlib
import io
import re
import sys
import unittest
from pathlib import Path

# Make the workspace root importable (so `from aisb_utils import report` works),
# regardless of how deeply this file is nested.
_root = next(p for p in Path(__file__).resolve().parents if (p / "aisb_utils").is_dir())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aisb_utils.solution_parsing import build


def generated_instructions(track: str) -> str:
    section = Path(__file__).resolve().parents[1]
    instructions = io.StringIO()
    instructions.name = f"{track}_instructions.md"
    tests = io.StringIO()
    tests.name = f"{track}_test.py"
    with (section / f"{track}_solution.py").open() as source:
        with contextlib.redirect_stdout(io.StringIO()):
            build(source, instructions, tests)
    return instructions.getvalue()


def uncommented_call(markdown: str, variable: str) -> ast.Expression:
    """Extract the first complete assignment after uncommenting its example."""
    for code in re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL):
        lines = code.splitlines()
        for index, line in enumerate(lines):
            if not line.startswith(f"# {variable} = "):
                continue
            uncommented = []
            for line in lines[index:]:
                if not line.startswith("# "):
                    break
                uncommented.append(line[2:])
                try:
                    tree = ast.parse("\n".join(uncommented))
                except SyntaxError:
                    continue  # A multiline call may not have closed yet.
                return ast.Expression(tree.body[0].value)
    raise AssertionError(
        f"No runnable commented example assigning {variable!r} was generated."
    )


class InstructionScaffoldTests(unittest.TestCase):
    def test_examples_use_the_records_available_in_each_track(self):
        source = [
            {"id": str(i), "text": f"Record {i}", "author": "hidden"} for i in range(40)
        ]
        evidence = [{"id": row["id"], "text": row["text"]} for row in source]
        for track in ("transcript", "wiki"):
            with self.subTest(track=track):
                markdown = generated_instructions(track)
                records = source if track == "transcript" else evidence
                namespace = {
                    "source": source,
                    "summarize": lambda group: group,
                    "summarize_in_chunks": lambda group, callback, **kwargs: group,
                    "codebook": {},
                    "classify": None,
                    "classify_in_batches": lambda original, context, *args, **kwargs: (
                        original,
                        context,
                    ),
                }
                if track == "wiki":
                    namespace["evidence"] = evidence
                for variable, expected in (
                    ("summary", records[:30]),
                    ("summaries", records),
                    ("records", (source, records)),
                ):
                    expression = uncommented_call(markdown, variable)
                    result = eval(
                        compile(expression, "<generated example>", "eval"), namespace
                    )
                    self.assertEqual(
                        result, expected, f"Incorrect {variable} inputs in {track}."
                    )


if __name__ == "__main__":
    unittest.main()
