# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import json
import sys
from collections import Counter
from pathlib import Path
from aisb_utils import report
from utils.call_llm import call_llm
from utils.analysis_utils import (
    build_codebook,
    classify_in_batches,
    read_jsonl,
    summarize_in_chunks,
    write_json,
)

folder = Path(__file__).resolve().parent



@report
def test_source(source=None):
    """Check input identity and evidence fields; no model calls."""
    if source is None:
        source = read_jsonl(folder / "inputs/00-wiki.jsonl")
    assert source, "Expected source records; check the input path."
    ids = [r["id"] for r in source]
    assert all(isinstance(i, str) and i for i in ids), "Every record needs an ID."
    assert len(ids) == len(set(ids)), "Duplicate source IDs would make citations ambiguous."
    for r in source:
        assert isinstance(r["text"], str) and r["text"], f"{r['id']}: missing source text"
        assert r["time"] is None or isinstance(r["time"], str), f"{r['id']}: invalid time"
    print(f"  All tests passed! {len(source):,} source records.")



@report
def test_summaries(summaries=None, source=None):
    """Check complete chunk coverage and citations, not the truth of summaries."""
    if source is None:
        source = read_jsonl(folder / "inputs/00-wiki.jsonl")
    if summaries is None:
        summaries = json.loads((folder / "outputs/wiki/01-summaries-solution.json").read_text())
    covered = [i for chunk in summaries for i in chunk["source_ids"]]
    assert covered == [r["id"] for r in source], "Chunks must cover every source ID once, in order."
    for chunk in summaries:
        activities = chunk["summary"]["activities"]
        assert activities, f"Chunk starting {chunk['source_ids'][0]} has no activities."
        for activity in activities:
            assert all(isinstance(activity[k], str) and activity[k].strip()
                       for k in ("name", "summary")), "Each activity needs a name and summary."
            ids = activity["source_ids"]
            assert ids and set(ids) <= set(chunk["source_ids"]), (
                f"{activity['name']}: citations must refer to records in its own chunk."
            )
    print("  All tests passed! Now check whether the cited text supports each claim.")



@report
def test_codebook(codebook=None):
    """Check flat, bounded label definitions; semantic coverage needs review."""
    if codebook is None:
        codebook = json.loads((folder / "outputs/wiki/01-codebook-solution.json").read_text())
    labels = codebook["labels"]
    assert 1 <= len(labels) <= 22, f"Expected 1–22 labels, got {len(labels)}."
    for label in labels:
        assert set(label) == {"id", "name", "definition", "exclude"}, (
            "Labels need id, name, definition, exclude; no parent categories or sublabels."
        )
        assert all(isinstance(v, str) and v.strip() for v in label.values()), (
            f"Empty or non-string field in {label!r}"
        )
    ids = [label["id"] for label in labels]
    assert len(ids) == len(set(ids)), "Codebook label IDs must be unique."
    assert not set(ids) & {"other", "insufficient_context"}, "Fallbacks are supplied separately."
    print("  All tests passed! Now review the label boundaries against the summaries.")



@report
def test_classifications(result=None, source=None, codebook=None):
    """Check source preservation, one label, and literal highlights for every record."""
    if source is None:
        source = read_jsonl(folder / "inputs/00-wiki.jsonl")
    if result is None:
        result = json.loads((folder / "outputs/wiki/02-classifications-solution.json").read_text())
    if codebook is None:
        codebook = json.loads((folder / "outputs/wiki/01-codebook-solution.json").read_text())
    assert result["track"] == "wiki", "The viewer needs the correct dataset track."
    assert result["labels"] == codebook["labels"], "Export the codebook used for this run."
    records = result["records"]
    assert [r["id"] for r in records] == [r["id"] for r in source], (
        "Every source ID must appear once and in order; do not export context twice."
    )
    allowed = {label["id"] for label in codebook["labels"]} | {"other", "insufficient_context"}
    for original, record in zip(source, records):
        assert all(record.get(k) == v for k, v in original.items()), (
            f"{original['id']}: preserve all source fields in the viewer export."
        )
        assert isinstance(record["label"], str) and record["label"] in allowed, (
            f"{record['id']}: expected exactly one permitted label ID."
        )
        quote = record["quote"]
        assert isinstance(quote, str) and len(quote) < 240 and quote in original["text"], (
            f"{record['id']}: highlight must be empty or an exact substring under 240 characters."
        )
    print("  All tests passed! Label correctness and outcome claims still need an evidence audit.")
