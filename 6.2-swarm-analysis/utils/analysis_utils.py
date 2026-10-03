"""Supplied mechanics: files, batching, parallelism, retries, and evidence checks.

Prompts and analysis choices stay in the solution files. No model calls are
defined here: each stage receives the solution's function as a callback.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

from .call_llm import usage_callback, validate_cached_responses


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def retry(function):
    """Try malformed replies three times; network/configuration errors propagate."""
    for attempt in range(3):
        try:
            with validate_cached_responses():
                return function()
        except (ValueError, KeyError, TypeError):
            if attempt == 2:
                raise


def chunks(records, max_records, max_chars):
    """Include every record exactly once, in order, without cutting its text."""
    group, size = [], 0
    for record in records:
        if group and (len(group) >= max_records or size + len(record["text"]) > max_chars):
            yield group
            group, size = [], 0
        group.append(record)
        size += len(record["text"])
    if group:
        yield group


def summarize_in_chunks(records, summarize, *, workers, max_records, max_chars, compact_ids=False):
    """Summarize in parallel, showing completed chunks and this run's reported cost."""
    cost = 0.0
    unreported = 0

    def show_cost(usage):
        nonlocal cost, unreported
        amount = usage.get("cost") if isinstance(usage, dict) else None
        with tqdm.get_lock():
            if type(amount) in (int, float) and math.isfinite(amount):
                cost += amount
            else:
                unreported += 1
            suffix = f" + {unreported} unreported" if unreported else ""
            progress.set_postfix_str(f"cost=${cost:.4f}{suffix}")

    def run(group):
        aliases = {(f"R{i}" if compact_ids else r["id"]): r["id"] for i, r in enumerate(group)}
        evidence = [dict(record, id=alias) for record, alias in zip(group, aliases)]

        def generate():
            summary = summarize(evidence)
            if not isinstance(summary["activities"], list) or not summary["activities"]:
                raise ValueError("Expected a nonempty activities list")
            for activity in summary["activities"]:
                if not isinstance(activity["name"], str) or not isinstance(activity["summary"], str):
                    raise ValueError("Each activity needs a name and summary")
                if not activity["source_ids"] or not set(activity["source_ids"]) <= aliases.keys():
                    raise ValueError("Supporting IDs must come from the supplied records")
            return summary

        summary = retry(generate)
        for activity in summary["activities"]:
            activity["source_ids"] = [aliases[i] for i in activity["source_ids"]]
        return {"source_ids": [r["id"] for r in group], "summary": summary}

    def run_with_progress(group):
        # Scope cost updates to this invocation, including its rejected replies.
        token = usage_callback.set(show_cost)
        try:
            result = run(group)
            with tqdm.get_lock():
                progress.update(1)
            return result
        finally:
            usage_callback.reset(token)

    groups = list(chunks(records, max_records, max_chars))
    with tqdm(total=len(groups), desc="Summarizing", unit="chunk",
              postfix="cost=$0.0000") as progress:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Workers update completion immediately; map preserves source order.
            return list(pool.map(run_with_progress, groups))


def build_codebook(summaries, make_codebook):
    """Require a flat codebook with unique IDs and clear label boundaries."""
    def generate():
        codebook = make_codebook(summaries)
        labels = codebook["labels"]
        ids = {label["id"] for label in labels}
        if not 0 < len(labels) <= 22 or len(ids) != len(labels):
            raise ValueError("Expected 1–22 labels with unique IDs")
        if ids & {"other", "insufficient_context"}:
            raise ValueError("Fallback labels must not be added to the codebook")
        for label in labels:
            if (set(label) != {"id", "name", "definition", "exclude"}
                    or not all(isinstance(value, str) and value for value in label.values())):
                raise ValueError("Each label needs nonempty id, name, definition, and exclude strings")
        return codebook

    return retry(generate)


def apply_labels(targets, annotations, allowed):
    """Join by ID; reject missing targets and clear quotes that are not literal."""
    if len(annotations) != len(targets):
        raise ValueError(f"Expected exactly {len(targets)} classifications")
    if {a["id"] for a in annotations} != {r["id"] for r in targets}:
        raise ValueError("Classify every target ID once, and no context IDs")
    by_id = {a["id"]: a for a in annotations}
    records = []
    for record in targets:
        annotation = by_id[record["id"]]
        if annotation["label"] not in allowed or not isinstance(annotation["quote"], str):
            raise ValueError("Each target needs a permitted label and a string quote")
        quote = annotation["quote"]
        if quote not in record["text"] or len(quote) >= 240:
            quote = ""
        records.append(dict(record, label=annotation["label"], quote=quote))
    return records


def classify_in_batches(source, evidence, codebook, classify, *, workers, batch_size, context_records,
                        compact_ids=False, split_failed_batches=False):
    """Add context, classify each target once, and preserve the original source."""
    if [r["id"] for r in evidence] != [r["id"] for r in source]:
        raise ValueError("Evidence must preserve source IDs and order")
    if compact_ids:
        evidence = [dict(record, id=f"R{i}") for i, record in enumerate(evidence)]
    allowed = {label["id"] for label in codebook["labels"]} | {"other", "insufficient_context"}

    def run(start, size):
        targets = evidence[start:start + size]
        end = start + len(targets)
        batch = {
            "codebook": codebook, "targets": targets,
            "target_ids": [r["id"] for r in targets],
            "context_before": evidence[max(0, start - context_records):start],
            "context_after": evidence[end:end + context_records],
        }

        def generate():
            result = classify(batch)
            return apply_labels(targets, result["records"], allowed)

        try:
            labeled = retry(generate)
        except (ValueError, KeyError, TypeError):
            if not split_failed_batches or len(targets) == 1:
                raise
            half = len(targets) // 2
            return run(start, half) + run(start + half, len(targets) - half)
        return [dict(record, label=a["label"], quote=a["quote"])
                for record, a in zip(source[start:end], labeled)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches = pool.map(lambda start: run(start, batch_size), range(0, len(source), batch_size))
        return [record for batch in batches for record in batch]
