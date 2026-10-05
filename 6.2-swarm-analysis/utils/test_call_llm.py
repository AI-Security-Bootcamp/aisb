"""Offline tests of the cache and retry mechanics; no model calls or fake model."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from . import call_llm as api
from .analysis_utils import retry


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "folder", tmp_path)


def test_reuses_disk_response_without_repeating_work():
    calls = []

    def calculate():
        calls.append(1)
        return {"sum": sum([1, 2, 3])}

    first = api.cached_response("endpoint", b"request", calculate)
    first["sum"] = -1  # A caller's later mutation must not change the saved answer.
    api.request_locks.clear()  # The next call has no in-memory request state.
    assert api.cached_response("endpoint", b"request", calculate) == {"sum": 6}
    assert len(calls) == 1
    assert len(list((api.folder / ".cache/llm").iterdir())) == 1


def test_endpoint_and_request_bytes_affect_cache_key():
    calls = []

    def calculate():
        calls.append(1)
        return len(calls)

    assert api.cached_response("a", b"one", calculate) == 1
    assert api.cached_response("b", b"one", calculate) == 2
    assert api.cached_response("a", b"two", calculate) == 3
    assert api.cached_response("a", b"one", calculate) == 1


def test_identical_concurrent_requests_compute_once():
    calls = []
    ready = Barrier(8)

    def calculate():
        calls.append(1)
        return {"sum": 6}

    def run(_):
        ready.wait(timeout=5)
        return api.cached_response("endpoint", b"request", calculate)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(run, range(8))) == [{"sum": 6}] * 8
    assert len(calls) == 1


def test_distinct_requests_can_compute_in_parallel():
    ready = Barrier(2)

    def calculate():
        ready.wait(timeout=5)
        return 6

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda body: api.cached_response("endpoint", body, calculate),
                                [b"one", b"two"]))
    assert results == [6, 6]


def test_fetch_failures_are_not_cached():
    def fail():
        raise ValueError("Incomplete response")

    with pytest.raises(ValueError, match="Incomplete response"):
        api.cached_response("endpoint", b"request", fail)
    assert not list(api.folder.rglob("*.json"))
    assert api.cached_response("endpoint", b"request", lambda: 6) == 6


def test_corrupt_file_is_replaced():
    api.cached_response("endpoint", b"request", lambda: 6)
    path = next(api.folder.rglob("*.json"))
    path.write_text("{unfinished", encoding="utf-8")
    assert api.cached_response("endpoint", b"request", lambda: 7) == 7
    assert json.loads(path.read_text()) == 7


def test_validation_rejection_discards_cache_before_retry():
    calls = []

    def calculate():
        calls.append(1)
        return len(calls)

    def validate():
        answer = api.cached_response("endpoint", b"request", calculate)
        if answer < 2:
            raise ValueError("Rejected by the caller")
        return answer

    # A previously saved answer must also be removed when validation rejects it.
    assert api.cached_response("endpoint", b"request", calculate) == 1
    assert retry(validate) == 2
    assert retry(validate) == 2
    assert len(calls) == 2


def test_failed_final_attempt_leaves_no_cached_answer():
    def validate():
        api.cached_response("endpoint", b"request", lambda: 6)
        raise ValueError("Still invalid")

    with pytest.raises(ValueError, match="Still invalid"):
        retry(validate)
    assert not list(api.folder.rglob("*.json"))


def test_call_llm_reads_matching_cache_without_key_or_usage(monkeypatch):
    """Use a real saved codebook as the answer; do not simulate an API server."""
    section = Path(__file__).resolve().parents[1]
    answer = json.loads((section / "outputs/transcript/01-codebook-solution.json").read_text())
    settings = {
        "OPENROUTER_MODEL": "model-a", "OPENROUTER_API_KEY": "",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "OPENROUTER_PROVIDER": "provider-a", "OPENROUTER_QUANTIZATION": "fp8",
    }
    monkeypatch.setattr(api, "settings", settings)
    body = json.dumps({
        "model": "model-a",
        "messages": [{"role": "system", "content": "Summarize"},
                     {"role": "user", "content": 'Return JSON:\n{"a": 1, "b": 2}'}],
        "reasoning": {"enabled": False}, "response_format": {"type": "json_object"},
        "max_tokens": 6500, "temperature": 0,
        "provider": {"only": ["provider-a"], "allow_fallbacks": False,
                     "require_parameters": True, "quantizations": ["fp8"]},
    }, sort_keys=True).encode()
    api.cached_response(settings["OPENROUTER_BASE_URL"] + "/chat/completions", body,
                        lambda: answer)
    assert api.call_llm("Summarize", {"b": 2, "a": 1}, task="transcript_codebook") == answer
    assert not list(api.folder.rglob("usage.jsonl")), "Cache hits must not log another charge."

    # Every meaningful input change must miss, hence need the deliberately absent key.
    for prompt, data, options in [
        ("Different prompt", {"a": 1, "b": 2}, {}),
        ("Summarize", {"a": 2, "b": 2}, {}),
        ("Summarize", {"a": 1, "b": 2}, {"effort": "high"}),
        ("Summarize", {"a": 1, "b": 2}, {"max_tokens": 9000}),
    ]:
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            api.call_llm(prompt, data, task="transcript_codebook", **options)
    for setting in ("OPENROUTER_MODEL", "OPENROUTER_BASE_URL",
                    "OPENROUTER_PROVIDER", "OPENROUTER_QUANTIZATION"):
        original = settings[setting]
        settings[setting] = original + "-different"
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            api.call_llm("Summarize", {"a": 1, "b": 2}, task="transcript_codebook")
        settings[setting] = original
