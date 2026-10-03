"""Model access, disk caching, and usage recording for the analyses.

Defaults come from .env.example. Environment variables override those defaults;
the repository-root .env and then the section's .env override the environment.
The cache saves parsed answers, keyed by a hash of the request. Usage is logged
only for API calls. Neither file stores request prompts, keys, or model reasoning.
"""

import hashlib
import json
import os
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

from dotenv import dotenv_values

folder = Path(__file__).resolve().parents[1]
settings = {
    **dotenv_values(folder / ".env.example"),
    **os.environ,
    **dotenv_values(folder.parent / ".env"),
    **dotenv_values(folder / ".env"),
}
usage_lock = Lock()
request_locks = {}
pending_cache_files = ContextVar("pending_cache_files", default=None)
usage_callback = ContextVar("usage_callback", default=None)


@contextmanager
def validate_cached_responses():
    """Discard an attempt's cached answers if downstream validation rejects them."""
    paths = []
    token = pending_cache_files.set(paths)
    try:
        yield
    except (ValueError, KeyError, TypeError):
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        pending_cache_files.reset(token)


def cached_response(url, body, fetch):
    """Read a saved answer or fetch it once; distinct requests run in parallel."""
    digest = hashlib.sha256(url.encode() + b"\0" + body).hexdigest()
    path = folder / ".cache/llm" / f"{digest}.json"
    # Identical requests in this process share one call, not one call per worker.
    with usage_lock:
        lock = request_locks.setdefault(digest, Lock())
    with lock:
        try:
            answer = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            answer = fetch()  # Errors and incomplete/non-JSON replies are not cached.
            path.parent.mkdir(parents=True, exist_ok=True)
            # Publish a complete file so another process cannot read a partial write.
            with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                temporary_path.write_text(json.dumps(answer), encoding="utf-8")
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
        paths = pending_cache_files.get()
        if paths is not None:
            paths.append(path)
        return answer


def record_usage(track, step, response_id, model, usage, provider=None):
    """Append each response once, before validating its contents."""
    path = folder / "outputs" / track / "usage.jsonl"
    row = {"step": step, "response_id": response_id, "model": model,
           "provider": provider, "usage": usage}
    # Parallel workers must not interleave writes to the same JSONL file.
    with usage_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")
    callback = usage_callback.get()
    if callback is not None:
        callback(usage)


def call_llm(prompt, data, *, task, effort="none", max_tokens=6500):
    """Return parsed JSON for a task such as wiki_summarize or transcript_codebook.

    Every task uses OPENROUTER_MODEL and OPENROUTER_API_KEY from .env. The task
    identifies the usage log; effort and token limits stay explicit at the call
    site. Identical requests reuse .cache/llm without another charge or usage row.
    The supplied analysis utilities discard malformed answers before retrying.
    """
    track, _, step = task.partition("_")
    if track not in {"transcript", "wiki"} or step not in {"summarize", "codebook", "classify"}:
        raise RuntimeError(f"Unknown analysis task: {task}")
    model = settings["OPENROUTER_MODEL"]
    base_url = settings["OPENROUTER_BASE_URL"].rstrip("/")
    url = base_url + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": "Return JSON:\n" + json.dumps(data, sort_keys=True)}],
        "reasoning": ({"enabled": False} if effort == "none"
                      else {"effort": effort, "exclude": True}),
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens, "temperature": 0,
        "provider": {"only": [settings["OPENROUTER_PROVIDER"]], "allow_fallbacks": False,
                     "require_parameters": True,
                     "quantizations": [settings["OPENROUTER_QUANTIZATION"]]},
    }, sort_keys=True).encode()

    def fetch():
        key = settings.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY in 6.2-swarm-analysis/.env")
        request = urllib.request.Request(
            url, data=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        record_usage(track, step, result.get("id"), result.get("model"),
                     result.get("usage"), result.get("provider"))
        choice = result["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("Incomplete model response")
        return json.loads(choice["message"]["content"])

    return cached_response(url, body, fetch)
