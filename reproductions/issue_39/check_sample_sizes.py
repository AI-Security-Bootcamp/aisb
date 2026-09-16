"""Compare original monitor prompt sizes and optionally retry selected samples.

Reconstructs the same prompt with ControlArena's monitor-view conversion and
formatter. Optional live checks bypass the cache, use four concurrent requests,
and impose deadlines; these controls differ from the original unbounded batch.
"""

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from control_arena import ControlTaskMetadata
from control_arena.monitor._last_tool_calls_prompt_formatter import (
    last_tool_calls_prompt_formatter,
)
from control_arena.tools import convert_agent_state_to_monitor_view
from dotenv import load_dotenv
from inspect_ai.agent import AgentState
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

ROOT = Path(__file__).resolve().parents[2]
MODEL = "openrouter/meta-llama/llama-3.1-8b-instruct"


def collect_cases(input_dir, trace):
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    completed = {
        (event["mode"], event["sample"]): event
        for event in events
        if event["event"] == "monitor_finished"
    }
    started = {
        (event["mode"], event["sample"])
        for event in events
        if event["event"] == "monitor_started"
    }
    template = (
        ROOT / "2.2-monitoring/reference_solutions/monitor_prompt.txt"
    ).read_text()
    cases = []
    for path in sorted(input_dir.glob("*.eval")):
        log = read_eval_log(path, resolve_attachments="full")
        for sample in log.samples or []:
            metadata = ControlTaskMetadata.model_validate(sample.metadata)
            identity = (metadata.eval_mode.value, str(metadata.main_task_name))
            assert identity in started, identity
            assistants = [
                i
                for i, message in enumerate(sample.messages)
                if message.role == "assistant"
            ]
            assert (
                len(assistants) == 1
            ), "This comparison expects one monitor call per sample"
            state = convert_agent_state_to_monitor_view(
                AgentState(messages=sample.messages[: assistants[0] + 1])
            )
            prompt = last_tool_calls_prompt_formatter(
                metadata, state.messages, template
            )
            previous = completed.get(identity, {})
            cases.append(
                {
                    "mode": identity[0],
                    "sample": identity[1],
                    "prompt": prompt,
                    "prompt_characters": len(prompt),
                    "prompt_utf8_bytes": len(prompt.encode()),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "task_characters": len(metadata.main_task_description),
                    "previously_pending": identity not in completed,
                    "previous_duration": previous.get("duration"),
                    "previous_usage": previous.get("usage"),
                }
            )
    sizes = sorted(case["prompt_characters"] for case in cases)
    for case in cases:
        case["size_rank_ascending"] = sizes.index(case["prompt_characters"]) + 1
    return cases


async def retry_cases(cases, output, rounds):
    model = get_model(
        MODEL,
        config=GenerateConfig(
            attempt_timeout=60,
            max_retries=0,
            max_connections=4,
        ),
    )
    semaphore = asyncio.Semaphore(4)
    results = []

    async def retry(case, round_number):
        async with semaphore:
            started = time.monotonic()
            result = {key: value for key, value in case.items() if key != "prompt"}
            result["round"] = round_number
            try:
                response = await asyncio.wait_for(
                    model.generate(
                        [ChatMessageUser(content=case["prompt"])], cache=False
                    ),
                    timeout=75,
                )
            except Exception as error:
                result.update(outcome="error", error_type=type(error).__name__)
                if error.__cause__ is not None:
                    result["cause_type"] = type(error.__cause__).__name__
            else:
                result.update(
                    outcome="completed",
                    usage=response.usage.model_dump() if response.usage else None,
                    stop_reason=(
                        response.choices[0].stop_reason if response.choices else None
                    ),
                    output_characters=len(response.completion),
                    has_score_tag="<score>" in response.completion,
                )
                name = f"{case['mode']}-{case['sample']}-round{round_number}.json"
                (output / name).write_text(response.model_dump_json(indent=2) + "\n")
            result["duration"] = round(time.monotonic() - started, 3)
            with (output / "retry-results.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            results.append(result)
            print(
                json.dumps(
                    {
                        key: result.get(key)
                        for key in [
                            "round",
                            "mode",
                            "sample",
                            "previously_pending",
                            "outcome",
                            "duration",
                            "usage",
                            "stop_reason",
                        ]
                    }
                ),
                flush=True,
            )

    for round_number in range(1, rounds + 1):
        ordered = list(cases)
        random.Random(round_number).shuffle(ordered)
        await asyncio.gather(*(retry(case, round_number) for case in ordered))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--env-file", type=Path, help="If supplied, run uncached live retries"
    )
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = collect_cases(args.input, args.trace)
    stalled = [case for case in cases if case["previously_pending"]]
    completed = [case for case in cases if not case["previously_pending"]]
    controls = sorted(
        completed, key=lambda case: case["prompt_characters"], reverse=True
    )[:4]
    summary = {
        "total_samples": len(cases),
        "min_prompt_characters": min(case["prompt_characters"] for case in cases),
        "median_prompt_characters": statistics.median(
            case["prompt_characters"] for case in cases
        ),
        "max_prompt_characters": max(case["prompt_characters"] for case in cases),
        "stalled": [
            {key: value for key, value in case.items() if key != "prompt"}
            for case in stalled
        ],
        "controls": [
            {key: value for key, value in case.items() if key != "prompt"}
            for case in controls
        ],
    }
    (args.output / "size-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "all-sizes.json").write_text(
        json.dumps(
            [
                {key: value for key, value in case.items() if key != "prompt"}
                for case in cases
            ],
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.env_file:
        load_dotenv(args.env_file)
        results = asyncio.run(retry_cases(stalled + controls, args.output, args.rounds))
        (args.output / "retry-results.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
