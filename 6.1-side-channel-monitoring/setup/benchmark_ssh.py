"""Measure small-message RTT over persistent authenticated SSH connections."""
import argparse
import asyncio
import datetime
import json
import math
from pathlib import Path
import shlex
import statistics
import time


async def measure(host, args):
    echo = "import sys; print('ready', flush=True)\nfor line in sys.stdin:\n print(line, end='', flush=True)"
    command = ["ssh", "-T", "-F", args.config, host, "python3 -u -c " + shlex.quote(echo)]
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    result = {"host": host, "samples_ms": []}
    try:
        ready = await asyncio.wait_for(process.stdout.readline(), timeout=45)
        if ready != b"ready\n":
            raise RuntimeError("SSH echo process did not become ready")
        result["connection_and_process_start_ms"] = (time.perf_counter() - started) * 1000
        for i in range(args.samples + 3):
            payload = f"{i:08d}:0123456789abcdef0123456789abcdef\n".encode()
            sent = time.perf_counter()
            process.stdin.write(payload)
            await process.stdin.drain()
            reply = await asyncio.wait_for(process.stdout.readline(), timeout=10)
            elapsed = (time.perf_counter() - sent) * 1000
            if reply != payload:
                raise RuntimeError("SSH echo response did not match the request")
            if i >= 3:
                result["samples_ms"].append(elapsed)
            await asyncio.sleep(0.1)
        samples = sorted(result["samples_ms"])
        result.update(median_ms=statistics.median(samples),
                      p95_ms=samples[math.ceil(len(samples) * 0.95) - 1],
                      minimum_ms=samples[0], maximum_ms=samples[-1],
                      count=len(samples), success=True)
    except Exception as error:
        result.update(success=False, error=str(error))
    finally:
        if process.stdin:
            process.stdin.close()
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            _, stderr = await process.communicate()
        if not result.get("success"):
            result["stderr"] = stderr.decode(errors="replace")[-2000:]
    print(json.dumps({k: v for k, v in result.items() if k != "samples_ms"}), flush=True)
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="unspecified",
                        help="Measurement location, for example 'US participant venue'")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("hosts", nargs="+")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    results = await asyncio.gather(*(measure(host, args) for host in args.hosts))
    output = {"date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "method": "42-byte sequential echoes, persistent SSH, no PTY, 3 warmups, 100 ms pause between echoes; paths measured concurrently",
              "source": args.source,
              "results": results}
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
