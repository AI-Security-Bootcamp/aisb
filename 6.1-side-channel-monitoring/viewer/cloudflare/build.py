"""Assemble the Pages deployment without copying raw captures or credentials."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def build(catalog, output):
    captures = json.loads(catalog.read_text())
    if not captures or "reference" not in captures:
        raise ValueError("Expected a capture catalog with the reference recording")
    for capture in captures.values():
        # JavaScript numbers cannot preserve 64-bit Unix nanoseconds. Keep the
        # exact metadata JSON for downloads while using parsed fields to plot.
        capture["summary_json"] = json.dumps(capture["summary"], ensure_ascii=False, separators=(",", ":"))
    output.mkdir(parents=True, exist_ok=True)
    for path in (ROOT.parent / "static").iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)
    for name in ("_worker.js", "waveform.mjs"):
        shutil.copy2(ROOT / name, output / name)
    (output / "catalog.json").write_text(json.dumps(captures, ensure_ascii=False) + "\n")
    # Functions handle only the API and data downloads; static assets use Pages.
    (output / "_routes.json").write_text(json.dumps({"version": 1, "include": ["/api/*", "/data/*"], "exclude": []}) + "\n")
    (output / "_headers").write_text(
        "/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: strict-origin-when-cross-origin\n"
        "/\n  Cache-Control: public, max-age=300\n"
    )
    (output / "404.html").write_text("<!doctype html><title>Not found</title><h1>Not found</h1>\n")
    print(f"Prepared {len(captures)} recordings in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    build(args.catalog, args.output)
