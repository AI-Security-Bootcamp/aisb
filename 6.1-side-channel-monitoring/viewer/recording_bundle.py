"""Read portable, lossless waveform bundles without trusting archive paths."""

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import stat
import zipfile

MAX_BUNDLE_BYTES = 3 * 1024**3
MAX_RAW_BYTES = 4 * 1024**3
FILES = {"bundle.json", "summary.json", "channel-a-adc.npy", "trace.json.gz"}


def unpack_bundle(archive_path: Path, destination: Path) -> tuple[str, dict]:
    """Validate and unpack into a new staging folder; never overwrite a capture.

    Hashes detect damaged downloads. They are not signatures or proof of origin.
    The caller publishes the folder only after validating the ADC array too.
    """
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) != len(FILES) or {m.filename for m in members} != FILES:
            raise ValueError("Expected bundle.json, summary.json, channel-a-adc.npy, and trace.json.gz")
        limits = {"bundle.json": 65536, "summary.json": 65536,
                  "channel-a-adc.npy": MAX_RAW_BYTES, "trace.json.gz": 100 * 1024**2}
        for member in members:
            if (member.is_dir() or stat.S_ISLNK(member.external_attr >> 16)
                    or member.flag_bits & 1 or not 0 < member.file_size <= limits[member.filename]):
                raise ValueError("Unsupported or oversized file in recording bundle")
        if shutil.disk_usage(destination.parent).free < sum(m.file_size for m in members) + 256 * 1024**2:
            raise ValueError("Not enough free disk space to unpack this recording")
        manifest = json.loads(archive.read("bundle.json"))
        if not isinstance(manifest, dict):
            raise ValueError("Invalid bundle manifest")
        if not isinstance(manifest.get("files"), dict):
            raise ValueError("Invalid bundle file inventory")
        identity = manifest.get("id", "")
        if manifest.get("format") != "aisb-waveform-v1" or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", identity):
            raise ValueError("Unsupported recording bundle format or ID")
        summary = json.loads(archive.read("summary.json"))
        if not isinstance(summary, dict):
            raise ValueError("Invalid capture metadata")
        for name in ("interval_ns", "current_A_per_adc_count", "duration_ms", "sample_rate_hz"):
            value = summary.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid capture metadata: {name}")
        samples = summary.get("samples")
        if type(samples) is not int or not 0 < samples <= MAX_RAW_BYTES // 2:
            raise ValueError("Invalid sample count")
        if not math.isclose(summary["duration_ms"], samples * summary["interval_ns"] / 1e6, rel_tol=1e-8):
            raise ValueError("Duration disagrees with sample count and interval")
        if not isinstance(summary.get("title"), str) or not isinstance(summary.get("phases"), list):
            raise ValueError("Missing title or phase metadata")
        if not isinstance(summary.get("perfetto"), dict) or not isinstance(summary.get("timing"), dict):
            raise ValueError("Missing execution-trace or clock metadata")
        destination.mkdir()
        for member in members:
            digest = hashlib.sha256()
            with archive.open(member) as source, (destination / member.filename).open("xb") as output:
                while block := source.read(4 * 1024**2):
                    digest.update(block)
                    output.write(block)
            if member.filename != "bundle.json":
                expected = manifest.get("files", {}).get(member.filename, {})
                if not isinstance(expected, dict) or expected.get("bytes") != member.file_size or expected.get("sha256") != digest.hexdigest():
                    raise ValueError(f"Checksum mismatch for {member.filename}")
        return identity, summary
