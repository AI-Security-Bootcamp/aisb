# Archived waveform viewer

This folder contains the frontend recovered from the port-6008 waveform viewer,
a portable NumPy backend, and six Perfetto traces. Run it without a GPU,
oscilloscope, PyTorch, or PicoSDK. See [agent.md](agent.md) for setup.

The original Python backend could not be retrieved because SSH authentication
was unavailable during export. `serve.py` implements the recovered frontend's
HTTP interface; it is a new implementation, not a copy of that backend. The live
server has not been changed.

## Recordings

Choose a recording and click **Open in Perfetto**, or download a trace below and
open it directly at [ui.perfetto.dev](https://ui.perfetto.dev/). Every trace contains
CPU/CUDA execution and the **AC current RMS 0.1 ms (A)** counter. Where present,
layer annotations and NVIDIA driver readings are retained.

| Recording | Workload | Compressed size | Trace |
| --- | --- | ---: | --- |
| Original recording | One training cycle | 6.50 MB | [reference](traces/reference/trace.json.gz) |
| Workshop practice | One training cycle, with layer annotations | 6.51 MB | [pilot-layer-labels](traces/pilot-layer-labels/trace.json.gz) |
| Challenge A | One training cycle | 6.06 MB | [challenge-a](traces/challenge-a/trace.json.gz) |
| Granite MoE | One training cycle | 8.28 MB | [granite-moe](traces/granite-moe/trace.json.gz) |
| Qwen3-30B-A3B | One inference pass | 30.41 MB | [qwen3-30b-a3b-inference](traces/qwen3-30b-a3b-inference/trace.json.gz) |
| Qwen3-32B | One inference pass | 6.11 MB | [qwen3-32b-inference](traces/qwen3-32b-inference/trace.json.gz) |

These are the original compressed downloads, without trimming or re-encoding.
All are below GitHub's 100 MiB per-file limit, so they can share one commit.
Separate commits would not bypass that limit. See GitHub's
[large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Download URLs, exact sizes, and SHA-256 hashes are in
[traces/manifest.json](traces/manifest.json). Capture settings and clock alignment
are in each recording's `summary.json`.

## Raw recording

Raw ADC arrays are deliberately excluded from Git. The separately downloaded
reference crop belongs in `recordings/reference/` and covers **30 ms inclusive
to 175 ms exclusive** of the original recording. It retains **362,500,000 int16
samples at 0.4 ns spacing**: 145 ms, approximately 725 MB. No samples are averaged
or discarded within that interval.

The crop's own `summary.json` shifts phase times and the acquisition origin by
30 ms. `original-summary.json` preserves the original metadata. To relate the
cropped waveform to the unchanged Perfetto trace:

```text
original waveform time (ms) = crop time (ms) + 30
Perfetto time (ms) = crop time (ms) + 230.2919716796875
```

The viewer shows exact min/max envelopes at wide zoom and every ADC sample at
close zoom. It computes RMS windows of 0.01, 0.1, 0.5, and 2 ms from approximately
1 µs mean-square bins; every input sample contributes. At most 20,000 display
points are returned per request. RMS edges use nearest-bin padding, so the
recomputed RMS near the crop boundaries can differ from the original full trace.
The existing clock calibration has approximately ±0.1 ms uncertainty.

Other recordings work in Perfetto without their raw ADC files. To enable their
raw waveform views, install matching `channel-a-adc.npy` and `summary.json` files
under `recordings/<recording-id>/`.

## Provenance

The HTML, JavaScript, CSS, Plotly bundle, metadata, and traces were retrieved from
`http://amodo-gigabyte-3.pony-regulus.ts.net:6008/` on 2026-09-29. The frontend was
split into readable files, the outdated exercise link was removed, and missing
raw recordings now have an explicit message. The original Perfetto launch
handshake is retained. Plotly 3.1.1 is vendored with its
[MIT license](static/plotly-LICENSE.txt).
