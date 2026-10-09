# Archived waveform viewer

This folder contains the frontend recovered from the port-6008 waveform viewer,
a portable NumPy backend, and six Perfetto traces. Run it without a GPU,
oscilloscope, PyTorch, or PicoSDK. See [agent.md](agent.md) for setup.

The original Python backend could not be retrieved because SSH authentication
was unavailable during export. `serve.py` implements the recovered frontend's
HTTP interface; it is a new implementation, not a copy of that backend. The live
server retains its original backend. Its Plotly script now loads from the CDN,
and it is public through [Tailscale Funnel on HTTPS port 8443](https://amodo-gigabyte-3.pony-regulus.ts.net:8443/).

## Import a raw recording

Download either complete, losslessly compressed bundle from US-hosted R2:

| Recording | Raw bundle | Raw samples after extraction |
| --- | --- | --- |
| Baseline training cycle | [reference-raw.aisb.zip, 1.26 GB](https://traces.aisb.dev/side-channels/2026-10-09/reference/reference-raw.aisb.zip) | 1,050,000,000 int16 samples; 2.10 GB |
| Granite MoE training cycle | [granite-moe-raw.aisb.zip, 1.27 GB](https://traces.aisb.dev/side-channels/2026-10-09/granite-moe/granite-moe-raw.aisb.zip) | 1,050,000,000 int16 samples; 2.10 GB |

Both retain the entire 420 ms recording at 0.4 ns, with no cropping or
downsampling. Each ZIP contains `channel-a-adc.npy`, matching `summary.json`,
`trace.json.gz`, and a `bundle.json` inventory with hashes. See the
[raw manifest](https://traces.aisb.dev/side-channels/2026-10-09/raw-downloads.json)
and [archive checksums](https://traces.aisb.dev/side-channels/2026-10-09/RAW-SHA256SUMS).

From this directory, start the viewer on your own machine:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python serve.py
```

Open **http://127.0.0.1:6008/**, choose the downloaded file under **Import
recording**, and click **Load bundle**. Leave it compressed. Upload progress is
followed by checksum validation and waveform preparation; the recording opens
when ready. Allow approximately 5 GB of free disk per import. Preparation scans
every sample once, then zooming uses the prepared reductions and memory mapping.

Imported files persist under ignored `recordings/imports/<unique-id>/` and return
after a server restart. Repeated imports receive separate IDs and do not replace
existing recordings. `--imports /path/to/directory` selects another storage disk.
Import is available when the server binds to `127.0.0.1` or `localhost`; shared
network listeners remain read-only. The public viewer already has the original
recordings. There is no need to send a downloaded bundle back to Sheffield.

The overview shows exact min/max envelopes; close zoom reveals each raw sample.
RMS windows include 0.01, 0.1, 0.5, and 2 ms. **Open in Perfetto** uses the
included execution trace and its original timing metadata.

## Recordings

Choose a recording and click **Open in Perfetto**, or download a trace below and
open it directly at [ui.perfetto.dev](https://ui.perfetto.dev/). Every trace contains
CPU/CUDA execution and the **AC current RMS 0.1 ms (A)** counter. Where present,
layer annotations and NVIDIA driver readings are retained.

The baseline and small Granite MoE recordings also have public downloads from
US-jurisdiction Cloudflare R2, so participants can get them without SSH or a VPN:

- [Baseline training cycle, 6.51 MB](https://traces.aisb.dev/side-channels/2026-10-09/reference/trace.json.gz)
- [Granite-3.1-1B-A400M training cycle, 8.28 MB](https://traces.aisb.dev/side-channels/2026-10-09/granite-moe/trace.json.gz)
- [Download manifest and metadata links](https://traces.aisb.dev/side-channels/2026-10-09/downloads.json)
  and [SHA-256 checksums](https://traces.aisb.dev/side-channels/2026-10-09/SHA256SUMS)

These downloads preserve the original trace bytes. They include the 0.1 ms RMS
counter without raw ADC arrays or min/max envelopes. The local copies below
remain available when working offline.

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

## Earlier cropped recording

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

For new instrument recordings, `--traces /path/to/traces` selects a separate
prepared catalog alongside `--recordings /path/to/recordings`. This keeps capture
artifacts outside the source checkout. Each trace folder contains `summary.json`
and `trace.json.gz`; the matching raw folder contains `channel-a-adc.npy` and its
own `summary.json`. The host acceptance test can create this layout automatically.

## Provenance

The HTML, JavaScript, CSS, Plotly bundle, metadata, and traces were retrieved from
`http://amodo-gigabyte-3.pony-regulus.ts.net:6008/` on 2026-09-29. The frontend was
split into readable files, the outdated exercise link was removed, and missing
raw recordings now have an explicit message. The original Perfetto launch
handshake is retained. Plotly 3.1.1 loads from `https://cdn.plot.ly/plotly-3.1.1.min.js`,
with the vendored copy as a fallback for local offline use. It retains its
[MIT license](static/plotly-LICENSE.txt).
