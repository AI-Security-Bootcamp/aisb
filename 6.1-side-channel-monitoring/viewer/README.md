# Archived waveform viewer

This folder contains the frontend recovered from the port-6008 waveform viewer,
a portable NumPy backend, and six Perfetto traces. Run it without a GPU,
oscilloscope, PyTorch, or PicoSDK. See [agent.md](agent.md) for setup.

Use the [hosted viewer at waveforms.aisb.dev](https://waveforms.aisb.dev/), including
[Baseline](https://waveforms.aisb.dev/?capture=reference) and
[Granite MoE](https://waveforms.aisb.dev/?capture=granite-moe). Cloudflare Pages
serves the site and API; the original six recordings and six
[attention/MLP comparison captures](https://waveforms.aisb.dev/attention-comparison.html)
live in the existing US R2 bucket.
The hosted viewer does not require the GPU servers or a Tailscale connection.
See [Cloudflare deployment](cloudflare/README.md) for maintenance.

In the [attention comparison](https://waveforms.aisb.dev/attention-comparison):

- Choose **0.001 ms (1 µs)** or **0.0001 ms (100 ns)** RMS. Zooming fetches
  finer detail; the wide view preserves RMS extrema instead of dropping peaks.
- Use the two spring sliders to the right of each graph to shift and stretch it.
  Hold left/right to adjust; release to recenter the knob and retain the change.
  **Auto-align FFT peaks** estimates stretch from spectral peaks and shift from
  waveform correlation, leaving weak matches unchanged. **Match forward bounds**
  remains available for matching profiler boundaries. **Overlay traces** compares
  signals on one plot. Recorded timings stay intact.
- **Align by peaks** lets you choose a fixed source and a target to move. Click
  two source peaks (S1, S2), then the corresponding target peaks (T1, T2).
  Each click snaps to the highest original RMS sample within ±10 horizontal
  pixels. The fourth pick shifts and stretches the target to match both pairs.
  **Undo point** corrects a pick; Escape cancels. Zoom first to distinguish nearby
  peaks. It also works in overlay mode and with previously adjusted traces.
- Name a preset and **Save scalings** / **Load scalings**. Presets include the
  viewport, RMS window, and reference. **Export JSON** / **Import JSON** move
  them between browsers.
- **FFT in place** replaces the displayed traces with their spectra, preserving
  the time selection. Frequencies follow each trace's time stretch. Clear
  **Log FFT amplitude (dB)** for a linear amplitude axis in amperes.
- **Play shown signals** mixes the aligned traces on their shared timeline;
  each graph also has a **Listen** button for that trace alone. Playback includes
  silence introduced by shifts and is slowed to 0.1× by default. The optional
  spectrogram follows the same displayed time/frequency coordinates. Analysis
  uses uniform samples before display reduction and reports any bandwidth limit.

All twelve hosted Perfetto traces include 0.01 and 0.1 ms RMS, sampled at the
original 1 µs output spacing. **Open in Perfetto** pins both tracks. The shorter
RMS window changes smoothing, not the instrument's timing uncertainty.

`serve.py` is the portable NumPy implementation for local viewing and imports.
It implements the original frontend's HTTP interface. The original server is
still available through [Tailscale Funnel on port 8443](https://amodo-gigabyte-3.pony-regulus.ts.net:8443/)
as a fallback. Plotly loads from its CDN, with a bundled offline fallback.

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
CPU/CUDA execution and **AC current RMS 0.01 ms (A)** and
**AC current RMS 0.1 ms (A)** counters. Where present,
layer annotations and NVIDIA driver readings are retained.

The baseline and small Granite MoE recordings also have public downloads from
US-jurisdiction Cloudflare R2, so participants can get them without SSH or a VPN:

- [Baseline training cycle, 12.17 MB](https://traces.aisb.dev/waveform-viewer/2026-10-10-rms-001/reference/trace.json.gz)
- [Granite-3.1-1B-A400M training cycle, 14.03 MB](https://traces.aisb.dev/waveform-viewer/2026-10-10-rms-001/granite-moe/trace.json.gz)
- [Download manifest and SHA-256 checksums](traces/manifest.json)

These revised downloads preserve the original events and timing, adding the
precomputed 0.01 ms RMS array without resampling. They omit raw ADC arrays.
The local copies below remain available when working offline. Historical raw
ZIP bundles still contain their original embedded trace; download the current
trace separately to get both RMS windows.

| Recording | Workload | Compressed size | Trace |
| --- | --- | ---: | --- |
| Original recording | One training cycle | 12.17 MB | [reference](traces/reference/trace.json.gz) |
| Workshop practice | One training cycle, with layer annotations | 12.19 MB | [pilot-layer-labels](traces/pilot-layer-labels/trace.json.gz) |
| Challenge A | One training cycle | 11.68 MB | [challenge-a](traces/challenge-a/trace.json.gz) |
| Granite MoE | One training cycle | 14.03 MB | [granite-moe](traces/granite-moe/trace.json.gz) |
| Qwen3-30B-A3B | One inference pass | 51.92 MB | [qwen3-30b-a3b-inference](traces/qwen3-30b-a3b-inference/trace.json.gz) |
| Qwen3-32B | One inference pass | 11.70 MB | [qwen3-32b-inference](traces/qwen3-32b-inference/trace.json.gz) |

The revision adds a counter track without trimming the original execution trace.
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
