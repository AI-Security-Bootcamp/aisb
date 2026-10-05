# Viewer setup and maintenance

Work in this directory. This is hosting infrastructure; the exercise's source of
truth remains `../section1_solution.py`. Do not change the exercise when updating
the viewer.

## Start locally

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python serve.py
```

Open **http://127.0.0.1:6008/**. The six archived Perfetto recordings work
immediately. Raw waveform controls become available for each recording that has
both `recordings/<id>/channel-a-adc.npy` and `recordings/<id>/summary.json`.
The downloaded 30–175 ms crop uses ID `reference`.

Use `--recordings /absolute/path/to/recordings` to keep ADC data on another disk.
The server builds min/max and RMS reductions once at startup, memory-maps the
raw arrays, and serves only the selected display interval. Allow time for that
initial reduction before opening the page. The reference crop uses about 725 MB
of disk; raw data and Python environments are ignored by Git.

## Host on Tailscale

On the machine that has the files, bind to its Tailscale address:

```bash
.venv/bin/python serve.py --host "$(tailscale ip -4)" --port 6008
```

Keep the process in a persistent terminal such as tmux, or your existing service
manager. The URL is `http://<machine-name>.<tailnet>.ts.net:6008/`. If port 6008
already has a server, test the copy on `--port 6010` before replacing that service.
There is no need to reboot the acquisition machine or open the oscilloscope.

The Perfetto button opens `https://ui.perfetto.dev`, downloads the selected gzip
trace, and hands it to the new tab with a source/origin-checked `postMessage`
handshake. Allow the popup and accept Perfetto's open-trace prompt if shown.
Internet access is needed to load Perfetto; Plotly itself is served locally.
If the popup is blocked, use the trace download link and open the file manually.

## Files and conventions

- `serve.py`: portable backend for the archived frontend's HTTP API. The
  original remote backend was not available during export; do not describe this
  file as its original source.
- `static/`: cleaned frontend and vendored Plotly, with its license. Do not
  reformat the minified Plotly distribution.
- `traces/<id>/`: original `trace.json.gz` and full-capture `summary.json`.
- `traces/manifest.json`: source URLs, compressed sizes, and download hashes.
- `recordings/<id>/`: local raw ADC data and its matching metadata; never stage
  this directory. Cropped data must use cropped metadata, not the full-capture
  summary from `traces/`.

The server accepts only catalogued capture IDs and explicit static/data paths.
It supports HEAD and single byte-range downloads, including for large ADC files.
Acquisition scripts, model weights, SSH keys, and machine configuration do not
belong in this folder.

When cropping another recording, keep all samples in the requested interval,
update sample count and duration, clip/rebase phase intervals, and advance both
the acquisition origin and `perfetto.capture_zero_trace_ms` by the crop offset.
Preserve the original metadata and calibration uncertainty. Never shift the
waveform to make its peaks match execution labels.

## Verify a change

```bash
.venv/bin/python -m unittest -v test_serve.py
```

Then open the viewer, inspect the full envelope, zoom until individual raw
samples appear, choose an RMS window, and open the matching Perfetto trace.
Switch to a recording without installed raw samples and check that its Perfetto
button still works. Keep gzip traces compressed: their on-disk Git blob size is
what matters for GitHub's file limit.
