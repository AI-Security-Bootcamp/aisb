# Cloudflare archive hosting

The public viewer is **https://waveforms.aisb.dev/**, with
**https://aisb-waveforms.pages.dev/** as its Pages address. This hosts twelve
archived recordings independently of the GPU servers: the original six and six
[attention/MLP comparison captures](https://waveforms.aisb.dev/attention-comparison.html).
Acquisition still happens
over SSH; the portable Python viewer still supports local imports.

Cloudflare Pages serves the existing frontend. Its `_worker.js` implements the
same read-only catalog, summary, waveform, and download endpoints using the
`TRACES` R2 binding. The bucket is `aisb-side-channel-traces-us-20261009`, in the
US jurisdiction. Original objects use `waveform-viewer/2026-10-10/`; the attention
comparison uses `waveform-viewer/2026-10-10-attention/`. Merge both generated
catalogs before building a deployment so all twelve recordings remain available.
Existing downloadable ZIP bundles under `side-channels/2026-10-09/` remain valid.

Revised Perfetto traces with both 0.01 and 0.1 ms RMS use
`waveform-viewer/2026-10-10-rms-001/`. `add_rms_track.py` adds the original
0.01 ms array at every 1 µs output bin, preserving existing events and timing.
Upload its new inventory, then build from its updated catalog. Trace links
include a revision query parameter to avoid reusing older browser downloads.
Raw sample keys remain unchanged. Historical ZIP bundles retain their original
embedded trace; use the current viewer's separate Perfetto download for both RMS windows.

## Data and calibration

`prepare.py` reads the original archive bundles, including their `manifest.json`,
raw NPY array, precomputed envelope files, and `rms.npz`. It produces:

- An upload inventory with byte counts and SHA-256 hashes, and a capture catalog.
- Compact int16 min/max pairs at the original envelope strides.
- Original float64 RMS values, plus display levels selecting those same values.

Raw NPY files and compressed Perfetto traces are uploaded unchanged. Their sample
count, interval, current conversion, phase annotations, and timing uncertainty
stay with the original capture. Exact metadata JSON is retained separately from
JavaScript numbers so Unix nanosecond timestamps are not rounded on download.

Fine zoom reads only the requested ADC byte range. Wide zoom reads min/max bins;
partial boundary bins are corrected using smaller bins and raw edge samples,
so an out-of-view spike cannot appear in the selected interval. RMS requests
return original values at original times; only the displayed point count changes.
No request needs to contact the acquisition machine.

### Fine RMS and comparison alignment

The attention comparison defaults to **0.001 ms (1 µs)** RMS and also offers
**0.0001 ms (100 ns)**, alongside the existing 0.01 and 0.1 ms curves.
`add_fine_rms.py` computes each fine value directly from 2,500 or 250 original
0.4 ns ADC samples, using integer square sums before applying the probe gain.
Windows are consecutive and nonoverlapping; timestamps are the mean sample time.
Only complete windows are emitted (the six attention captures have no remainder).
These are additional arrays; raw captures, older moving RMS, and Perfetto stay intact.

The API returns every fine RMS bin when at most 6,000 fit the selected interval.
At wider zoom it returns each display bucket's minimum and maximum RMS at their
original timestamps, in time order. This preserves short excursions without
plotting millions of points. Partial boundary buckets are recomputed from the
fine array so off-screen extrema do not contaminate the view. The UI states
whether all bins are visible and requests new data on zoom or pan.

Each comparison trace has two spring sliders to the right of its graph. Their
deflection controls the rate of time shift or stretch, with fine adjustment near
the center. Pointer release, cancellation, key release, and blur recenter the
knob while retaining the transform. Shift speed scales with the time viewport.
Stretching pivots around the original forward start, then adds the shift.
**Match forward bounds** maps both profiler boundaries onto the selected
reference. Waveforms and phase shading use the same transform;
API queries use its inverse. Hover shows original and display time. Overlay mode
shows the selected reference's phase shading. Alignment is saved in browser
local storage and resettable; it is a visual comparison, not clock calibration.
The original approximately ±100 µs clock uncertainty still applies.

**Align by peaks** is a separate manual mode. Choose the source (stays fixed)
and target (moves), then click S1, S2, T1, T2 in the waveform plots. It also works
in overlay mode because the two recordings are explicit. Starting from FFT mode
returns to the time waveform. Each click maps a ±10 CSS-pixel horizontal window
through the plot axis and inverse time transform, then fetches every original
RMS sample in that window. It selects the highest sample, breaking equal-height
ties by distance from the click. Snapping uses the chosen RMS window, not a
reduced display series; one-sample crops are supported.

The target's new scale is `(displayed_S2 - displayed_S1) / (recorded_T2 - recorded_T1)`;
its shift makes T1 map exactly to displayed S1, using the existing forward-start
pivot. Both source and target may already have display transforms. Only the
target changes; source and other traces stay fixed. Duplicate picks, reversed
pair order, and unsupported transforms are rejected. Points are marked on the
graphs, **Undo point** removes the most recent pick, and Escape cancels. Changing
RMS windows, FFT mode, or another alignment cancels an unfinished selection;
viewport changes cancel pending snap requests while preserving completed picks.
The resulting transform uses the existing automatic browser persistence and
**Save scalings** presets. It never edits capture metadata or exported timestamps.

**Auto-align FFT peaks** uses each recording's forward-pass 1 µs RMS. The
alignment worker identifies significant local FFT maxima between two cycles per
forward pass and 25 kHz (bounded further by the signal bandwidth). It scores
candidate peak-frequency ratios and refines stretch using multiple matched
peaks. Waveform correlation on a common, anti-aliased 2,048-point grid estimates
shift within ±12% of the reference forward duration. FFT magnitude alone cannot
determine time shift. At least three matched peaks, a spectral score of 0.3,
and correlation of 0.25 are required; otherwise the transform is unchanged.
These are heuristic matching thresholds, not statistical confidence levels.
The source metadata and profiler timestamps are never rewritten. Known synthetic
stretches and positive/negative shifts are tested; actual captures currently
match three of five variants to the baseline, with the two weakest left alone.

To prepare an additional immutable export from locally restored raw arrays:

```bash
python add_fine_rms.py --catalog /path/to/current/catalog.json \
  --raw-dir /path/to/raw-files --output /path/to/fine-rms-export \
  --prefix waveform-viewer/YYYY-MM-DD-fine-rms \
  --captures attn-efficient-20261010 attn-flash-20261010 attn-eager-20261010 \
  attn-no-attention-20261010 attn-no-mlp-20261010 attn-no-blocks-20261010
```

Raw files are named `<capture-id>.npy`. Verify their archived SHA-256 values
before processing. Upload and verify the resulting `uploads.json`, then deploy
using its `catalog.json`, which retains all twelve captures and their current
Perfetto keys. No GPU or scope access is needed.

```bash
python -m unittest discover -s . -p test_fine_rms.py
node --test waveform.test.mjs comparison.test.mjs
```

The comparison also has named alignment presets. **Save scalings** stores the
six transforms, reference, playback selection, RMS window, time viewport,
overlay setting, FFT mode, and FFT amplitude scale in browser local storage.
Older preset files remain supported. Saving an existing name replaces that
preset. **Load scalings** restores it. **Export JSON** and **Import JSON** allow
portable copies; imports validate version, capture IDs, finite ranges, and
positive time scales before applying any settings.

### Viewport spectrum and audio

**FFT in place** replaces all six time plots (or their overlay) with spectra of
the same visible, transformed time intervals. Switching back restores the time
viewport; frequency zoom is stored separately and does not change the time crop.
FFT, spectrogram, and **Play shown signals** use uniform RMS samples from
`/api/rms-samples`, not the peak-selected plot points.
This read-only endpoint streams little-endian float64 values with sample count,
origin, and interval headers; it never transfers an entire raw ADC capture.
The FFT is of the chosen **RMS waveform**, not the GHz-rate signed ADC signal.

`signal-analysis.mjs` runs in a browser worker. Wide viewports are filtered with
a Blackman-windowed sinc FIR before integer decimation to at most 131,072 samples.
The UI reports the effective passband. The spectrum removes the mean, applies a
Hann window, zero-pads to a power of two, and uses coherent-gain-corrected
one-sided amplitude. **Log FFT amplitude (dB)** switches between dB relative to
1 A (not A-weighted decibels) and linear amplitude in amperes, reusing the same
cached spectra. The spectrogram uses up to 1,024-sample Hann windows and 256 time
columns. All frequency axes divide recorded frequency by the trace's time scale;
spectrogram timestamps also follow display alignment. A 2× stretch halves pitch.
Two fetch/analysis jobs run concurrently; recently used workers are cached so
changing amplitude scale or overlay does not refetch the recordings.

Audio follows the displayed time stretch and chosen playback speed (default
0.1×). It removes DC, low-pass filters before audio-rate resampling, and applies
a short fade at either end. Each clip is placed on the common display timeline,
including any leading/trailing silence caused by shifts. **All shown traces**
mixes visible signals with relative amplitudes preserved; per-trace **Listen**
plays one signal on that same timeline. The final mix is normalized to 0.8 peak
amplitude, not each constituent trace independently. The volume
control defaults to 15%; playback requires an explicit click. Stop, changed
viewports, and changed selected traces cancel playback. Playback is limited to
30 seconds; zoom in or choose a faster speed for longer transformed intervals.
This is sonification of RMS-current variations, not recorded microphone audio.

`node --test signal.test.mjs` checks known FFT tones, amplitudes, anti-alias
suppression, spectrogram frequencies, audio pitch/duration, and preset import
validation, spectral stretch/shift recovery, and mixed-audio time placement.
Browser checks cover spring behavior, in-place FFT, linear amplitude, shared
playback, weak alignment matches, and preservation of time zoom across FFT toggles.
With Playwright and its Firefox/Chromium browsers installed, run
`node spring-slider.browser.cjs` to check native drag, outside release, keyboard,
and track-click events. This guards against Firefox's final native `input` event
after `pointerup` restoring the dragged value or restarting the adjustment.

## Prepare and deploy

On the archive host, using Python with NumPy:

```bash
python prepare.py --catalog /path/to/original/bundles \
  --output /path/to/new/export --prefix waveform-viewer/YYYY-MM-DD
```

Upload every file in `uploads.json` to its specified R2 key using the S3 API.
Use multipart uploads for raw arrays. Preserve the inventory's SHA-256 value as
object metadata, verify uploaded sizes/checksums, and retain the inventory for
teardown. This repository deliberately contains neither upload credentials nor
multi-gigabyte ADC files. Use a fresh object prefix for a changed dataset.
Capture IDs are immutable: assign a new ID when replacing a recording so cached
waveforms and downloads cannot be confused with its previous version.

From this directory, after copying the generated catalog locally:

```bash
python build.py --catalog /path/to/export/catalog.json
node --test waveform.test.mjs
export CLOUDFLARE_ACCOUNT_ID=b8db77ee9a2f1cbcfedf9084d3c01779
# Supply CLOUDFLARE_API_TOKEN through your secret manager/environment.
npx wrangler@4.106.0 pages deploy dist --project-name=aisb-waveforms --branch=main
```

The Pages configuration binds the existing US R2 bucket; it needs no secret
inside the deployed Worker. Only `/api/*` and `/data/*` invoke the Worker.
The site is read-only and accepts GET/HEAD; use the local Python viewer to import
participant recordings. Plotly loads from its CDN with the existing local fallback.

After deployment, verify each added recording: full envelope, raw zoom, each RMS
window, compressed Perfetto download, and **Open in Perfetto**. Check byte-range
responses against the original raw arrays. The numerical tests cover extrema
outside viewport boundaries, partial final bins, raw conversion, RMS sampling,
and invalid ranges.

## Operations

`waveforms.aisb.dev` is a proxied CNAME to `aisb-waveforms.pages.dev` and is
registered as a custom domain on the Pages project. The original Tailscale
viewer is retained as a fallback; SSH/Funnel configuration is independent.

To retire this deployment, detach its custom domain, remove only that DNS record
and Pages project, and delete only the corresponding `waveform-viewer/` object
prefix once the recordings are preserved elsewhere. Keep the shared R2 bucket
and the separate public ZIP downloads if they are still in use. Local deployment
IDs, inventories, verification results, and teardown notes belong under the
ignored `instructor_setup/amodo-20261010/` directory.

References: [Pages advanced mode](https://developers.cloudflare.com/pages/functions/advanced-mode/),
[Pages R2 bindings](https://developers.cloudflare.com/pages/functions/bindings/#r2-buckets),
and the [R2 Workers API](https://developers.cloudflare.com/r2/api/workers/workers-api-reference/).
