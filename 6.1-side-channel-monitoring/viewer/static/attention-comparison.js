import { displayTime, recordedTime, matchForward, validTransform } from './comparison-math.mjs';
import { setupPresets } from './comparison-presets.mjs';
import { AnalysisCache, AudioPanel, loadUniform } from './comparison-analysis.mjs';
import { springSlider } from './spring-slider.mjs';
import { PeakAlignment } from './peak-alignment.mjs';

const labels = {
  efficient: 'Memory-efficient attention · baseline', flash: 'FlashAttention', eager: 'Eager attention',
  'no-attention': 'Attention bypassed · MLP remains', 'no-mlp': 'MLP bypassed · attention remains',
  'no-blocks': 'Attention and MLP bypassed',
};
const phaseColors = { forward: '#5aabff', backward: '#bb85ed', optimizer: '#ffaa53' };
const traceColors = ['#85c7ff', '#f9ba65', '#79dbb9', '#ea8cae', '#bc9bff', '#f3ea84'];
const graph = document.getElementById('comparison'), status = document.getElementById('comparison-status');
const mode = document.getElementById('rms-window'), reference = document.getElementById('alignment-reference');
const overlay = document.getElementById('overlay'), fftMode = document.getElementById('show-spectrum');
const fftLog = document.getElementById('fft-log'), shell = document.getElementById('comparison-shell');
const storageKey = 'aisb-attention-comparison-alignment-v1';
let recordings = [], displayRange = [0, 220], frequencyRange = null, requestNumber = 0, controller, timer;
let hasPlot = false, plotIsFFT = false, shapeSources = [], alignmentEpoch = 0, alignmentJob;
let peakPicker;
const cache = new AnalysisCache();
const player = new AudioPanel(cache, id => recordings.filter(item =>
  (id === 'all' ? item.visible !== false : item.record.capture === id)).map(selection), () => reference.value);

function forward(item) { return item.summary.phases.find(p => p.phase === 'forward'); }
function toDisplay(item, time) { return displayTime(time, forward(item).capture_start_ms, item.transform); }
function toRecorded(item, time) { return recordedTime(time, forward(item).capture_start_ms, item.transform); }
function referenceItem() { return recordings.find(item => item.record.capture === reference.value); }
function selection(item) {
  return { capture: item.record.capture, title: labels[item.record.variant], mode: mode.value,
    lo: Math.max(0, toRecorded(item, displayRange[0])), hi: Math.min(item.summary.duration_ms, toRecorded(item, displayRange[1])),
    forwardStart: forward(item).capture_start_ms, transform: { ...item.transform }, viewport: [...displayRange] };
}
async function json(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw Error(`Could not load comparison data (${response.status}). Try again.`);
  return response.json();
}
function syncControls(save = true) {
  if (peakPicker?.active) peakPicker.cancel('Peak picking cancelled because an alignment changed. Choose Align by peaks to start again.');
  for (const item of recordings) {
    item.scaleOutput.value = `${Number(item.transform.scale.toPrecision(7))}×`;
    item.shiftOutput.value = `${Number(item.transform.shift.toPrecision(7))} ms`;
  }
  const changed = recordings.some(item => item.transform.scale !== 1 || item.transform.shift !== 0);
  document.getElementById('alignment-status').textContent = changed
    ? 'Display alignment applied to waveform, FFT frequencies, and playback. Recorded durations and Perfetto are unchanged.'
    : 'Original recorded time; no display alignment applied.';
  if (save) try {
    localStorage.setItem(storageKey, JSON.stringify(Object.fromEntries(recordings.map(item => [item.record.capture, item.transform]))));
  } catch { /* Optional browser storage. */ }
}
function cancelAlignment() {
  alignmentEpoch++;
  alignmentJob?.abort.abort();
  for (const cancel of alignmentJob?.cancels || []) cancel();
}
function scheduleRefresh(dataChanged = true) {
  peakPicker?.viewChanged();
  requestNumber++;
  controller?.abort();
  if (dataChanged) { cancelAlignment(); cache.cancelPending(); player.changed(); }
  clearTimeout(timer); timer = setTimeout(refresh, 160);
}
function positionControls() {
  if (!graph._fullLayout?._size) return;
  const size = graph._fullLayout._size;
  for (const [index, item] of recordings.entries()) {
    const suffix = index ? String(index + 1) : '';
    const top = overlay.checked ? index / recordings.length : 1 - graph._fullLayout[`yaxis${suffix}`].domain[1];
    item.control.style.top = `${size.t + top * size.h}px`;
  }
  peakPicker?.render();
}
function previewTransform(item) {
  syncControls(false);
  player.stop();
  if (!hasPlot || plotIsFFT !== fftMode.checked) return;
  const now = performance.now();
  if (now - (item.previewTime || 0) < 45) return;
  item.previewTime = now;
  const index = recordings.indexOf(item);
  if (plotIsFFT && item.analysis) {
    Plotly.restyle(graph, { x: [item.analysis.result.spectrum.frequency.map(f => f / item.transform.scale)] }, [index]);
  } else if (!plotIsFFT && item.waveform) {
    Plotly.restyle(graph, { x: [item.waveform.x_ms.map(t => toDisplay(item, t))] }, [index]);
    const updates = {};
    shapeSources.forEach((source, i) => {
      if (source.item === item) {
        updates[`shapes[${i}].x0`] = toDisplay(item, source.phase.capture_start_ms);
        updates[`shapes[${i}].x1`] = toDisplay(item, source.phase.capture_end_ms);
      }
    });
    if (Object.keys(updates).length) Plotly.relayout(graph, updates);
  }
  status.textContent = 'Adjusting alignment locally… Release the slider to reload the exact visible samples.';
}
function addControls(item) {
  const name = labels[item.record.variant];
  reference.add(new Option(name, item.record.capture));
  const card = item.control = document.createElement('section');
  card.className = 'trace-control'; card.dataset.capture = item.record.capture;
  const title = document.createElement('h3'); title.textContent = name; card.appendChild(title);
  for (const [field, caption] of [['shift', 'Shift ← / →'], ['scale', 'Compress / stretch']]) {
    const label = document.createElement('label'); label.textContent = caption;
    const output = document.createElement('output'); item[`${field}Output`] = output; label.appendChild(output);
    const input = document.createElement('input');
    Object.assign(input, { type: 'range', min: '-1', max: '1', step: '0.01', value: '0' });
    input.dataset.field = field; input.setAttribute('aria-label', `${name}: ${field} spring slider`);
    input.title = 'Hold left or right to adjust continuously. Release to recenter.';
    label.appendChild(input); card.appendChild(label);
    let began = false;
    springSlider(input, (value, dt) => {
      if (!began) { began = true; requestNumber++; clearTimeout(timer); cancelAlignment(); controller?.abort(); cache.cancelPending(); }
      // Fine near the center, fast near either end; keep keyboard taps useful.
      const velocity = value * (0.05 + 0.95 * value * value);
      if (field === 'shift') item.transform.shift = Math.max(-1e6, Math.min(1e6,
        item.transform.shift + velocity * (displayRange[1] - displayRange[0]) * 0.18 * dt));
      else item.transform.scale = Math.max(0.01, Math.min(100, item.transform.scale * Math.exp(velocity * 0.65 * dt)));
      item.alignmentResult.textContent = ''; previewTransform(item);
    }, () => { began = false; syncControls(); scheduleRefresh(); });
  }
  for (const [caption, action, className] of [
    ['Auto-align', () => autoAlign([item]), 'auto-align'],
    ['Listen', () => player.play(item.record.capture), 'listen-trace'],
    ['Reset', () => { item.transform = { scale: 1, shift: 0 }; item.alignmentResult.textContent = ''; syncControls(); scheduleRefresh(); }, 'reset-trace'],
  ]) {
    const button = document.createElement('button'); button.textContent = caption; button.className = className;
    button.addEventListener('click', () => Promise.resolve().then(action).catch(error => { if (error.name !== 'AbortError') status.textContent = error.message; }));
    card.appendChild(button);
  }
  item.alignmentResult = document.createElement('span'); item.alignmentResult.className = 'alignment-result';
  item.alignmentResult.setAttribute('role', 'status'); card.appendChild(item.alignmentResult);
  document.getElementById('trace-controls').appendChild(card);
  const row = document.createElement('tr'), cell = document.createElement('td'), link = document.createElement('a');
  link.href = '/?capture=' + encodeURIComponent(item.record.capture); link.textContent = name; link.target = '_blank'; link.rel = 'noopener';
  cell.appendChild(link); row.appendChild(cell);
  for (const phase of ['forward', 'backward', 'optimizer']) {
    const cell = document.createElement('td'); cell.textContent = item.record.phases[phase].duration_ms.toFixed(2); row.appendChild(cell);
  }
  document.getElementById('comparison-table').appendChild(row);
}
function relayout(event) {
  if (Object.keys(event).some(key => /^xaxis\d*\.autorange$/.test(key) && event[key])) {
    if (plotIsFFT) frequencyRange = null; else displayRange = [0, 220];
    scheduleRefresh(!plotIsFFT); return;
  }
  for (const key of Object.keys(event)) {
    const match = /^(xaxis\d*)\.range(?:\[0\])?$/.exec(key);
    if (!match) continue;
    const range = Array.isArray(event[key]) ? event[key] : [event[`${match[1]}.range[0]`], event[`${match[1]}.range[1]`]];
    const old = plotIsFFT ? frequencyRange : displayRange;
    if (range.every(Number.isFinite) && range[1] > range[0] && (!old || range.some((value, i) => value !== old[i]))) {
      if (plotIsFFT) frequencyRange = range; else displayRange = range;
      scheduleRefresh(!plotIsFFT);
    }
    break;
  }
}
async function refresh() {
  peakPicker?.viewChanged();
  const number = ++requestNumber, useFFT = fftMode.checked;
  controller?.abort(); controller = new AbortController();
  shell.setAttribute('aria-busy', 'true');
  status.textContent = useFFT ? 'Calculating FFTs of the six aligned signals in the selected time interval…'
    : `Loading ${mode.value} ms RMS for the visible interval…`;
  try {
    const data = await Promise.all(recordings.map(async item => {
      const selected = selection(item);
      if (useFFT) return { analysis: await cache.get(selected) };
      if (selected.hi <= selected.lo || selected.hi - selected.lo < item.summary.interval_ns * 1e-6) return { wave: { x_ms: [], y: [] } };
      const query = new URLSearchParams({ capture: selected.capture, mode: selected.mode, lo_ms: selected.lo, hi_ms: selected.hi });
      return { wave: await json(`/api/waveform?${query}`, controller.signal) };
    }));
    if (number !== requestNumber) return;
    let maximum = useFFT && fftLog.checked ? -Infinity : 0, minimumFrequency = Infinity, maximumFrequency = 0;
    for (const [index, result] of data.entries()) {
      const curve = useFFT ? result.analysis?.result.spectrum : result.wave;
      const values = useFFT ? (fftLog.checked ? curve?.db : curve?.magnitude) : curve?.y;
      if (recordings[index].visible !== false) for (const value of values || []) maximum = Math.max(maximum, value);
      if (useFFT && curve) {
        minimumFrequency = Math.min(minimumFrequency, curve.frequency[0] / recordings[index].transform.scale);
        maximumFrequency = Math.max(maximumFrequency, curve.frequency.at(-1) / recordings[index].transform.scale);
      }
    }
    if (!Number.isFinite(maximum)) maximum = 0;
    const xRange = useFFT ? frequencyRange || (Number.isFinite(minimumFrequency)
      ? [Math.log10(minimumFrequency), Math.log10(maximumFrequency)] : [0, 6]) : displayRange;
    const yRange = useFFT && fftLog.checked ? [Math.max(-240, maximum - 100), maximum + 5] : [0, maximum > 0 ? maximum * 1.08 : 1];
    const layout = { paper_bgcolor: '#101723', plot_bgcolor: '#101723', font: { color: '#d4e0ef' },
      margin: { l: 85, r: 15, t: overlay.checked ? 100 : 35, b: 65 }, height: 1500,
      showlegend: overlay.checked, legend: { orientation: 'h', y: 1.08 }, shapes: [], annotations: [],
      dragmode: peakPicker?.active ? false : 'zoom', hovermode: 'closest' };
    const traces = [], gap = 0.035, height = (1 - gap * (data.length - 1)) / data.length;
    shapeSources = [];
    for (const [index, item] of recordings.entries()) {
      const suffix = overlay.checked || !index ? '' : String(index + 1), x = `x${suffix}`, y = `y${suffix}`;
      const top = 1 - index * (height + gap), bottom = overlay.checked || index === data.length - 1;
      layout[`xaxis${suffix}`] = { anchor: y, type: useFFT ? 'log' : 'linear', range: [...xRange],
        gridcolor: '#26364b', showticklabels: bottom, ...(suffix ? { matches: 'x' } : {}),
        ...(bottom ? { title: { text: useFFT ? 'Frequency after displayed time stretch (Hz)' : 'Comparison time (ms)' } } : {}) };
      layout[`yaxis${suffix}`] = { anchor: x, domain: overlay.checked ? [0, 1] : [top - height, top],
        type: 'linear', range: [...yRange], gridcolor: '#26364b', title: { text: useFFT
          ? (fftLog.checked ? 'FFT (dB re 1 A)' : 'FFT amplitude (A)') : 'RMS (A)' } };
      if (!overlay.checked) layout.annotations.push({ xref: 'paper', yref: 'paper', x: 0, y: top + 0.005,
        xanchor: 'left', yanchor: 'bottom', text: labels[item.record.variant], showarrow: false });
      let xs, ys, customdata, hover;
      if (useFFT) {
        item.analysis = data[index].analysis;
        const spectrum = item.analysis?.result.spectrum;
        xs = spectrum?.frequency.map(f => f / item.transform.scale) || [];
        ys = (fftLog.checked ? spectrum?.db : spectrum?.magnitude) || [];
        customdata = spectrum?.frequency || [];
        hover = `%{x:.3f} displayed Hz<br>%{customdata:.3f} recorded Hz<br>%{y:.5f} ${fftLog.checked ? 'dB re 1 A' : 'A'}<extra>%{fullData.name}</extra>`;
      } else {
        item.waveform = data[index].wave;
        xs = item.waveform.x_ms.map(t => toDisplay(item, t)); ys = item.waveform.y; customdata = item.waveform.x_ms;
        hover = 'Display %{x:.7f} ms<br>Recorded %{customdata:.7f} ms<br>%{y:.5f} A<extra>%{fullData.name}</extra>';
        if (!overlay.checked || item === referenceItem()) for (const phase of item.summary.phases) {
          if (!phaseColors[phase.phase]) continue;
          shapeSources.push({ item, phase });
          layout.shapes.push({ type: 'rect', xref: x, yref: `${y} domain`, x0: toDisplay(item, phase.capture_start_ms),
            x1: toDisplay(item, phase.capture_end_ms), y0: 0, y1: 1, fillcolor: phaseColors[phase.phase],
            opacity: 0.15, line: { width: 0 }, layer: 'below' });
        }
      }
      traces.push({ type: 'scatter', mode: 'lines', x: xs, y: ys, customdata, xaxis: x, yaxis: y,
        name: labels[item.record.variant], visible: item.visible === false ? 'legendonly' : true,
        line: { color: overlay.checked ? traceColors[index] : '#85c7ff', width: 1.2 }, hovertemplate: hover });
    }
    graph.style.height = `${layout.height}px`;
    await Plotly.react(graph, traces, layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['autoScale2d', 'resetScale2d'] });
    if (!hasPlot) {
      graph.on('plotly_relayout', relayout);
      graph.on('plotly_legendclick', event => {
        peakPicker?.cancel('Peak picking cancelled because trace visibility changed.');
        recordings[event.curveNumber].visible = recordings[event.curveNumber].visible === false; player.changed();
      });
      graph.on('plotly_legenddoubleclick', () => false);
      graph.on('plotly_afterplot', () => peakPicker?.render());
      hasPlot = true;
    }
    if (number !== requestNumber) return;
    plotIsFFT = useFFT; positionControls();
    document.getElementById('legend').hidden = useFFT;
    if (useFFT) {
      const shown = recordings.filter(item => item.analysis);
      const bands = shown.map(item => item.analysis.result.passbandHz / item.transform.scale);
      status.textContent = `FFT of aligned time interval ${displayRange[0].toFixed(4)}–${displayRange[1].toFixed(4)} ms · ${mode.value} ms RMS. ` +
        `${shown.length}/6 spectra; ${fftLog.checked ? 'log amplitude (dB)' : 'linear amplitude (A)'}. ` +
        (bands.length ? `Analysis bandwidth after stretch: ${(Math.min(...bands) / 1000).toPrecision(3)}–${(Math.max(...bands) / 1000).toPrecision(3)} kHz. ` : '') +
        'Toggle FFT off to return to the same time selection.';
    } else {
      const waves = data.map(d => d.wave).filter(wave => wave.y.length);
      const full = waves.filter(wave => wave.display_reduction === 'none').length;
      const detail = Number(mode.value) <= 0.001 ? (full === waves.length ? 'Every RMS bin is shown.' : 'Peak-preserving overview; zoom for every RMS bin.') : 'Original moving RMS; zoom for more detail.';
      status.textContent = `${mode.value} ms RMS. ${waves.length ? detail : 'No recordings in this interval.'} Drag to zoom all plots. Spring sliders keep their adjustment when released.`;
    }
    window.comparisonRecordings = recordings;
    window.comparisonView = { fft: useFFT, timeRange: [...displayRange], frequencyRange, logarithmicAmplitude: fftLog.checked };
  } catch (error) {
    if (error.name !== 'AbortError' && number === requestNumber) status.textContent = error.message;
  } finally { if (number === requestNumber) shell.setAttribute('aria-busy', 'false'); }
}

async function autoAlign(items) {
  peakPicker?.cancel('Peak picking cancelled for automatic FFT alignment.');
  cancelAlignment();
  const epoch = alignmentEpoch, target = referenceItem(), targetTransform = { ...target.transform };
  const targets = items.filter(item => item !== target);
  if (!targets.length) { target.alignmentResult.textContent = 'This is the reference trace.'; return; }
  const job = alignmentJob = { abort: new AbortController(), cancels: [] };
  document.querySelectorAll('.auto-align, #auto-align-all').forEach(button => { button.disabled = true; });
  const input = async item => {
    const phase = forward(item);
    const data = await loadUniform({ capture: item.record.capture, mode: '0.001',
      lo: phase.capture_start_ms, hi: phase.capture_end_ms }, job.abort.signal);
    if (!data) throw Error('Not enough forward samples for alignment.');
    return { buffer: data.buffer, sampleRate: data.sampleRate,
      originSeconds: (data.origin - phase.capture_start_ms) / 1000,
      durationSeconds: (phase.capture_end_ms - phase.capture_start_ms) / 1000 };
  };
  status.textContent = 'Matching forward-pass FFT peaks for stretch, then correlating waveforms for shift…';
  try {
    const ref = await input(target);
    const results = await Promise.all(targets.map(async item => {
      item.alignmentResult.textContent = 'Matching FFT peaks…';
      const source = await input(item);
      if (epoch !== alignmentEpoch) throw new DOMException('Alignment cancelled', 'AbortError');
      const result = await new Promise((resolve, reject) => {
        const worker = new Worker(new URL('./alignment-worker.mjs', import.meta.url), { type: 'module' });
        const cancel = () => { worker.terminate(); reject(new DOMException('Alignment cancelled', 'AbortError')); };
        job.cancels.push(cancel);
        worker.onerror = () => { worker.terminate(); reject(Error('Automatic alignment failed.')); };
        worker.onmessage = ({ data }) => { worker.terminate(); data.error ? reject(Error(data.error)) : resolve(data.result); };
        const target = { ...ref, buffer: ref.buffer.slice(0) };
        worker.postMessage({ source, target }, [source.buffer, target.buffer]);
      });
      return { item, result };
    }));
    if (epoch !== alignmentEpoch) return;
    for (const { item, result } of results) {
      const transform = { scale: result.ratio * targetTransform.scale,
        shift: toDisplay(target, forward(target).capture_start_ms) - forward(item).capture_start_ms +
          result.offsetSeconds * 1000 * targetTransform.scale };
      if (result.accepted && validTransform(transform)) {
        item.transform = transform;
        item.alignmentResult.textContent = `${result.matchedPeaks} FFT peaks · correlation ${result.correlation.toFixed(2)}`;
      } else item.alignmentResult.textContent = result.reason || 'Match is outside supported scale limits.';
      item.autoAlignment = result;
    }
    alignmentJob = null;
    syncControls(); scheduleRefresh();
  } catch (error) {
    if (error.name !== 'AbortError') status.textContent = error.message;
    for (const item of targets) if (item.alignmentResult.textContent === 'Matching FFT peaks…') item.alignmentResult.textContent = 'Alignment cancelled; trace unchanged.';
  } finally {
    if (alignmentJob === job) alignmentJob = null;
    document.querySelectorAll('.auto-align, #auto-align-all').forEach(button => { button.disabled = false; });
  }
}

async function start() {
  const records = await json('attention-comparison.json');
  let saved = {}; try { saved = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch { /* Optional storage. */ }
  recordings = await Promise.all(records.map(async record => ({ record,
    summary: await json(`/api/summary?${new URLSearchParams({ capture: record.capture })}`),
    transform: validTransform(saved?.[record.capture]) ? saved[record.capture] : { scale: 1, shift: 0 }, visible: true })));
  recordings.forEach(addControls); player.setRecordings(recordings, labels);
  peakPicker = new PeakAlignment({ graph, recordings, labels, referenceId: reference.value,
    toDisplay, toRecorded, forwardStart: item => forward(item).capture_start_ms,
    mode: () => mode.value, isFFT: () => fftMode.checked,
    isBusy: () => shell.getAttribute('aria-busy') === 'true',
    prepare: async () => {
      cancelAlignment(); player.stop(); clearTimeout(timer);
      fftMode.checked = false; frequencyRange = null; await refresh();
    },
    apply: (item, transform) => {
      item.transform = transform; item.alignmentResult.textContent = 'Matched two manually selected peak pairs.';
      syncControls(); scheduleRefresh();
    },
  });
  window.comparisonPeakAlignment = peakPicker;
  setupPresets(recordings.map(item => item.record.capture), () => ({
    transforms: Object.fromEntries(recordings.map(item => [item.record.capture, { ...item.transform }])),
    reference: reference.value, analysisTrace: player.trace.value, rmsWindow: mode.value,
    range: [...displayRange], overlay: overlay.checked, fftMode: fftMode.checked, fftLog: fftLog.checked,
  }), preset => {
    peakPicker.cancel('Loaded scalings; choose Align by peaks to start a new selection.');
    for (const item of recordings) { item.transform = { ...preset.transforms[item.record.capture] }; item.alignmentResult.textContent = ''; }
    reference.value = preset.reference; player.trace.value = preset.analysisTrace;
    mode.value = preset.rmsWindow; displayRange = [...preset.range]; overlay.checked = preset.overlay;
    fftMode.checked = preset.fftMode ?? false; fftLog.checked = preset.fftLog ?? true; frequencyRange = null;
    syncControls(); scheduleRefresh();
  });
  for (const id of ['rms-window', 'alignment-reference', 'auto-align-all', 'align-all', 'zoom-forward', 'reset-alignment', 'reset-zoom']) document.getElementById(id).disabled = false;
  document.getElementById('comparison-upload-status').hidden = !recordings.some(item => item.summary.raw_upload_pending);
  mode.addEventListener('change', () => { peakPicker.cancel('RMS window changed; choose Align by peaks to start again.'); scheduleRefresh(); });
  overlay.addEventListener('change', () => scheduleRefresh(false));
  reference.addEventListener('change', () => scheduleRefresh());
  fftMode.addEventListener('change', () => {
    peakPicker.cancel('Peak picking uses the time waveform. Choose Align by peaks to return to it.');
    frequencyRange = null; scheduleRefresh(false);
  });
  fftLog.addEventListener('change', () => scheduleRefresh(false));
  document.getElementById('reset-zoom').addEventListener('click', () => {
    if (fftMode.checked) frequencyRange = null; else displayRange = [0, 220]; scheduleRefresh(!fftMode.checked);
  });
  document.getElementById('auto-align-all').addEventListener('click', () => autoAlign(recordings));
  document.getElementById('align-all').addEventListener('click', () => {
    const target = referenceItem(), targetTransform = { ...target.transform };
    const transforms = recordings.map(item => matchForward(forward(item), forward(target), targetTransform));
    if (!transforms.every(validTransform)) { status.textContent = 'Matching exceeds supported scales. Reset the reference scale first.'; return; }
    recordings.forEach((item, index) => { item.transform = transforms[index]; item.alignmentResult.textContent = 'Matched profiler forward bounds.'; });
    syncControls(); scheduleRefresh();
  });
  document.getElementById('zoom-forward').addEventListener('click', () => {
    const item = referenceItem(), phase = forward(item);
    const start = toDisplay(item, phase.capture_start_ms), end = toDisplay(item, phase.capture_end_ms), padding = (end - start) * 0.04;
    displayRange = [start - padding, end + padding]; frequencyRange = null; scheduleRefresh();
  });
  document.getElementById('reset-alignment').addEventListener('click', () => {
    for (const item of recordings) { item.transform = { scale: 1, shift: 0 }; item.alignmentResult.textContent = ''; }
    syncControls(); scheduleRefresh();
  });
  new ResizeObserver(positionControls).observe(graph);
  syncControls(); await refresh();
}
start().catch(error => { status.textContent = error.message; });
