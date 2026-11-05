import { displayTime } from './comparison-math.mjs';
import { mixTimeline } from './signal-analysis.mjs';
const aborted = () => new DOMException('View changed', 'AbortError');

export async function loadUniform(selection, signal, minimumCount = 8) {
  if (selection.hi <= selection.lo) return null;
  const response = await fetch(`/api/rms-samples?${new URLSearchParams({ capture: selection.capture,
    mode: selection.mode, lo_ms: selection.lo, hi_ms: selection.hi })}`, { signal });
  if (!response.ok) throw Error((await response.json()).error || 'Could not load uniform RMS samples');
  const count = Number(response.headers.get('X-Sample-Count'));
  const interval = Number(response.headers.get('X-Sample-Interval-Ms'));
  const origin = Number(response.headers.get('X-Sample-Origin-Ms'));
  if (count < minimumCount) return null;
  if (!Number.isFinite(interval) || interval <= 0 || !Number.isFinite(origin)) throw Error('Invalid sample metadata');
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== count * 8) throw Error('Incomplete RMS samples');
  return { buffer, origin, count, intervalMs: interval, sampleRate: 1000 / interval };
}

// One cached worker per visible recording; two concurrent fetch/analysis jobs
// keep memory and bandwidth bounded when all six graphs switch to FFT at once.
export class AnalysisCache {
  constructor() { this.entries = new Map(); this.active = 0; this.queue = []; }
  async slot() {
    if (this.active >= 2) await new Promise(resolve => this.queue.push(resolve));
    this.active++;
  }
  release() { this.active--; this.queue.shift()?.(); }
  cancelPending() {
    for (const [key, entry] of this.entries) if (!entry.finished) {
      entry.abort.abort(); entry.dispose?.(); this.entries.delete(key);
    }
  }
  get(selection) {
    if (selection.hi <= selection.lo) return Promise.resolve(null);
    const key = JSON.stringify([selection.capture, selection.mode, selection.lo, selection.hi]);
    if (this.entries.has(key)) {
      const entry = this.entries.get(key); this.entries.delete(key); this.entries.set(key, entry);
      return entry.promise;
    }
    const entry = { abort: new AbortController(), finished: false };
    this.entries.set(key, entry);
    entry.promise = (async () => {
      await this.slot();
      try {
        if (entry.abort.signal.aborted) throw aborted();
        const series = await loadUniform(selection, entry.abort.signal);
        if (!series) { entry.finished = true; return null; }
        if (entry.abort.signal.aborted) throw aborted();
        const worker = new Worker(new URL('./signal-worker.mjs', import.meta.url), { type: 'module' });
        let serial = 0;
        const requests = new Map();
        entry.dispose = () => {
          worker.terminate();
          for (const request of requests.values()) request.reject(aborted());
          requests.clear();
        };
        worker.onmessage = ({ data }) => {
          const request = requests.get(data.requestId);
          if (!request) return;
          requests.delete(data.requestId);
          if (data.type === 'error') request.reject(Error(data.message)); else request.resolve(data);
        };
        worker.onerror = () => {
          for (const request of requests.values()) request.reject(Error('Signal worker failed. Zoom in and retry.'));
          requests.clear();
        };
        const call = (data, transfer = []) => new Promise((resolve, reject) => {
          const requestId = ++serial; requests.set(requestId, { resolve, reject });
          worker.postMessage({ ...data, requestId }, transfer);
        });
        const result = await call({ type: 'analyse', buffer: series.buffer, sampleRate: series.sampleRate }, [series.buffer]);
        entry.finished = true;
        const value = { result, origin: series.origin, count: series.count,
          audio: async (scale, speed, outputRate) => new Float32Array((await call({ type: 'audio',
            scale, speed, outputRate, normalize: false })).buffer) };
        // Keep the six current traces and a few recently used crops.
        for (const [oldKey, old] of this.entries) {
          if (this.entries.size <= 12) break;
          if (old !== entry && old.finished) { old.dispose?.(); this.entries.delete(oldKey); }
        }
        return value;
      } catch (error) {
        entry.dispose?.();
        if (this.entries.get(key) === entry) this.entries.delete(key);
        throw error;
      } finally { this.release(); }
    })();
    return entry.promise;
  }
}

export class AudioPanel {
  constructor(cache, getSelections, getReference) {
    this.cache = cache; this.getSelections = getSelections; this.getReference = getReference;
    this.serial = 0; this.view = 0;
    this.trace = document.getElementById('analysis-trace');
    this.status = document.getElementById('analysis-status');
    this.speed = document.getElementById('audio-speed');
    this.volume = document.getElementById('audio-volume');
    this.playButton = document.getElementById('play-viewport');
    this.stopButton = document.getElementById('stop-audio');
    this.spectrogram = document.getElementById('show-spectrogram');
    this.playButton.addEventListener('click', () => this.play().catch(error => this.error(error)));
    this.stopButton.addEventListener('click', () => this.stop());
    this.speed.addEventListener('change', () => this.stop());
    this.trace.addEventListener('change', () => this.changed());
    this.spectrogram.addEventListener('change', () => {
      document.getElementById('spectrogram-plot').hidden = !this.spectrogram.checked;
      if (this.spectrogram.checked) this.renderSpectrogram().catch(error => this.error(error));
    });
    this.volume.addEventListener('input', () => this.gain?.gain.setTargetAtTime(Number(this.volume.value), this.context.currentTime, 0.02));
  }
  setRecordings(records, labels) {
    this.trace.add(new Option('All shown traces · mix', 'all'));
    for (const item of records) this.trace.add(new Option(labels[item.record.variant], item.record.capture));
    this.trace.disabled = this.playButton.disabled = false;
  }
  error(error) { if (error.name !== 'AbortError') this.status.textContent = error.message; }
  changed() {
    this.stop(); this.view++;
    clearTimeout(this.timer);
    this.status.textContent = 'Listen to the displayed time interval, including its shifts and stretches. FFT view uses this same signal.';
    if (this.spectrogram.checked) {
      Plotly.purge('spectrogram-plot');
      this.timer = setTimeout(() => this.renderSpectrogram().catch(error => this.error(error)), 230);
    }
  }
  async renderSpectrogram() {
    const view = this.view;
    const selected = this.getSelections(this.trace.value === 'all' ? this.getReference() : this.trace.value)[0];
    if (!selected) return;
    const analysis = await this.cache.get(selected);
    if (view !== this.view || !analysis || !this.spectrogram.checked) return;
    const s = analysis.result.spectrogram, scale = selected.transform.scale;
    await Plotly.react('spectrogram-plot', [{ type: 'heatmap',
      x: s.seconds.map(t => displayTime(analysis.origin + t * 1000, selected.forwardStart, selected.transform)),
      y: s.frequency.map(f => f / scale), z: s.db, colorscale: 'Viridis', zmin: s.maximum - 80, zmax: s.maximum,
      colorbar: { title: { text: 'dB re 1 A' } },
      hovertemplate: '%{x:.5f} ms<br>%{y:.2f} displayed Hz<br>%{z:.2f} dB re 1 A<extra></extra>' }], {
      paper_bgcolor: '#101723', plot_bgcolor: '#101723', font: { color: '#d4e0ef' },
      margin: { l: 85, r: 35, t: 45, b: 65 },
      title: { text: 'Spectrogram · ' + selected.title, font: { size: 15 } },
      xaxis: { title: { text: 'Comparison time (ms)' } },
      yaxis: { type: 'log', title: { text: 'Frequency after time stretch (Hz)' } }, shapes: [],
    }, { responsive: true, displaylogo: false });
  }
  async play(id) {
    this.stop();
    if (id) this.trace.value = id;
    const serial = this.serial, speed = Number(this.speed.value);
    const selections = this.getSelections(this.trace.value);
    if (!selections.length) throw Error('No selected traces in this viewport.');
    const start = selections[0].viewport[0], end = selections[0].viewport[1];
    const duration = (end - start) / 1000 / speed;
    if (duration > 30) throw Error('Playback exceeds 30 seconds. Zoom in or choose a faster playback speed.');
    this.context ||= new AudioContext({ sampleRate: 48000 });
    await this.context.resume();
    if (serial !== this.serial) throw aborted();
    this.playButton.disabled = true; this.stopButton.disabled = false;
    this.status.textContent = 'Preparing the displayed, aligned waveform for playback…';
    try {
      const clips = await Promise.all(selections.map(async selection => {
        const data = await this.cache.get(selection);
        if (!data || serial !== this.serial) return null;
        const values = await data.audio(selection.transform.scale, speed, this.context.sampleRate);
        const origin = displayTime(data.origin, selection.forwardStart, selection.transform);
        return { values, offsetSeconds: (origin - start) / 1000 / speed, capture: selection.capture };
      }));
      if (serial !== this.serial) throw aborted();
      const available = clips.filter(Boolean);
      if (!available.length) throw Error('Select a wider interval containing RMS samples.');
      const samples = mixTimeline(available, duration, this.context.sampleRate);
      const buffer = this.context.createBuffer(1, samples.length, this.context.sampleRate);
      buffer.copyToChannel(samples, 0);
      const source = this.source = this.context.createBufferSource(); source.buffer = buffer;
      this.gain = this.context.createGain(); this.gain.gain.value = Number(this.volume.value);
      source.connect(this.gain).connect(this.context.destination);
      source.onended = () => { if (this.source === source) this.stop(); };
      source.start();
      window.comparisonPlayback = { duration: buffer.duration, speed, clips: available.map(c => ({
        capture: c.capture, offsetSeconds: c.offsetSeconds, samples: c.values.length })) };
      this.status.textContent = `Playing ${available.length} aligned trace${available.length === 1 ? '' : 's'} · ${buffer.duration.toFixed(2)} s at ${speed}× · shared timeline, DC removed, normalized.`;
      const started = this.context.currentTime; let last = 0;
      const tick = () => {
        if (this.source !== source) return;
        const elapsed = this.context.currentTime - started;
        if (this.spectrogram.checked && document.getElementById('spectrogram-plot').data && elapsed - last > 0.05) {
          const x = start + elapsed * speed * 1000;
          Plotly.relayout('spectrogram-plot', { shapes: [{ type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x,
            y0: 0, y1: 1, line: { color: '#fff', width: 2 } }] }); last = elapsed;
        }
        this.animation = requestAnimationFrame(tick);
      };
      this.animation = requestAnimationFrame(tick);
    } finally {
      if (serial === this.serial) { this.playButton.disabled = false; this.stopButton.disabled = !this.source; }
    }
  }
  stop() {
    this.serial++;
    if (this.source) { const source = this.source; this.source = null; source.stop(); source.disconnect(); this.gain?.disconnect(); }
    cancelAnimationFrame(this.animation);
    this.playButton.disabled = false; this.stopButton.disabled = true;
    if (this.status.textContent.startsWith('Playing') || this.status.textContent.startsWith('Preparing')) this.status.textContent = 'Playback stopped.';
  }
}
