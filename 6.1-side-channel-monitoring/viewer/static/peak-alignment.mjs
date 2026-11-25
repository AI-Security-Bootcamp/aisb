import { highestSample, matchPeakPairs } from './comparison-math.mjs';
import { loadUniform } from './comparison-analysis.mjs';

// Four explicit picks, kept in recorded coordinates while the viewport moves.
// The selected source is fixed; only the target gets a new affine transform.
export class PeakAlignment {
  constructor(options) {
    this.options = options;
    this.graph = options.graph;
    this.active = false;
    this.points = [];
    this.pending = null;
    this.sourceSelect = document.getElementById('peak-source');
    this.targetSelect = document.getElementById('peak-target');
    this.button = document.getElementById('pick-peaks');
    this.undo = document.getElementById('undo-peak');
    this.status = document.getElementById('peak-status');
    this.markers = document.createElement('div');
    this.markers.className = 'peak-markers';
    for (const item of options.recordings) {
      for (const select of [this.sourceSelect, this.targetSelect]) {
        select.add(new Option(options.labels[item.record.variant], item.record.capture));
      }
    }
    this.sourceSelect.value = options.referenceId;
    this.targetSelect.value = options.recordings.find(item => item.record.capture !== options.referenceId).record.capture;
    this.button.addEventListener('click', () => this.active ? this.cancel('Peak alignment cancelled.') : this.start());
    this.undo.addEventListener('click', () => {
      this.viewChanged(); this.points.pop(); this.prompt(); this.render(); this.updateControls();
    });
    for (const select of [this.sourceSelect, this.targetSelect]) {
      select.addEventListener('change', () => this.cancel('Choose Align by peaks to begin a new set of four points.'));
    }

    // Plotly's point-click hit test only works very close to a rendered sample.
    // Use plot coordinates instead, so a click beside a peak also finds it.
    let down = null;
    this.graph.addEventListener('pointerdown', event => {
      if (!this.active || event.button !== 0 || event.target.closest('.modebar')) return;
      event.preventDefault(); event.stopImmediatePropagation();
      down = { x: event.clientX, y: event.clientY, id: event.pointerId };
      this.graph.setPointerCapture(event.pointerId);
    }, true);
    this.graph.addEventListener('pointerup', event => {
      if (!down || down.id !== event.pointerId) return;
      event.preventDefault(); event.stopImmediatePropagation();
      const click = Math.hypot(event.clientX - down.x, event.clientY - down.y) <= 5;
      down = null;
      if (this.graph.hasPointerCapture(event.pointerId)) this.graph.releasePointerCapture(event.pointerId);
      if (click) this.pick(event.clientX, event.clientY);
    }, true);
    this.graph.addEventListener('pointercancel', () => { down = null; });
    this.graph.addEventListener('click', event => {
      if (this.active && !event.target.closest('.modebar')) { event.preventDefault(); event.stopImmediatePropagation(); }
    }, true);
    window.addEventListener('keydown', event => { if (event.key === 'Escape' && this.active) this.cancel('Peak alignment cancelled.'); });
    this.updateControls();
  }

  updateControls() {
    this.sourceSelect.disabled = this.targetSelect.disabled = this.active;
    this.button.disabled = false;
    this.button.textContent = this.active ? 'Cancel peak alignment' : 'Align by peaks';
    this.button.setAttribute('aria-pressed', String(this.active));
    this.undo.disabled = !this.active || !this.points.length;
    this.graph.classList.toggle('peak-picking', this.active);
  }

  prompt() {
    if (!this.active) return;
    const source = this.points.length < 2, item = source ? this.source : this.target;
    const label = `${source ? 'S' : 'T'}${this.points.length % 2 + 1}`;
    this.status.textContent = `${this.points.length}/4 selected · Click ${label} on ${this.options.labels[item.record.variant]}. ` +
      'Snap: highest RMS point within ±10 horizontal pixels. Pair S1↔T1 and S2↔T2 in the same time order. Esc cancels.';
  }

  async start() {
    if (this.sourceSelect.value === this.targetSelect.value) {
      this.status.textContent = 'Choose different source and target waveforms.'; return;
    }
    const find = id => this.options.recordings.find(item => item.record.capture === id);
    this.source = find(this.sourceSelect.value); this.target = find(this.targetSelect.value);
    if (this.source.visible === false || this.target.visible === false) {
      this.status.textContent = 'Show both selected waveforms in the legend before picking peaks.'; return;
    }
    this.points = []; this.active = true; this.updateControls(); this.prompt(); this.render();
    try { await this.options.prepare(); }
    catch (error) { this.cancel(error.message); }
  }

  viewChanged() {
    this.pending?.abort(); this.pending = null;
    if (this.active) this.prompt();
  }

  cancel(message) {
    if (!this.active && !this.points.length) return;
    this.viewChanged(); this.active = false; this.points = [];
    this.updateControls(); this.render(); this.status.textContent = message;
    if (this.graph._fullLayout) Plotly.relayout(this.graph, { dragmode: 'zoom' });
  }

  axes(item) {
    const index = this.options.recordings.indexOf(item);
    const trace = this.graph.data?.[index], layout = this.graph._fullLayout;
    if (!trace || !layout) return null;
    return { x: layout[`xaxis${(trace.xaxis || 'x').slice(1)}`], y: layout[`yaxis${(trace.yaxis || 'y').slice(1)}`] };
  }

  async pick(clientX, clientY) {
    if (!this.active || this.pending || this.options.isFFT() || this.options.isBusy()) return;
    const item = this.points.length < 2 ? this.source : this.target;
    const axes = this.axes(item);
    if (!axes || axes.x.type !== 'linear') return;
    const bounds = this.graph.getBoundingClientRect();
    const x = clientX - bounds.left - axes.x._offset, y = clientY - bounds.top - axes.y._offset;
    if (x < 0 || x > axes.x._length || y < 0 || y > axes.y._length) {
      this.status.textContent = `Click inside ${this.options.labels[item.record.variant]} for the next point. Undo point corrects an earlier pick.`;
      return;
    }
    const clickedTime = this.options.toRecorded(item, axes.x.p2d(x));
    const lo = Math.max(0, this.options.toRecorded(item, axes.x.p2d(Math.max(0, x - 10))));
    const hi = Math.min(item.summary.duration_ms, this.options.toRecorded(item, axes.x.p2d(Math.min(axes.x._length, x + 10))));
    const request = this.pending = new AbortController();
    this.status.textContent = 'Finding the highest RMS sample within ±10 pixels…';
    try {
      const data = await loadUniform({ capture: item.record.capture, mode: this.options.mode(), lo, hi }, request.signal, 1);
      if (this.pending !== request || !this.active) return;
      const peak = data && highestSample(new Float64Array(data.buffer), data.origin, data.intervalMs, clickedTime);
      if (!peak) throw Error('No RMS samples within 10 pixels. Click closer to the waveform or zoom out.');
      if (this.points.some(point => point.item === item && Math.abs(point.time - peak.time) < data.intervalMs / 2)) {
        throw Error('That snapped to the same peak. Pick a different peak, or zoom in to separate nearby peaks.');
      }
      const point = { ...peak, item, label: `${this.points.length < 2 ? 'S' : 'T'}${this.points.length % 2 + 1}` };
      let transform;
      if (this.points.length === 3) {
        transform = matchPeakPairs(this.points.slice(0, 2).map(p => this.options.toDisplay(p.item, p.time)),
          [this.points[2].time, point.time], this.options.forwardStart(this.target));
      }
      this.points.push(point);
      if (transform) {
        this.active = false;
        this.lastResult = { source: this.source.record.capture, target: this.target.record.capture,
          points: this.points.map(p => ({ capture: p.item.record.capture, time: p.time, value: p.value, label: p.label })),
          transform: { ...transform } };
        this.options.apply(this.target, transform);
        this.status.textContent = `Aligned ${this.options.labels[this.target.record.variant]} to ${this.options.labels[this.source.record.variant]}: ` +
          `S1↔T1 and S2↔T2 · ${transform.scale.toPrecision(7)}× stretch, ${transform.shift.toFixed(6)} ms shift. Source unchanged. Save scalings to keep a named preset.`;
        await Plotly.relayout(this.graph, { dragmode: 'zoom' });
      } else this.prompt();
      this.updateControls(); this.render();
      Plotly.Fx.unhover(this.graph);
    } catch (error) {
      if (error.name !== 'AbortError' && this.pending === request) this.status.textContent = `${error.message} Use Undo point to revise a selection.`;
    } finally { if (this.pending === request) this.pending = null; }
  }

  render() {
    if (!this.markers.isConnected) this.graph.appendChild(this.markers);
    this.markers.replaceChildren();
    if (this.options.isFFT()) return;
    for (const point of this.points) {
      const axes = this.axes(point.item);
      if (!axes || axes.x.type !== 'linear' || point.item.visible === false) continue;
      const x = axes.x.d2p(this.options.toDisplay(point.item, point.time)), y = axes.y.d2p(point.value);
      if (x < 0 || x > axes.x._length || y < 0 || y > axes.y._length) continue;
      const marker = document.createElement('div'); marker.className = 'peak-marker';
      marker.style.left = `${axes.x._offset + x}px`; marker.style.top = `${axes.y._offset + y}px`;
      marker.style.color = point.label[0] === 'S' ? '#79dbb9' : '#f9ba65';
      const label = document.createElement('span'); label.textContent = point.label; marker.appendChild(label);
      marker.title = `${point.label}: ${point.time.toFixed(7)} recorded ms, ${point.value.toPrecision(6)} A`;
      this.markers.appendChild(marker);
    }
  }
}
