// Query immutable R2 arrays. All time values and current gains come from the
// original capture metadata; this module never adjusts clock alignment.
const RAW_POINTS = 12000;
const DISPLAY_POINTS = 6000;

// Frequency analysis needs a uniform series, never the extrema-selected points
// used for plotting. Return a byte range into the original RMS array.
export function rmsSampleRange(capture, lo, hi, mode) {
  if (![lo, hi].every(Number.isFinite) || hi <= lo) throw new RangeError("Select a finite, nonempty interval");
  let curve = capture.fine_rms?.[mode];
  if (!curve && Object.hasOwn(capture.rms, mode)) curve = {
    origin_ms: capture.rms_origin_ms, interval_ms: capture.rms_interval_ms,
    count: capture.rms_count, values_key: capture.rms[mode][1], window_ms: Number(mode),
  };
  if (!curve) throw new RangeError("Unknown RMS window");
  const first = Math.max(0, Math.ceil((Math.max(0, lo) - curve.origin_ms) / curve.interval_ms));
  const stop = Math.min(curve.count, Math.floor((Math.min(capture.summary.duration_ms, hi) - curve.origin_ms) / curve.interval_ms) + 1);
  const count = Math.max(0, stop - first);
  if (count > 8_000_000) throw new RangeError("Zoom in to analyse at most eight million RMS samples");
  return { key: curve.values_key, offset: first * 8, length: count * 8,
    count, origin_ms: curve.origin_ms + first * curve.interval_ms,
    interval_ms: curve.interval_ms, window_ms: curve.window_ms };
}

async function read(bucket, key, offset, length) {
  const object = await bucket.get(key, { range: { offset, length } });
  if (!object) throw Error("Recording data is unavailable");
  const buffer = await object.arrayBuffer();
  if (buffer.byteLength !== length) throw Error("Incomplete recording data");
  return new DataView(buffer);
}

async function rawCounts(bucket, capture, first, stop) {
  return read(bucket, capture.raw_key, capture.raw_offset + first * 2, (stop - first) * 2);
}

// Use complete 256-sample bins and exact raw values at the two edges. This
// avoids including a spike outside the viewport or reading a million-sample
// boundary bucket into the Worker just to find its extrema.
async function extrema(bucket, capture, first, stop) {
  const stride = 256;
  const left = Math.ceil(first / stride), right = Math.floor(stop / stride);
  const tasks = [];
  if (left >= right) {
    tasks.push(rawCounts(bucket, capture, first, stop));
  } else {
    if (first < left * stride) tasks.push(rawCounts(bucket, capture, first, left * stride));
    tasks.push(read(bucket, capture.envelopes[stride], left * 4, (right - left) * 4));
    if (right * stride < stop) tasks.push(rawCounts(bucket, capture, right * stride, stop));
  }
  let low = Infinity, high = -Infinity;
  for (const view of await Promise.all(tasks)) {
    for (let i = 0; i < view.byteLength; i += 2) {
      const value = view.getInt16(i, true);
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
  }
  return [low, high];
}

async function fineRMS(bucket, curve, lo, hi) {
  const begin = Math.max(0, Math.ceil((lo - curve.origin_ms) / curve.interval_ms));
  const end = Math.min(curve.count, Math.floor((hi - curve.origin_ms) / curve.interval_ms) + 1);
  const points = [];
  const count = Math.max(0, end - begin);
  let stride = 1;
  if (count <= DISPLAY_POINTS) {
    if (count) {
      const values = await read(bucket, curve.values_key, begin * 8, count * 8);
      for (let i = 0; i < count; i++) points.push([begin + i, values.getFloat64(i * 8, true)]);
    }
  } else {
    const levels = Object.keys(curve.peaks).map(Number).sort((a, b) => a - b);
    stride = levels.find(value => 2 * (Math.ceil(end / value) - Math.floor(begin / value)) <= DISPLAY_POINTS);
    if (!stride) throw new RangeError("Recording needs a coarser RMS display level");
    const first = Math.floor(begin / stride), last = Math.ceil(end / stride);
    const values = await read(bucket, curve.peaks[stride], first * 24, (last - first) * 24);
    // Recompute partially visible display buckets from the original fine RMS,
    // so a spike just outside the viewport cannot leak into this view.
    const edges = new Map();
    await Promise.all([...new Set([first, last - 1])].map(async index => {
      const start = Math.max(begin, index * stride);
      const stop = Math.min(end, (index + 1) * stride);
      if (start === index * stride && stop === Math.min(curve.count, (index + 1) * stride)) return;
      const raw = await read(bucket, curve.values_key, start * 8, (stop - start) * 8);
      let low = [start, Infinity], high = [start, -Infinity];
      for (let j = 0; j < stop - start; j++) {
        const value = raw.getFloat64(j * 8, true);
        if (value < low[1]) low = [start + j, value];
        if (value > high[1]) high = [start + j, value];
      }
      edges.set(index, [low, high]);
    }));
    for (let i = first; i < last; i++) {
      const offset = (i - first) * 24;
      const pair = edges.get(i) ?? [
        [values.getUint32(offset, true), values.getFloat64(offset + 4, true)],
        [values.getUint32(offset + 12, true), values.getFloat64(offset + 16, true)],
      ];
      pair.sort((a, b) => a[0] - b[0]);
      points.push(pair[0]);
      if (pair[1][0] !== pair[0][0]) points.push(pair[1]);
    }
  }
  return {
    kind: "rms", x_ms: points.map(([index]) => curve.origin_ms + index * curve.interval_ms),
    y: points.map(([, value]) => value), rms_window_ms: curve.window_ms,
    rms_interval_ms: curve.interval_ms, rms_samples_per_window: curve.samples_per_window,
    rms_bins_in_range: count, display_stride: stride, display_reduction: stride === 1 ? "none" : "minmax",
    description: `${curve.window_ms} ms RMS · ${curve.samples_per_window} raw samples/window · ` +
      (stride === 1 ? "every RMS bin" : `peak-preserving display, up to ${stride} RMS bins/bucket; zoom for every bin`),
  };
}

export async function waveform(bucket, capture, lo, hi, mode = "raw") {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) throw new RangeError("Time limits must be finite");
  const s = capture.summary, dt = s.interval_ns * 1e-6, gain = s.current_A_per_adc_count;
  lo = Math.max(0, lo);
  hi = Math.min(s.duration_ms, hi);
  if (hi <= lo) throw new RangeError("Select a nonempty interval inside the recording");
  const first = Math.min(s.samples, Math.ceil(lo / dt));
  const stop = Math.min(s.samples, Math.ceil(hi / dt));
  if (stop <= first) throw new RangeError("The interval contains no samples");
  const result = { lo_ms: lo, hi_ms: hi, raw_samples_in_range: stop - first };

  if (mode !== "raw") {
    if (capture.fine_rms && Object.hasOwn(capture.fine_rms, mode)) {
      return { ...result, ...await fineRMS(bucket, capture.fine_rms[mode], lo, hi) };
    }
    if (!Object.hasOwn(capture.rms, mode)) throw new RangeError("Unknown RMS window");
    const begin = Math.max(0, Math.ceil((lo - capture.rms_origin_ms) / capture.rms_interval_ms));
    const end = Math.min(capture.rms_count, Math.floor((hi - capture.rms_origin_ms) / capture.rms_interval_ms) + 1);
    if (begin >= end) return { ...result, kind: "rms", x_ms: [], y: [], description: "No RMS bins in this interval; choose raw samples" };
    const levels = Object.keys(capture.rms[mode]).map(Number).sort((a, b) => a - b);
    const stride = levels.find((value) => Math.ceil((end - begin) / value) <= 65536) ?? levels.at(-1);
    const start = Math.ceil(begin / stride), last = Math.ceil(end / stride);
    const values = await read(bucket, capture.rms[mode][stride], start * 8, (last - start) * 8);
    const step = Math.max(1, Math.ceil((last - start) / DISPLAY_POINTS));
    const x_ms = [], y = [];
    for (let i = 0; i < last - start; i += step) {
      x_ms.push(capture.rms_origin_ms + (start + i) * stride * capture.rms_interval_ms);
      y.push(values.getFloat64(i * 8, true));
    }
    return { ...result, kind: "rms", x_ms, y, description: `${mode} ms RMS; original recorded values; display every ${stride * step} bins` };
  }

  if (stop - first <= RAW_POINTS) {
    const values = await rawCounts(bucket, capture, first, stop);
    const x_ms = [], y = [];
    for (let i = 0; i < stop - first; i++) {
      x_ms.push((first + i) * dt);
      y.push(values.getInt16(i * 2, true) * gain);
    }
    return { ...result, kind: "raw", x_ms, y, description: `Every raw sample · ${s.interval_ns} ns spacing · no averaging or decimation` };
  }

  const levels = Object.keys(capture.envelopes).map(Number).sort((a, b) => a - b);
  const stride = levels.find((value) => Math.ceil((stop - first) / value) + 1 <= DISPLAY_POINTS) ?? levels.at(-1);
  const begin = Math.floor(first / stride), end = Math.ceil(stop / stride);
  const values = await read(bucket, capture.envelopes[stride], begin * 4, (end - begin) * 4);
  const x_ms = [], low = [], high = [];
  for (let i = 0; i < end - begin; i++) {
    const start = Math.max(first, (begin + i) * stride);
    const last = Math.min(stop, (begin + i + 1) * stride);
    x_ms.push((start + last - 1) * 0.5 * dt);
    low.push(values.getInt16(i * 4, true) * gain);
    high.push(values.getInt16(i * 4 + 2, true) * gain);
  }
  await Promise.all([...new Set([0, end - begin - 1])].map(async (i) => {
    const start = Math.max(first, (begin + i) * stride);
    const last = Math.min(stop, (begin + i + 1) * stride);
    if (start % stride === 0 && (last % stride === 0 || last === s.samples)) return;
    const [minimum, maximum] = await extrema(bucket, capture, start, last);
    low[i] = minimum * gain;
    high[i] = maximum * gain;
  }));
  return { ...result, kind: "envelope", x_ms, low, high, description: `Exact min/max envelope · up to ${stride.toLocaleString("en-US")} raw samples per bucket · zoom in for individual samples` };
}
