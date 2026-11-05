import assert from "node:assert/strict";
import test from "node:test";
import { waveform, rmsSampleRange } from "./waveform.mjs";

const count = 2200013, intervalNs = 400, gain = 0.025;
const raw = Int16Array.from({ length: count }, (_, i) => (i * 97) % 2001 - 1000);
// A spike just outside one tested interval must not contaminate its envelope.
raw[122] = -32760;
raw[1900123] = 32760;
const objects = new Map([["raw", new Uint8Array(raw.buffer)]]);
const capture = {
  summary: { samples: count, interval_ns: intervalNs, current_A_per_adc_count: gain, duration_ms: count * intervalNs * 1e-6 },
  raw_key: "raw", raw_offset: 0, envelopes: {}, rms: { "0.1": {} },
  rms_origin_ms: 0.0005, rms_interval_ms: 0.001, rms_count: 880005,
};
for (const stride of [256, 4096, 65536, 1048576]) {
  const pairs = new Int16Array(Math.ceil(count / stride) * 2);
  for (let begin = 0; begin < count; begin += stride) {
    let low = Infinity, high = -Infinity;
    for (let i = begin; i < Math.min(count, begin + stride); i++) {
      low = Math.min(low, raw[i]); high = Math.max(high, raw[i]);
    }
    pairs[(begin / stride) * 2] = low;
    pairs[(begin / stride) * 2 + 1] = high;
  }
  capture.envelopes[stride] = `envelope-${stride}`;
  objects.set(capture.envelopes[stride], new Uint8Array(pairs.buffer));
}
const rmsValue = (i) => 2 + (i % 173) / 19;
for (const stride of [1, 4, 16, 64, 256, 1024]) {
  const values = Float64Array.from({ length: Math.ceil(capture.rms_count / stride) }, (_, i) => rmsValue(i * stride));
  capture.rms["0.1"][stride] = `rms-${stride}`;
  objects.set(`rms-${stride}`, new Uint8Array(values.buffer));
}
const reads = [];
const bucket = {
  async get(key, { range: { offset, length } }) {
    reads.push({ key, offset, length });
    const data = objects.get(key);
    if (!data) return null;
    assert(offset >= 0 && length > 0 && offset + length <= data.byteLength);
    return { arrayBuffer: async () => data.slice(offset, offset + length).buffer };
  },
};
const dt = intervalNs * 1e-6;

test("fine zoom returns every calibrated raw sample", async () => {
  const first = 113, stop = 531;
  const result = await waveform(bucket, capture, (first - 0.25) * dt, (stop - 0.25) * dt);
  assert.equal(result.kind, "raw");
  assert.equal(result.y.length, stop - first);
  for (let i = 0; i < result.y.length; i++) {
    assert.equal(result.y[i], raw[first + i] * gain);
    assert.equal(result.x_ms[i], (first + i) * dt);
  }
});

test("wide zoom preserves exact extrema and excludes spikes outside both edges", async () => {
  reads.length = 0;
  const first = 123, stop = 1900123;
  const result = await waveform(bucket, capture, (first - 0.25) * dt, (stop - 0.25) * dt);
  assert.equal(result.kind, "envelope");
  const stride = 4096, begin = Math.floor(first / stride);
  for (let i = 0; i < result.low.length; i++) {
    const start = Math.max(first, (begin + i) * stride), end = Math.min(stop, (begin + i + 1) * stride);
    let low = Infinity, high = -Infinity;
    for (let j = start; j < end; j++) { low = Math.min(low, raw[j]); high = Math.max(high, raw[j]); }
    assert.equal(result.low[i], low * gain);
    assert.equal(result.high[i], high * gain);
  }
  assert(reads.filter((r) => r.key === "raw").every((r) => r.length <= 1024));
});

test("the partial last bucket includes the final sample", async () => {
  const result = await waveform(bucket, capture, 0, capture.summary.duration_ms);
  const start = Math.floor((count - 1) / 4096) * 4096;
  assert.equal(result.low.at(-1), Math.min(...raw.slice(start)) * gain);
  assert.equal(result.high.at(-1), Math.max(...raw.slice(start)) * gain);
});

test("RMS display levels return original values at their original timestamps", async () => {
  const result = await waveform(bucket, capture, 0.0001, 879, "0.1");
  assert.equal(result.kind, "rms");
  assert(result.y.length <= 6000);
  for (let i = 0; i < result.y.length; i++) {
    const source = Math.round((result.x_ms[i] - capture.rms_origin_ms) / capture.rms_interval_ms);
    assert.equal(result.y[i], rmsValue(source));
    assert(result.x_ms[i] >= 0.0001 && result.x_ms[i] <= 879);
  }
});

test("invalid and empty ranges fail before any R2 read", async () => {
  reads.length = 0;
  for (const [lo, hi, mode] of [[NaN, 2, "raw"], [0, Infinity, "raw"], [2, 1, "raw"], [0, 0, "raw"], [0, 1, "bad"]]) {
    await assert.rejects(waveform(bucket, capture, lo, hi, mode), RangeError);
  }
  assert.equal(reads.length, 0);
});

// A fine-RMS series with isolated extrema at boundaries. The 24-byte peak
// records use the same published binary layout as the Python export.
const fineValues = Float64Array.from({ length: 20011 }, (_, i) => 2 + (i * 37 % 997) / 97);
fineValues[123] = 300;
fineValues[19407] = 400;
const curve = { window_ms: 0.0001, interval_ms: 0.0001, origin_ms: 0.0000498,
  count: fineValues.length, samples_per_window: 250, values_key: "fine-rms", peaks: {} };
objects.set(curve.values_key, new Uint8Array(fineValues.buffer));
for (const stride of [16, 64, 256, 1024]) {
  const view = new DataView(new ArrayBuffer(Math.ceil(curve.count / stride) * 24));
  for (let start = 0; start < curve.count; start += stride) {
    let low = start, high = start;
    for (let i = start; i < Math.min(curve.count, start + stride); i++) {
      if (fineValues[i] < fineValues[low]) low = i;
      if (fineValues[i] > fineValues[high]) high = i;
    }
    const offset = start / stride * 24;
    view.setUint32(offset, low, true); view.setFloat64(offset + 4, fineValues[low], true);
    view.setUint32(offset + 12, high, true); view.setFloat64(offset + 16, fineValues[high], true);
  }
  curve.peaks[stride] = `fine-peaks-${stride}`;
  objects.set(curve.peaks[stride], new Uint8Array(view.buffer));
}
const fineCapture = { ...capture, fine_rms: { "0.0001": curve } };
const rmsTime = index => curve.origin_ms + index * curve.interval_ms;

test("100 ns fine zoom returns every bin at its original midpoint", async () => {
  const result = await waveform(bucket, fineCapture, rmsTime(18.5), rmsTime(102.5), "0.0001");
  assert.equal(result.display_reduction, "none");
  assert.equal(result.rms_samples_per_window, 250);
  assert.equal(result.y.length, 84);
  for (let i = 0; i < result.y.length; i++) {
    assert.equal(result.x_ms[i], rmsTime(i + 19));
    assert.equal(result.y[i], fineValues[i + 19]);
  }
});

test("fine RMS overview preserves peaks and excludes off-screen boundary spikes", async () => {
  const begin = 124, end = 19407;
  const result = await waveform(bucket, fineCapture, rmsTime(begin - 0.25), rmsTime(end - 0.25), "0.0001");
  assert.equal(result.display_reduction, "minmax");
  assert(result.y.length <= 6000);
  assert.equal(Math.max(...result.y), Math.max(...fineValues.slice(begin, end)));
  assert.equal(Math.min(...result.y), Math.min(...fineValues.slice(begin, end)));
  for (let i = 0; i < result.y.length; i++) {
    const source = Math.round((result.x_ms[i] - curve.origin_ms) / curve.interval_ms);
    assert(source >= begin && source < end);
    assert.equal(result.y[i], fineValues[source]);
    if (i) assert(result.x_ms[i] > result.x_ms[i - 1]);
  }
});

test("fine RMS includes the final partial DISPLAY bucket and handles no visible midpoint", async () => {
  const wide = await waveform(bucket, fineCapture, 0, rmsTime(curve.count), "0.0001");
  assert.equal(wide.y.length <= 6000, true);
  assert.equal(Math.max(...wide.y), 400);
  assert(wide.x_ms.every(time => time <= rmsTime(curve.count - 1)));
  const empty = await waveform(bucket, fineCapture, rmsTime(19.1), rmsTime(19.9), "0.0001");
  assert.deepEqual(empty.y, []);
});

test("FFT download selects uniform fine and legacy RMS arrays without display reduction", () => {
  const fine = rmsSampleRange(fineCapture, rmsTime(18.5), rmsTime(102.5), "0.0001");
  assert.equal(fine.count, 84);
  assert.equal(fine.offset, 19 * 8);
  assert.equal(fine.length, 84 * 8);
  assert.equal(fine.key, curve.values_key);
  assert.equal(fine.origin_ms, rmsTime(19));
  const legacy = rmsSampleRange(capture, 0, 1, "0.1");
  assert.equal(legacy.key, capture.rms['0.1'][1]);
  assert.equal(legacy.interval_ms, capture.rms_interval_ms);
  assert.throws(() => rmsSampleRange(capture, 0, 1, 'raw'), RangeError);
  assert.throws(() => rmsSampleRange(capture, 2, 1, '0.1'), RangeError);
});
