import assert from "node:assert/strict";
import test from "node:test";
import { fft, decimateLowpass, prepareAnalysis, audioSamples, spectrum, mixTimeline } from "../static/signal-analysis.mjs";
import { estimateAlignment } from "../static/auto-alignment.mjs";
import { validatePreset } from "../static/comparison-presets.mjs";

test("FFT resolves a known sine with calibrated amplitude and removes DC", () => {
  const rate = 65536, n = 16384, frequency = 4096;
  const values = Float64Array.from({ length: n }, (_, i) => 20 + 3 * Math.sin(2 * Math.PI * frequency * i / rate));
  const result = spectrum(values, rate);
  const peak = result.db.indexOf(Math.max(...result.db));
  assert.equal(result.frequency[peak], frequency);
  assert(Math.abs(result.db[peak] - 20 * Math.log10(3)) < 0.001);
  assert(Math.abs(result.magnitude[peak] - 3) < 0.001);
  const constant = spectrum(new Float64Array(256).fill(5), rate);
  assert(constant.db.every(value => value === -240));
  assert.throws(() => fft(new Float64Array(12), new Float64Array(12)));
});

test("stretch lowers audio pitch and timeline mixing preserves shifts, gaps and relative amplitude", () => {
  const rate = 96000;
  const values = Float64Array.from({ length: 9600 }, (_, i) => 2 + Math.sin(2 * Math.PI * 1000 * i / rate));
  const audio = audioSamples(values, rate / 2, 48000, false);
  assert.equal(audio.length, 9600); // 2× time stretch, now 200 ms.
  const s = spectrum(audio, 48000), peak = s.db.indexOf(Math.max(...s.db));
  assert(Math.abs(s.frequency[peak] - 500) < 4);
  const mixed = mixTimeline([
    { values: new Float32Array(20).fill(1), offsetSeconds: 0.2 },
    { values: new Float32Array(20).fill(2), offsetSeconds: 0.3 },
  ], 1, 100);
  assert.equal(mixed.length, 100);
  assert(mixed.slice(0, 20).every(value => value === 0));
  assert(Math.abs(mixed[25] - 0.8 / 3) < 1e-7);
  assert(Math.abs(mixed[35] - 0.8) < 1e-7);
  assert(Math.abs(mixed[45] - 1.6 / 3) < 1e-7);
  assert(mixed.slice(50).every(value => value === 0));
});

test("spectral peak matching recovers a known stretch and correlation recovers its shift", () => {
  const sampleRate = 100000, duration = 0.05;
  const signal = t => 3 + (1 + 0.3 * Math.sin(2 * Math.PI * 41 * t)) * Math.sin(2 * Math.PI * 710 * t) +
    0.7 * Math.sin(2 * Math.PI * 1330 * t + 0.3) + 0.5 * Math.sin(2 * Math.PI * 2170 * t + 0.8) +
    0.3 * Math.sin(2 * Math.PI * 3910 * t + 1.7);
  const target = { sampleRate, originSeconds: 0, durationSeconds: duration,
    values: Float64Array.from({ length: duration * sampleRate }, (_, i) => signal(i / sampleRate)) };
  for (const [ratio, offset] of [[1.25, 0.001], [0.8, -0.0008]]) {
    const source = { sampleRate, originSeconds: 0, durationSeconds: duration / ratio,
      values: Float64Array.from({ length: duration / ratio * sampleRate }, (_, i) => signal(i / sampleRate * ratio + offset)) };
    const result = estimateAlignment(source, target);
    assert(result.accepted);
    assert(Math.abs(result.ratio - ratio) < 0.002, `Wrong stretch: ${result.ratio}`);
    assert(Math.abs(result.offsetSeconds - offset) <= result.gridSeconds * 1.5, `Wrong shift: ${result.offsetSeconds}`);
    assert(result.correlation > 0.98);
  }
  assert.equal(estimateAlignment({ ...target, values: new Float64Array(5000).fill(3) }, target).accepted, false);
});

test("presets keep old files compatible and save in-place FFT settings", () => {
  const preset = { version: 1, name: 'FFT overlay', reference: 'one', analysisTrace: 'all',
    transforms: { one: { scale: 2, shift: 3 } }, rmsWindow: '0.001', range: [0, 220],
    overlay: true, fftMode: true, fftLog: false };
  assert.deepEqual(validatePreset(preset, ['one']), preset);
  assert.throws(() => validatePreset({ ...preset, fftMode: 'true' }, ['one']));
});

test("low-pass decimation retains passband and suppresses aliased high-frequency tones", () => {
  const rate = 1_000_000, low = 15625, high = 406250, factor = 16;
  const samples = Float64Array.from({ length: 65536 }, (_, i) =>
    7 + 2 * Math.sin(2 * Math.PI * low * i / rate) + 5 * Math.cos(2 * Math.PI * high * i / rate));
  const filtered = decimateLowpass(samples, factor);
  const middle = filtered.slice(128, -128), length = middle.length;
  const tone = frequency => {
    let real = 0, imaginary = 0;
    for (let i = 0; i < length; i++) {
      real += (middle[i] - 7) * Math.cos(2 * Math.PI * frequency * (i + 128) / (rate / factor));
      imaginary += (middle[i] - 7) * Math.sin(2 * Math.PI * frequency * (i + 128) / (rate / factor));
    }
    return 2 * Math.hypot(real, imaginary) / length;
  };
  assert(Math.abs(tone(low) - 2) < 0.01);
  assert(tone(31250) < 0.001, 'High-frequency input aliased into the retained band');
  for (const value of decimateLowpass(new Float64Array(100).fill(6), 10)) assert(Math.abs(value - 6) < 1e-12);
});

test("analysis bounds sample count and spectrogram identifies a persistent tone", () => {
  const rate = 1_000_000;
  const samples = Float64Array.from({ length: 300000 }, (_, i) => 3 + Math.sin(2 * Math.PI * 10000 * i / rate));
  const result = prepareAnalysis(samples, rate);
  assert(result.values.length <= 131072);
  assert.equal(result.factor, 3);
  assert.equal(result.sampleRate, rate / 3);
  const s = result.spectrogram;
  assert(s.seconds.length <= 256);
  const column = s.db.map(row => row[Math.floor(s.seconds.length / 2)]);
  assert(Math.abs(s.frequency[column.indexOf(Math.max(...column))] - 10000) <= rate / 3 / 1024);
  assert.throws(() => prepareAnalysis(new Float64Array(3), rate));
});

test("audio has expected duration, bounded gain, faded edges and preserves audible pitch", () => {
  const rate = 256000;
  const samples = Float64Array.from({ length: rate / 4 }, (_, i) =>
    10 + Math.sin(2 * Math.PI * 1000 * i / rate) + Math.sin(2 * Math.PI * 80000 * i / rate));
  const audio = audioSamples(samples, rate);
  assert.equal(audio.length, 12000);
  assert(Math.max(...audio) <= 0.800001);
  assert(Math.min(...audio) >= -0.800001);
  assert.equal(audio[0], 0); assert.equal(Math.abs(audio.at(-1)), 0);
  const result = spectrum(audio, 48000);
  const peak = result.db.indexOf(Math.max(...result.db));
  assert(Math.abs(result.frequency[peak] - 1000) < 4);
  assert.throws(() => audioSamples(new Float64Array(100000), 1000), /30 seconds/);
});

test("named presets round-trip settings and reject malformed imports", () => {
  const ids = ['first', 'second'];
  const preset = { version: 1, name: 'Forward comparison', transforms: {
    first: { scale: 1, shift: 2 }, second: { scale: 1.3, shift: -8.4 } },
    reference: 'first', analysisTrace: 'second', rmsWindow: '0.0001', range: [38, 80], overlay: true };
  assert.deepEqual(validatePreset(JSON.parse(JSON.stringify(preset)), ids), preset);
  for (const bad of [{ name: '' }, { range: [80, 38] }, { version: 2 }, { reference: 'unknown' },
    { transforms: { first: { scale: 0, shift: 0 } } }]) {
    assert.throws(() => validatePreset({ ...preset, ...bad }, ids));
  }
  const polluted = JSON.parse(JSON.stringify(preset).replace('"version":1', '"__proto__":{"polluted":true},"version":1'));
  assert(!Object.hasOwn(validatePreset(polluted, ids), '__proto__'));
  assert.equal({}.polluted, undefined);
});
