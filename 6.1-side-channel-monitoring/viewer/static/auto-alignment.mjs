import { decimateLowpass, windowedFFT } from "./signal-analysis.mjs";

function prepare(input) {
  const factor = Math.max(1, Math.ceil(input.values.length / 16384));
  return { ...input, values: decimateLowpass(input.values, factor), sampleRate: input.sampleRate / factor };
}

function peaks(signal, minimum, maximum) {
  const size = 2 ** Math.ceil(Math.log2(signal.values.length));
  const amplitudes = windowedFFT(signal.values, size), bin = signal.sampleRate / size;
  const found = [];
  for (let i = Math.max(2, Math.ceil(minimum / bin)); i < Math.min(amplitudes.length - 1, maximum / bin); i++) {
    if (amplitudes[i] <= amplitudes[i - 1] || amplitudes[i] < amplitudes[i + 1]) continue;
    // Parabolic interpolation in log magnitude reduces integer-bin bias.
    const left = Math.log(Math.max(amplitudes[i - 1], 1e-15));
    const center = Math.log(Math.max(amplitudes[i], 1e-15));
    const right = Math.log(Math.max(amplitudes[i + 1], 1e-15));
    const delta = Math.max(-0.5, Math.min(0.5, 0.5 * (left - right) / (left - 2 * center + right || 1)));
    found.push({ frequency: (i + delta) * bin, amplitude: amplitudes[i] });
  }
  found.sort((a, b) => b.amplitude - a.amplitude);
  const selected = [];
  for (const peak of found) {
    if (peak.amplitude < (found[0]?.amplitude || 0) * 0.03) continue;
    if (selected.every(other => Math.abs(other.frequency - peak.frequency) > 2 * bin)) selected.push(peak);
    if (selected.length === 14) break;
  }
  const total = selected.reduce((sum, peak) => sum + peak.amplitude, 0);
  return { bin, peaks: selected.map(peak => ({ ...peak, weight: peak.amplitude / (total || 1) })) };
}

function peakScore(source, target, ratio) {
  let score = 0, matches = 0;
  const pairs = [];
  const used = new Set();
  for (const src of source.peaks) {
    const candidates = target.peaks.filter(ref => !used.has(ref));
    candidates.sort((a, b) => Math.abs(src.frequency / ratio - a.frequency) - Math.abs(src.frequency / ratio - b.frequency));
    const ref = candidates[0];
    if (!ref) continue;
    const tolerance = Math.max(1.5 * source.bin / ratio, 1.5 * target.bin, ref.frequency * 0.015);
    const error = Math.abs(src.frequency / ratio - ref.frequency) / tolerance;
    if (error < 1.5) {
      used.add(ref); matches++;
      const weight = Math.sqrt(src.weight * ref.weight);
      score += weight * Math.exp(-0.5 * error * error);
      pairs.push({ source: src.frequency, target: ref.frequency, weight });
    }
  }
  return { score, matches, pairs };
}

function interpolate(signal, time) {
  const index = (time - signal.originSeconds) * signal.sampleRate;
  if (index < 0 || index > signal.values.length - 1) return NaN;
  const first = Math.floor(index), mix = index - first;
  return signal.values[first] * (1 - mix) + signal.values[Math.min(first + 1, signal.values.length - 1)] * mix;
}

function correlate(source, target, ratio) {
  const count = 2048, dt = target.durationSeconds / count;
  // Anti-alias before evaluating either trace on the common correlation grid.
  function smooth(signal, effectiveRate) {
    const factor = Math.max(1, Math.floor(signal.sampleRate / effectiveRate));
    return { ...signal, values: decimateLowpass(signal.values, factor,
      Math.min(0.4, 0.4 * effectiveRate / signal.sampleRate)), sampleRate: signal.sampleRate / factor };
  }
  const ref = smooth(target, 1 / dt), src = smooth(source, ratio / dt);
  const a = Float64Array.from({ length: count }, (_, i) => interpolate(ref, (i + 0.5) * dt));
  const limit = Math.floor(count * 0.12);
  const b = Float64Array.from({ length: count + 2 * limit }, (_, i) => interpolate(src, (i - limit + 0.5) * dt / ratio));
  let best = { correlation: -1, lag: 0, quality: -1 };
  for (let lag = -limit; lag <= limit; lag++) {
    let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (let i = 0; i < count; i++) {
      const x = a[i], y = b[i - lag + limit];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      n++; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
    }
    if (n < count * 0.7) continue;
    const denominator = Math.sqrt(Math.max(0, (sxx - sx * sx / n) * (syy - sy * sy / n)));
    if (denominator < 1e-12) continue;
    const correlation = (sxy - sx * sy / n) / denominator;
    const quality = correlation - 0.01 * Math.abs(lag) / limit;
    if (quality > best.quality) best = { correlation, lag, quality };
  }
  return { correlation: best.correlation, offsetSeconds: best.lag * dt, gridSeconds: dt };
}

export function estimateAlignment(sourceInput, targetInput) {
  const source = prepare(sourceInput), target = prepare(targetInput);
  if (source.values.length < 32 || target.values.length < 32) return { accepted: false, reason: "Too few forward samples" };
  const prior = target.durationSeconds / source.durationSeconds;
  const minimum = 2 / Math.min(source.durationSeconds, target.durationSeconds);
  const maximum = Math.min(25000, source.sampleRate * 0.35, target.sampleRate * 0.35);
  const src = peaks(source, minimum, maximum), ref = peaks(target, minimum, maximum);
  if (src.peaks.length < 3 || ref.peaks.length < 3 || Math.max(...src.peaks.map(p => p.amplitude)) < 1e-8 ||
      Math.max(...ref.peaks.map(p => p.amplitude)) < 1e-8) return { accepted: false, reason: "No clear matching FFT peaks" };
  const candidates = [];
  for (const a of src.peaks) for (const b of ref.peaks) {
    let ratio = a.frequency / b.frequency;
    if (ratio < prior * 0.55 || ratio > prior * 1.8) continue;
    let fit = peakScore(src, ref, ratio);
    // Refine the scale using all matched peaks, rather than one frequency bin.
    if (fit.matches >= 3) {
      ratio = fit.pairs.reduce((s, p) => s + p.weight * p.source * p.target, 0) /
        fit.pairs.reduce((s, p) => s + p.weight * p.target * p.target, 0);
      fit = peakScore(src, ref, ratio);
      if (fit.matches >= 3 && candidates.every(c => Math.abs(Math.log(c.ratio / ratio)) > 0.003)) {
        candidates.push({ ratio, ...fit });
      }
    }
  }
  candidates.sort((a, b) => b.score - a.score);
  let best = null;
  for (const candidate of candidates.slice(0, 6)) {
    const shift = correlate(source, target, candidate.ratio);
    const quality = 0.45 * candidate.score + 0.55 * Math.max(0, shift.correlation);
    if (!best || quality > best.quality) best = { ...candidate, ...shift, quality };
  }
  if (!best || best.score < 0.3 || best.correlation < 0.25) return {
    accepted: false, reason: "Weak spectral/waveform match; alignment left unchanged", candidate: best,
  };
  return { accepted: true, ratio: best.ratio, offsetSeconds: best.offsetSeconds,
    correlation: best.correlation, matchedPeaks: best.matches, spectralScore: best.score,
    gridSeconds: best.gridSeconds, pairs: best.pairs };
}
