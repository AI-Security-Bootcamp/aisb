// Small, dependency-free DSP routines shared by the analysis worker and tests.
// Input values must be uniformly sampled RMS current, NOT plot display points.

export function decimateLowpass(input, factor, cutoff = 0.4 / factor) {
  if (factor === 1 && cutoff >= 0.4) return input.slice();
  // A windowed-sinc FIR suppresses frequencies that would alias after dropping
  // samples. A box average alone is insufficient for spectral/audio analysis.
  const radius = Math.ceil(12 / Math.min(1, cutoff / 0.4));
  const kernel = new Float64Array(2 * radius + 1);
  let total = 0;
  for (let j = -radius; j <= radius; j++) {
    const z = 2 * Math.PI * cutoff * j;
    const sinc = j === 0 ? 2 * cutoff : Math.sin(z) / (Math.PI * j);
    const window = 0.42 + 0.5 * Math.cos(Math.PI * j / radius) + 0.08 * Math.cos(2 * Math.PI * j / radius);
    kernel[j + radius] = sinc * window;
    total += kernel[j + radius];
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= total;
  const result = new Float64Array(Math.ceil(input.length / factor));
  for (let i = 0; i < result.length; i++) {
    const center = i * factor;
    let value = 0;
    for (let j = -radius; j <= radius; j++) {
      // Reflect at the boundary; do not introduce zero-padding discontinuities
      // into the filtered signal. Short signals may require multiple reflections.
      let index = center + j;
      if (input.length === 1) index = 0;
      else if (index < 0 || index >= input.length) {
        const period = 2 * (input.length - 1);
        index = ((index % period) + period) % period;
        if (index >= input.length) index = period - index;
      }
      value += input[index] * kernel[j + radius];
    }
    result[i] = value;
  }
  return result;
}

export function fft(real, imaginary) {
  const size = real.length;
  if (size < 2 || (size & (size - 1)) || imaginary.length !== size) throw Error("FFT length must be a power of two");
  for (let i = 1, j = 0; i < size; i++) {
    let bit = size >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imaginary[i], imaginary[j]] = [imaginary[j], imaginary[i]];
    }
  }
  for (let width = 2; width <= size; width *= 2) {
    const angle = -2 * Math.PI / width, stepReal = Math.cos(angle), stepImaginary = Math.sin(angle);
    for (let start = 0; start < size; start += width) {
      let wr = 1, wi = 0;
      for (let j = 0; j < width / 2; j++) {
        const even = start + j, odd = even + width / 2;
        const tr = wr * real[odd] - wi * imaginary[odd];
        const ti = wr * imaginary[odd] + wi * real[odd];
        real[odd] = real[even] - tr; imaginary[odd] = imaginary[even] - ti;
        real[even] += tr; imaginary[even] += ti;
        const next = wr * stepReal - wi * stepImaginary;
        wi = wr * stepImaginary + wi * stepReal; wr = next;
      }
    }
  }
}

export function windowedFFT(values, size) {
  const real = new Float64Array(size), imaginary = new Float64Array(size);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  let normalization = 0;
  for (let i = 0; i < values.length; i++) {
    const weight = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (values.length - 1));
    real[i] = (values[i] - mean) * weight;
    normalization += weight;
  }
  fft(real, imaginary);
  const amplitude = new Float64Array(size / 2 + 1);
  for (let i = 0; i < amplitude.length; i++) {
    amplitude[i] = Math.hypot(real[i], imaginary[i]) * (i === 0 || i === size / 2 ? 1 : 2) / normalization;
  }
  return amplitude;
}

export function spectrum(values, sampleRate) {
  const size = 2 ** Math.ceil(Math.log2(values.length));
  const amplitude = windowedFFT(values, size);
  // Keep peak amplitudes at their actual frequency bin when a very large FFT
  // would otherwise overload the chart. This reduction is ONLY for plotting.
  const step = Math.max(1, Math.ceil((amplitude.length - 1) / 5000));
  const frequency = [], db = [], magnitude = [];
  for (let first = 1; first < amplitude.length; first += step) {
    let peak = first;
    for (let j = first + 1; j < Math.min(amplitude.length, first + step); j++) {
      if (amplitude[j] > amplitude[peak]) peak = j;
    }
    frequency.push(peak * sampleRate / size);
    magnitude.push(amplitude[peak]);
    db.push(20 * Math.log10(Math.max(amplitude[peak], 1e-12)));
  }
  return { frequency, db, magnitude, fft_size: size, bin_hz: sampleRate / size };
}

export function spectrogram(values, sampleRate) {
  const size = Math.min(1024, 2 ** Math.floor(Math.log2(values.length)));
  const hop = Math.max(size / 4, Math.ceil((values.length - size) / 255));
  const frequency = Array.from({ length: size / 2 }, (_, i) => (i + 1) * sampleRate / size);
  const seconds = [], db = frequency.map(() => []);
  let maximum = -240;
  for (let first = 0; first + size <= values.length; first += hop) {
    const amplitude = windowedFFT(values.subarray(first, first + size), size);
    seconds.push((first + (size - 1) / 2) / sampleRate);
    for (let i = 0; i < frequency.length; i++) {
      const value = 20 * Math.log10(Math.max(amplitude[i + 1], 1e-12));
      db[i].push(value); maximum = Math.max(maximum, value);
    }
  }
  return { frequency, seconds, db, maximum, window_seconds: size / sampleRate };
}

export function prepareAnalysis(input, sampleRate) {
  if (input.length < 8) throw Error("Select at least eight RMS samples for frequency analysis");
  for (const value of input) if (!Number.isFinite(value)) throw Error("Non-finite RMS sample");
  const factor = Math.max(1, Math.ceil(input.length / 131072));
  const values = decimateLowpass(input, factor);
  const rate = sampleRate / factor;
  return { values, sampleRate: rate, factor, originalCount: input.length,
    passbandHz: factor > 1 ? rate * 0.4 : rate / 2,
    spectrum: spectrum(values, rate), spectrogram: spectrogram(values, rate) };
}

export function audioSamples(values, sampleRate, outputRate = 48000, normalize = true) {
  const length = Math.floor(values.length / sampleRate * outputRate);
  if (length > outputRate * 30) throw Error("Playback exceeds 30 seconds. Zoom in or choose a faster playback speed.");
  if (length < 8) throw Error("Viewport is too short to play. Zoom out or slow playback down.");
  // Filter before audio-rate resampling; then interpolate between the filtered
  // samples. Both downsampling stages keep the audible band below Nyquist.
  const factor = Math.max(1, Math.floor(sampleRate / outputRate));
  const filtered = sampleRate > outputRate
    ? decimateLowpass(values, factor, 0.4 * outputRate / sampleRate) : values;
  const rate = sampleRate / factor;
  const mean = filtered.reduce((sum, value) => sum + value, 0) / filtered.length;
  const audio = new Float32Array(length);
  let peak = 0;
  for (let i = 0; i < length; i++) {
    const time = Math.min(filtered.length - 1, i * rate / outputRate);
    const first = Math.floor(time), mix = time - first;
    const value = filtered[first] * (1 - mix) + filtered[Math.min(first + 1, filtered.length - 1)] * mix - mean;
    audio[i] = value; peak = Math.max(peak, Math.abs(value));
  }
  const fade = Math.min(Math.round(outputRate * 0.005), Math.floor(length / 10));
  for (let i = 0; i < length; i++) {
    const ramp = Math.min(1, i / Math.max(1, fade), (length - 1 - i) / Math.max(1, fade));
    audio[i] *= (normalize && peak > 1e-12 ? 0.8 / peak : 1) * ramp;
  }
  return audio;
}

// Place each already-resampled clip on the common DISPLAY timeline. A shifted
// trace must retain its leading/trailing silence when played alone or mixed.
export function mixTimeline(clips, durationSeconds, sampleRate = 48000) {
  if (!(durationSeconds > 0) || durationSeconds > 30) throw Error("Playback is limited to 30 seconds. Zoom in or increase playback speed.");
  const mix = new Float32Array(Math.round(durationSeconds * sampleRate));
  for (const { values, offsetSeconds } of clips) {
    const offset = Math.round(offsetSeconds * sampleRate);
    const begin = Math.max(0, -offset), end = Math.min(values.length, mix.length - offset);
    for (let i = begin; i < end; i++) mix[offset + i] += values[i];
  }
  let peak = 0;
  for (const value of mix) peak = Math.max(peak, Math.abs(value));
  if (peak > 1e-12) for (let i = 0; i < mix.length; i++) mix[i] *= 0.8 / peak;
  return mix;
}
