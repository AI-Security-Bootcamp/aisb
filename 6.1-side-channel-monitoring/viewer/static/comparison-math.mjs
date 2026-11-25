// Display-only affine time transforms. Stretch around the original forward
// start, then shift; raw timestamps and measured durations stay untouched.
export function displayTime(time, forwardStart, transform) {
  return forwardStart + (time - forwardStart) * transform.scale + transform.shift;
}

export function recordedTime(time, forwardStart, transform) {
  return forwardStart + (time - forwardStart - transform.shift) / transform.scale;
}

export function matchForward(source, target, targetTransform) {
  const start = displayTime(target.capture_start_ms, target.capture_start_ms, targetTransform);
  const end = displayTime(target.capture_end_ms, target.capture_start_ms, targetTransform);
  return {
    scale: (end - start) / (source.capture_end_ms - source.capture_start_ms),
    shift: start - source.capture_start_ms,
  };
}

export function validTransform(value) {
  return value && Number.isFinite(value.scale) && value.scale >= 0.01 && value.scale <= 100 &&
    Number.isFinite(value.shift) && Math.abs(value.shift) <= 1e6;
}

// Pair S1 with T1 and S2 with T2. Source times are already in display space;
// target times stay in recorded space so an existing target stretch is replaced
// correctly, even when both traces were adjusted before picking their peaks.
export function matchPeakPairs(sourceTimes, targetTimes, targetForwardStart) {
  if (sourceTimes.length !== 2 || targetTimes.length !== 2 ||
      ![...sourceTimes, ...targetTimes, targetForwardStart].every(Number.isFinite)) {
    throw Error('Choose two finite peak times on each waveform.');
  }
  const sourceSpan = sourceTimes[1] - sourceTimes[0], targetSpan = targetTimes[1] - targetTimes[0];
  if (sourceSpan === 0 || targetSpan === 0) throw Error('Choose two distinct peaks on each waveform.');
  if (sourceSpan * targetSpan < 0) throw Error('Pick the corresponding peaks in the same time order on both waveforms.');
  const scale = sourceSpan / targetSpan;
  const transform = { scale, shift: sourceTimes[0] - targetForwardStart - (targetTimes[0] - targetForwardStart) * scale };
  if (!validTransform(transform)) throw Error('These peaks require a stretch outside the supported 0.01×–100× range.');
  return transform;
}

// The API crop is the ±10-pixel neighborhood mapped back into recorded time.
// Search every RMS sample; equal maxima snap to the one nearest the click.
export function highestSample(values, origin, interval, clickedTime) {
  let peak = null;
  for (let i = 0; i < values.length; i++) {
    const value = values[i], time = origin + i * interval;
    if (!Number.isFinite(value)) continue;
    if (!peak || value > peak.value ||
        (value === peak.value && Math.abs(time - clickedTime) < Math.abs(peak.time - clickedTime))) {
      peak = { time, value };
    }
  }
  return peak;
}
