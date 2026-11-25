import assert from "node:assert/strict";
import test from "node:test";
import { displayTime, recordedTime, matchForward, validTransform, matchPeakPairs, highestSample } from "../static/comparison-math.mjs";

test("manual shift/stretch is invertible and pivots around forward start", () => {
  const transform = { scale: 1.7, shift: -3.8 }, start = 38.4;
  assert.equal(displayTime(start, start, transform), start - 3.8);
  for (const time of [0, 2, 38.4, 56.789, 120, 420]) {
    assert(Math.abs(recordedTime(displayTime(time, start, transform), start, transform) - time) < 1e-12);
  }
});

test("matching forward aligns BOTH endpoints even with a transformed reference", () => {
  const source = { capture_start_ms: 38.9, capture_end_ms: 89.7 };
  const target = { capture_start_ms: 38.4, capture_end_ms: 80.8 };
  const targetTransform = { scale: 0.7, shift: 9.5 };
  const transform = matchForward(source, target, targetTransform);
  for (const key of ["capture_start_ms", "capture_end_ms"]) {
    assert(Math.abs(displayTime(source[key], source.capture_start_ms, transform) -
      displayTime(target[key], target.capture_start_ms, targetTransform)) < 1e-12);
  }
  assert.equal(source.capture_start_ms, 38.9);
  assert.equal(source.capture_end_ms, 89.7);
});

test("invalid transforms cannot reverse time or enter a non-finite range", () => {
  for (const scale of [0, -1, NaN, Infinity, 101]) assert(!validTransform({ scale, shift: 0 }));
  for (const shift of [NaN, Infinity, -Infinity, 1e7]) assert(!validTransform({ scale: 1, shift }));
  assert(validTransform({ scale: 1, shift: -100 }));
});

test("four picks align the target to both peaks of an already transformed source", () => {
  const sourceStart = 38, targetStart = 41;
  const sourceTransform = { scale: 1.4, shift: -3 };
  const source = [43, 75].map(time => displayTime(time, sourceStart, sourceTransform));
  const target = [46, 90];
  const result = matchPeakPairs(source, target, targetStart);
  for (let i = 0; i < 2; i++) assert(Math.abs(displayTime(target[i], targetStart, result) - source[i]) < 1e-12);
  assert.deepEqual(sourceTransform, { scale: 1.4, shift: -3 });
  // Picking both pairs right-to-left gives the same positive transform.
  const reversed = matchPeakPairs([...source].reverse(), [...target].reverse(), targetStart);
  assert(Math.abs(reversed.scale - result.scale) < 1e-12);
  assert(Math.abs(reversed.shift - result.shift) < 1e-12);
});

test("duplicate, reversed, nonfinite, and extreme peak pairs cannot corrupt alignment", () => {
  assert.throws(() => matchPeakPairs([1, 1], [2, 3], 0), /distinct/);
  assert.throws(() => matchPeakPairs([1, 2], [3, 3], 0), /distinct/);
  assert.throws(() => matchPeakPairs([1, 2], [3, 2], 0), /same time order/);
  assert.throws(() => matchPeakPairs([1, NaN], [2, 3], 0), /finite/);
  assert.throws(() => matchPeakPairs([1, 1000], [2, 3], 0), /supported/);
});

test("peak snapping finds the highest sample in its crop and breaks ties nearest the click", () => {
  assert.deepEqual(highestSample(new Float64Array([1, 4, 2, 8, 1]), 20, 0.01, 20.01), { time: 20.03, value: 8 });
  assert.deepEqual(highestSample([1, 8, 2, 8, 1], 10, 1, 13.2), { time: 13, value: 8 });
  assert.deepEqual(highestSample([7], 10, 1, 10.4), { time: 10, value: 7 });
  assert.equal(highestSample([], 10, 1, 10), null);
  assert.equal(highestSample([NaN, Infinity], 10, 1, 10), null);
});
