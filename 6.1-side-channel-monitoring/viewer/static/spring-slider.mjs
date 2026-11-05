// A rate control: deflection changes the value continuously. Releasing the
// pointer/key returns the knob to zero, without resetting the trace transform.
export function springSlider(input, apply, finish) {
  let frame = 0, last = 0, active = false;
  let releaseTimer = 0, releasing = false;

  function begin() {
    clearTimeout(releaseTimer);
    releasing = false;
  }
  function tick(now) {
    if (!active) return;
    const dt = Math.min(0.05, Math.max(0, (now - last) / 1000));
    last = now;
    apply(Number(input.value), dt);
    frame = requestAnimationFrame(tick);
  }
  function stop() {
    const wasActive = active;
    active = false;
    cancelAnimationFrame(frame);
    input.value = "0";
    // Firefox's native range control can write its final drag value and emit
    // another input event AFTER pointerup/lostpointercapture. Ignore that event
    // and recenter once more after the native release action has completed.
    releasing = true;
    clearTimeout(releaseTimer);
    releaseTimer = setTimeout(() => {
      input.value = "0";
      releasing = false;
    }, 0);
    if (wasActive) finish();
  }
  input.addEventListener("input", () => {
    if (releasing) return;
    apply(Number(input.value), 1 / 60); // Also makes a brief keyboard tap useful.
    if (!active) { active = true; last = performance.now(); frame = requestAnimationFrame(tick); }
  });
  input.addEventListener("pointerdown", event => {
    begin();
    input.setPointerCapture(event.pointerId);
  });
  input.addEventListener("keydown", begin);
  for (const event of ["pointerup", "pointercancel", "lostpointercapture", "keyup", "blur"]) {
    input.addEventListener(event, stop);
  }
  window.addEventListener("blur", stop);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
  return stop;
}
