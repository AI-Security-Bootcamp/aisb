const graph = document.getElementById("graph"),
  status = document.getElementById("status"),
  mode = document.getElementById("mode");
const captureId = new URLSearchParams(location.search).get("capture");
const captureQuery = captureId
  ? "capture=" + encodeURIComponent(captureId)
  : "";
const captureSuffix = captureQuery ? "?" + captureQuery : "";
document.getElementById("details-link").href += captureSuffix;
for (const a of document.querySelectorAll('a[href^="/data/"]'))
  a.href += captureSuffix;
fetch("/api/catalog")
  .then((r) => r.json())
  .then((entries) => {
    const picker = document.getElementById("recording");
    for (const item of entries) {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.title;
      picker.appendChild(option);
    }
    if (captureId) picker.value = captureId;
    else if (entries.some((e) => e.default))
      picker.value = entries.find((e) => e.default).id;
    document.getElementById("recording-picker").hidden = entries.length < 2;
    picker.onchange = () =>
      (location.search = "capture=" + encodeURIComponent(picker.value));
  });
let summary,
  range = [0, 0],
  pending,
  serial = 0;
const colors = {
  forward: "#5aabff",
  backward: "#bb85ed",
  optimizer: "#ffaa53",
};
async function draw(lo, hi) {
  if (!summary || !summary.has_raw) return;
  lo = Math.max(0, lo);
  hi = Math.min(summary.duration_ms, hi);
  if (hi <= lo) return;
  range = [lo, hi];
  const id = ++serial;
  if (pending) pending.abort();
  pending = new AbortController();
  status.textContent = "Loading selected time range…";
  try {
    const r = await fetch(
      `/api/waveform?lo_ms=${lo}&hi_ms=${hi}&mode=${mode.value}&${captureQuery}`,
      { signal: pending.signal },
    );
    if (!r.ok) throw Error(await r.text());
    const d = await r.json();
    if (id !== serial) return;
    const width = hi - lo,
      scale = width < 0.01 ? 1e6 : width < 1 ? 1000 : 1,
      offset = scale === 1 ? 0 : lo;
    const unit = scale === 1 ? "ms" : scale === 1000 ? "µs" : "ns",
      tx = (x) => (x - offset) * scale,
      x = d.x_ms.map(tx);
    let traces;
    if (d.kind === "envelope")
      traces = [
        {
          x,
          y: d.low,
          type: "scatter",
          mode: "lines",
          line: { width: 0.6, color: "#7ebdff" },
          name: "Minimum",
        },
        {
          x,
          y: d.high,
          type: "scatter",
          mode: "lines",
          line: { width: 0.6, color: "#7ebdff" },
          fill: "tonexty",
          fillcolor: "rgba(106,176,255,.3)",
          name: "Maximum",
        },
      ];
    else
      traces = [
        {
          x,
          y: d.y,
          type: "scatter",
          mode: x.length < 800 ? "lines+markers" : "lines",
          marker: { size: 3 },
          line: { width: 1, color: "#85c7ff" },
          name: d.kind === "raw" ? "Raw current" : `RMS ${mode.value} ms`,
        },
      ];
    const selected = summary.phases.filter(
      (p) =>
        colors[p.phase] && p.capture_end_ms >= lo && p.capture_start_ms <= hi,
    );
    const shapes = selected.map((p) => ({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: tx(p.capture_start_ms),
      x1: tx(p.capture_end_ms),
      y0: 0,
      y1: 1,
      fillcolor: colors[p.phase],
      opacity: 0.1,
      line: { width: 0 },
      layer: "below",
    }));
    const annotations =
      width > 1
        ? selected.map((p) => ({
            x: tx((p.capture_start_ms + p.capture_end_ms) / 2),
            y: 1,
            xref: "x",
            yref: "paper",
            text: p.phase,
            showarrow: false,
            yanchor: "bottom",
            font: { color: colors[p.phase] },
          }))
        : [];
    if (graph.removeAllListeners) graph.removeAllListeners("plotly_relayout");
    await Plotly.react(
      graph,
      traces,
      {
        paper_bgcolor: "#101723",
        plot_bgcolor: "#101723",
        font: { color: "#d4e0ef" },
        margin: { l: 70, r: 25, t: 35, b: 70 },
        xaxis: {
          range: [tx(lo), tx(hi)],
          title: {
            text:
              scale === 1
                ? summary.crop
                  ? "Time since crop start (ms)"
                  : "Time since capture start (ms)"
                : `${unit} after ${offset.toFixed(9)} ms`,
          },
          gridcolor: "#26364b",
        },
        yaxis: {
          title: { text: "AC current (A)" },
          gridcolor: "#26364b",
          zerolinecolor: "#526378",
        },
        showlegend: false,
        shapes,
        annotations,
        dragmode: "zoom",
      },
      { responsive: true, scrollZoom: true, displaylogo: false },
    );
    status.textContent =
      d.description +
      ` · ${d.raw_samples_in_range.toLocaleString()} raw samples in view`;
    graph.on("plotly_relayout", (ev) => {
      if (ev["xaxis.autorange"]) draw(0, summary.duration_ms);
      else if (ev["xaxis.range[0]"] !== undefined)
        draw(
          ev["xaxis.range[0]"] / scale + offset,
          ev["xaxis.range[1]"] / scale + offset,
        );
    });
    window.lastWaveform = d;
  } catch (e) {
    if (e.name !== "AbortError") status.textContent = e.message;
  }
}
function phase(name) {
  return summary.phases.find((p) => p.phase === name);
}
function fine(name, width) {
  const p = phase(name);
  if (!p) return;
  const mid = (p.capture_start_ms + p.capture_end_ms) / 2;
  mode.value = "raw";
  draw(mid - width / 2, mid + width / 2);
}
fetch("/api/summary" + captureSuffix)
  .then((r) => r.json())
  .then((s) => {
    summary = s;
    if (!s.has_raw) {
      status.textContent =
        "This archive includes the Perfetto trace. Open it above to see CPU/CUDA execution and the 0.1 ms RMS waveform. Raw ADC samples are not installed for this recording.";
      document.getElementById("download-raw").hidden = true;
      graph.hidden = true;
      for (const control of document.querySelectorAll(".waveform-controls"))
        control.hidden = true;
    }
    document.getElementById("capture-title").textContent =
      s.title || "One training cycle · maximum-rate waveform";
    document.getElementById("capture-description").textContent =
      s.description ||
      "One training cycle: forward, backward, and optimizer. Values are the measured AC-current component.";
    const facts = document.getElementById("facts");
    for (const text of [
      `${s.interval_ns} ns · ${s.sample_rate_hz / 1e9} GS/s`,
      `${s.resolution_bits} bits`,
      `${s.duration_ms.toFixed(1)} ms continuous`,
      `${s.samples.toLocaleString()} raw samples`,
    ]) {
      const span = document.createElement("span");
      span.textContent = text;
      facts.appendChild(span);
    }
    if (s.crop) {
      const span = document.createElement("span");
      span.textContent = `Original recording: ${s.crop.original_start_ms}–${s.crop.original_end_ms_exclusive} ms`;
      facts.appendChild(span);
    }
    document.getElementById("timing-note").textContent = s.timing.display_note;
    document.getElementById("raw-spacing").textContent = `${s.interval_ns} ns`;
    for (const name of ["forward", "backward"])
      document.getElementById("fine-" + name).hidden = !phase(name);
    for (const name of ["forward", "backward", "optimizer"]) {
      if (!phase(name)) continue;
      const b = document.createElement("button");
      b.textContent = name[0].toUpperCase() + name.slice(1);
      b.onclick = () => {
        const p = phase(name);
        draw(p.capture_start_ms - 2, p.capture_end_ms + 2);
      };
      document.getElementById("phases").appendChild(b);
    }
    draw(0, s.duration_ms);
  })
  .catch((e) => (status.textContent = e.message));
document.getElementById("full").onclick = () => draw(0, summary.duration_ms);
document.getElementById("fine-forward").onclick = () => fine("forward", 0.001);
document.getElementById("fine-backward").onclick = () =>
  fine("backward", 0.0002);
document.getElementById("zin").onclick = () => {
  const mid = (range[0] + range[1]) / 2,
    w = Math.max((range[1] - range[0]) / 10, summary.interval_ns * 1e-6);
  draw(mid - w / 2, mid + w / 2);
};
document.getElementById("zout").onclick = () => {
  const mid = (range[0] + range[1]) / 2,
    w = (range[1] - range[0]) * 10;
  draw(mid - w / 2, mid + w / 2);
};
mode.onchange = () => draw(...range);
