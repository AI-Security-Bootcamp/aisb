import captures from "./catalog.json";
import { waveform, rmsSampleRange } from "./waveform.mjs";

function json(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": status === 200 ? "public, max-age=3600" : "no-store" },
  });
}

async function download(request, bucket, key, filename) {
  const meta = await bucket.head(key);
  if (!meta) return json({ error: "Recording data is unavailable" }, 404);
  let offset = 0, length = meta.size, partial = false;
  const range = request.headers.get("Range");
  if (range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (!match || (!match[1] && !match[2])) return json({ error: "Use a single byte range" }, 400);
    if (match[1]) {
      offset = Number(match[1]);
      const end = match[2] ? Math.min(Number(match[2]), meta.size - 1) : meta.size - 1;
      length = end - offset + 1;
    } else {
      length = Math.min(Number(match[2]), meta.size);
      offset = meta.size - length;
    }
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset >= meta.size || length <= 0) {
      return new Response(null, { status: 416, headers: { "Content-Range": `bytes */${meta.size}` } });
    }
    partial = true;
  }
  const headers = new Headers({
    "Content-Type": filename.endsWith(".gz") ? "application/gzip" : "application/octet-stream",
    "Content-Disposition": `attachment; filename="${filename}"`,
    "Content-Length": String(length), "Accept-Ranges": "bytes",
    "ETag": meta.httpEtag, "Cache-Control": "public, max-age=86400, immutable",
  });
  if (partial) headers.set("Content-Range", `bytes ${offset}-${offset + length - 1}/${meta.size}`);
  const object = request.method === "HEAD" ? null : await bucket.get(key, partial ? { range: { offset, length } } : undefined);
  if (request.method !== "HEAD" && !object) return json({ error: "Recording data is unavailable" }, 404);
  return new Response(object?.body ?? null, { status: partial ? 206 : 200, headers });
}

export default {
  async fetch(request, env, context) {
    const url = new URL(request.url);
    if (!["GET", "HEAD"].includes(request.method)) return new Response("Read-only viewer", { status: 405, headers: { Allow: "GET, HEAD" } });
    if (!url.pathname.startsWith("/api/") && !url.pathname.startsWith("/data/")) return env.ASSETS.fetch(request);
    const id = url.searchParams.get("capture") || "reference";
    if (url.pathname === "/api/capabilities") return json({ import_recording: false });
    if (url.pathname === "/api/catalog") return json(Object.entries(captures).map(([id, c]) => ({ id, title: c.summary.title, default: id === "reference" })));
    if (!Object.hasOwn(captures, id)) return json({ error: "Unknown recording" }, 404);
    const capture = captures[id];
    try {
      if (url.pathname === "/api/summary") return new Response(request.method === "HEAD" ? null : capture.summary_json, {
        headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": capture.summary.raw_upload_pending ? "no-store" : "public, max-age=3600" },
      });
      if (url.pathname === "/api/waveform") {
        const lo = Number(url.searchParams.get("lo_ms") ?? 0);
        const hi = Number(url.searchParams.get("hi_ms") ?? capture.summary.duration_ms);
        const mode = url.searchParams.get("mode") || "raw";
        const key = new URL(`/__waveform_cache__/v2/${id}`, url);
        key.search = new URLSearchParams({ lo: String(lo), hi: String(hi), mode }).toString();
        const cacheKey = new Request(key);
        const cached = await caches.default.match(cacheKey);
        if (cached) return request.method === "HEAD" ? new Response(null, cached) : cached;
        const response = json(await waveform(env.TRACES, capture, lo, hi, mode));
        context.waitUntil(caches.default.put(cacheKey, response.clone()));
        return request.method === "HEAD" ? new Response(null, response) : response;
      }
      if (url.pathname === "/api/rms-samples") {
        const series = rmsSampleRange(capture,
          Number(url.searchParams.get("lo_ms") ?? 0),
          Number(url.searchParams.get("hi_ms") ?? capture.summary.duration_ms),
          url.searchParams.get("mode") || "0.001");
        const headers = {
          "Content-Type": "application/octet-stream", "Content-Length": String(series.length),
          "Cache-Control": "public, max-age=86400, immutable",
          "X-Sample-Count": String(series.count), "X-Sample-Origin-Ms": String(series.origin_ms),
          "X-Sample-Interval-Ms": String(series.interval_ms), "X-RMS-Window-Ms": String(series.window_ms),
        };
        if (request.method === "HEAD" || !series.count) return new Response(null, { headers });
        const object = await env.TRACES.get(series.key, { range: { offset: series.offset, length: series.length } });
        if (!object) return json({ error: "Recording data is unavailable" }, 404);
        return new Response(object.body, { headers });
      }
      if (url.pathname === "/data/trace.json.gz") return await download(request, env.TRACES, capture.trace_key, "trace.json.gz");
      if (url.pathname === "/data/channel-a-adc.npy") return await download(request, env.TRACES, capture.raw_key, "channel-a-adc.npy");
      return json({ error: "Not found" }, 404);
    } catch (error) {
      if (error instanceof RangeError) return json({ error: error.message }, 400);
      console.error("Waveform request failed", id, error.message);
      return json({ error: "Could not load this recording. Please try again." }, 503);
    }
  },
};
