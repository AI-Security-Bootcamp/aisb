import { estimateAlignment } from "./auto-alignment.mjs";
self.onmessage = ({ data }) => {
  try {
    const source = { ...data.source, values: new Float64Array(data.source.buffer) };
    const target = { ...data.target, values: new Float64Array(data.target.buffer) };
    self.postMessage({ result: estimateAlignment(source, target) });
  } catch (error) { self.postMessage({ error: error.message }); }
};
