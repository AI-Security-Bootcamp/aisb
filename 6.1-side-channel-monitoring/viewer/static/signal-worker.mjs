import { prepareAnalysis, audioSamples } from "./signal-analysis.mjs";

let analysis;
self.onmessage = ({ data }) => {
  try {
    if (data.type === "analyse") {
      analysis = prepareAnalysis(new Float64Array(data.buffer), data.sampleRate);
      const { values, ...result } = analysis;
      self.postMessage({ type: "analysis", requestId: data.requestId, ...result });
    } else if (data.type === "audio" && analysis) {
      const audio = audioSamples(analysis.values, analysis.sampleRate * data.speed / data.scale, data.outputRate, data.normalize !== false);
      self.postMessage({ type: "audio", requestId: data.requestId, buffer: audio.buffer }, [audio.buffer]);
    }
  } catch (error) {
    self.postMessage({ type: "error", requestId: data.requestId, message: error.message });
  }
};
