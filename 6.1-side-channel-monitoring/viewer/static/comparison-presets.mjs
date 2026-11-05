import { validTransform } from "./comparison-math.mjs";

export const presetStorageKey = "aisb-attention-comparison-presets-v1";

export function validatePreset(preset, captureIds) {
  if (preset?.version !== 1 || typeof preset.name !== "string" || !preset.name.trim() || preset.name.length > 80) {
    throw Error("Choose a preset name of 1–80 characters.");
  }
  if (!captureIds.every(id => validTransform(preset.transforms?.[id])) ||
      !captureIds.includes(preset.reference) || !(preset.analysisTrace === "all" || captureIds.includes(preset.analysisTrace)) ||
      !["0.1", "0.01", "0.001", "0.0001"].includes(preset.rmsWindow) ||
      typeof preset.overlay !== "boolean" || !Array.isArray(preset.range) || preset.range.length !== 2 ||
      !preset.range.every(Number.isFinite) || preset.range[1] <= preset.range[0]) {
    throw Error("This preset has invalid settings or does not match the six comparison recordings.");
  }
  for (const field of ["fftMode", "fftLog"]) {
    if (preset[field] !== undefined && typeof preset[field] !== "boolean") throw Error("Invalid FFT preset setting.");
  }
  // Reconstruct known fields only. Never merge arbitrary imported object keys
  // into application state, DOM, or a prototype-bearing settings dictionary.
  return { version: 1, name: preset.name.trim(), reference: preset.reference,
    analysisTrace: preset.analysisTrace, rmsWindow: preset.rmsWindow, overlay: preset.overlay,
    range: [...preset.range], transforms: Object.fromEntries(captureIds.map(id => [id, {
      scale: preset.transforms[id].scale, shift: preset.transforms[id].shift,
    }])), ...(preset.fftMode === undefined ? {} : { fftMode: preset.fftMode }),
    ...(preset.fftLog === undefined ? {} : { fftLog: preset.fftLog }) };
}

export function setupPresets(captureIds, getSettings, applySettings) {
  const name = document.getElementById("preset-name"), list = document.getElementById("preset-list");
  const status = document.getElementById("preset-status"), file = document.getElementById("preset-file");
  let presets = [];
  try {
    const stored = JSON.parse(localStorage.getItem(presetStorageKey) || "[]");
    if (Array.isArray(stored)) for (const entry of stored.slice(0, 100)) {
      try { presets.push(validatePreset(entry, captureIds)); } catch { /* Ignore an obsolete preset. */ }
    }
  } catch { /* File export/import remains available without browser storage. */ }
  function updateList(selected) {
    list.replaceChildren(new Option("Choose a preset…", ""));
    presets.forEach((preset, index) => list.add(new Option(preset.name, String(index))));
    list.value = selected === undefined ? "" : String(selected);
  }
  function current() { return validatePreset({ ...getSettings(), version: 1, name: name.value }, captureIds); }
  function save(preset) {
    let index = presets.findIndex(entry => entry.name === preset.name);
    if (index < 0) {
      if (presets.length >= 100) throw Error("This browser has 100 presets. Reuse a name or export the current settings.");
      index = presets.length;
    }
    presets[index] = preset;
    updateList(index);
    try {
      localStorage.setItem(presetStorageKey, JSON.stringify(presets));
      status.textContent = `Saved “${preset.name}” in this browser. Export JSON to keep or share it.`;
    } catch {
      status.textContent = `“${preset.name}” is available for this session only; browser storage is unavailable. Export JSON to keep it.`;
    }
  }
  function action(id, handler) {
    const button = document.getElementById(id);
    button.disabled = false;
    button.addEventListener("click", () => { try { handler(); } catch (error) { status.textContent = error.message; } });
  }
  action("save-preset", () => save(current()));
  action("load-preset", () => {
    if (list.value === "") throw Error("Choose a saved preset first.");
    const preset = validatePreset(presets[Number(list.value)], captureIds);
    applySettings(preset); name.value = preset.name;
    status.textContent = `Loaded “${preset.name}”.`;
  });
  action("export-preset", () => {
    const preset = current();
    const url = URL.createObjectURL(new Blob([JSON.stringify(preset, null, 2) + "\n"], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url; link.download = (preset.name.replace(/[^a-zA-Z0-9_-]+/g, "-") || "alignment") + ".json";
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = `Exported current settings as “${preset.name}”.`;
  });
  action("import-preset", () => file.click());
  file.addEventListener("change", async () => {
    try {
      const chosen = file.files[0];
      if (!chosen) return;
      if (chosen.size > 100_000) throw Error("Preset JSON must be smaller than 100 KB.");
      const preset = validatePreset(JSON.parse(await chosen.text()), captureIds);
      save(preset); applySettings(preset); name.value = preset.name;
      status.textContent = `Imported and loaded “${preset.name}”.`;
    } catch (error) { status.textContent = error.message; }
    finally { file.value = ""; }
  });
  updateList();
}
