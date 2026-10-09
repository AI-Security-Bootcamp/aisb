// Send the File directly: do not create a multi-gigabyte ArrayBuffer in the tab.
fetch("/api/capabilities")
  .then((response) => response.json())
  .then((capabilities) => {
    document.getElementById("import-controls").hidden = !capabilities.import_recording;
  });

document.getElementById("import-recording").onclick = () => {
  const file = document.getElementById("import-file").files[0];
  const message = document.getElementById("import-status");
  const button = document.getElementById("import-recording");
  if (!file) {
    message.textContent = "Choose an .aisb.zip recording bundle first.";
    return;
  }
  button.disabled = true;
  const request = new XMLHttpRequest();
  request.open("POST", "/api/import");
  request.setRequestHeader("Content-Type", "application/zip");
  request.setRequestHeader("X-AISB-Import", "1");
  request.upload.onprogress = (event) => {
    message.textContent = event.loaded === event.total
      ? "Checking the bundle and preparing the waveform…"
      : `Uploading ${Math.round(100 * event.loaded / event.total)}%…`;
  };
  request.onload = () => {
    button.disabled = false;
    try {
      const result = JSON.parse(request.responseText);
      if (request.status !== 200) throw Error(result.error || "Import failed");
      location.search = "capture=" + encodeURIComponent(result.id);
    } catch (error) {
      message.textContent = error.message;
    }
  };
  request.onerror = () => {
    button.disabled = false;
    message.textContent = "Connection lost. Check that your local viewer is still running.";
  };
  message.textContent = "Uploading recording…";
  request.send(file);
};
