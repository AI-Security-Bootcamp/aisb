(() => {
  const button = document.getElementById("open-perfetto");
  const status = document.getElementById("perfetto-status");
  const origin = "https://ui.perfetto.dev";
  const capture = new URLSearchParams(location.search).get("capture");
  const filename =
    "/data/trace.json.gz" +
    (capture ? "?capture=" + encodeURIComponent(capture) : "");
  button.onclick = async () => {
    // Use a fresh UI instance: an already loaded tab can retain its old trace
    // or fail the readiness handshake while navigating between recordings.
    const child = window.open("about:blank", "_blank");
    if (!child) {
      status.textContent = "Allow pop-ups to open Perfetto.";
      return;
    }
    button.disabled = true;
    status.textContent = "Loading trace and opening Perfetto…";
    const controller = new AbortController();
    let timer, timeout, message;
    try {
      const commands = [
        {
          id: "dev.perfetto.ExpandTracksByRegex",
          args: [
            "Calibrated|Training phases|Workload phases|GPU layer attribution|python 0",
          ],
        },
        {
          id: "dev.perfetto.PinTracksByRegex",
          args: [
            "AC current RMS 0.1 ms|Phase overview|GPU decoder layers|NVIDIA buffered|stream 7",
          ],
        },
      ];
      child.location =
        origin +
        "/#!/?startupCommands=" +
        encodeURIComponent(JSON.stringify(commands));
      const ready = new Promise((resolve, reject) => {
        message = (event) => {
          if (
            event.source === child &&
            event.origin === origin &&
            event.data === "PONG"
          )
            resolve();
        };
        window.addEventListener("message", message);
        timer = setInterval(() => {
          if (child.closed) reject(Error("Perfetto tab closed."));
          else child.postMessage("PING", origin);
        }, 150);
        timeout = setTimeout(() => {
          controller.abort();
          reject(Error("Perfetto took too long to load. Please try again."));
        }, 120000);
      });
      const download = fetch(filename, { signal: controller.signal }).then(
        async (response) => {
          if (!response.ok)
            throw Error("Could not load the Perfetto trace. Please try again.");
          return response.arrayBuffer();
        },
      );
      const [, buffer] = await Promise.all([ready, download]);
      child.postMessage(
        {
          perfetto: {
            buffer,
            title:
              document.getElementById("capture-title").textContent +
              " — CUDA + 0.1 ms RMS current",
            fileName: "trace.json.gz",
            shareable: false,
            downloadable: true,
          },
        },
        origin,
      );
      status.textContent =
        "Opened in Perfetto. Choose Yes if asked to open the trace.";
      child.focus();
    } catch (error) {
      status.textContent =
        error.name === "AbortError"
          ? "Loading timed out. Please try again."
          : error.message;
    } finally {
      clearInterval(timer);
      clearTimeout(timeout);
      window.removeEventListener("message", message);
      controller.abort();
      button.disabled = false;
    }
  };
})();
