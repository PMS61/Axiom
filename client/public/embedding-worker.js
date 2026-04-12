/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Embedding Web Worker
   Handles Transformers.js inference off the main thread.
   ═══════════════════════════════════════════════════════════ */

import {
  pipeline,
  env,
} from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

env.allowLocalModels = false;
env.useBrowserCache = true;

let pipe = null;

self.onmessage = async (e) => {
  const { type, text, topic, chunks, deadline } = e.data;

  try {
    if (type === "init") {
      if (!pipe) {
        self.postMessage({
          status: "progress",
          message: "Initialising Transformers.js...",
        });
        pipe = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
          progress_callback: (p) => {
            if (p.status === "downloading") {
              const percent = p.total
                ? ` — ${Math.round((p.loaded / p.total) * 100)}%`
                : ` — ${(p.loaded / (1024 * 1024)).toFixed(1)}MB loaded`;
              self.postMessage({
                status: "progress",
                message: `Downloading ${p.file}${percent}`,
              });
            }
          },
        });
      }
      self.postMessage({ status: "ready" });
      return;
    }

    if (type === "embed") {
      if (!pipe) throw new Error("Pipeline not initialised.");
      const output = await pipe(text, { pooling: "mean", normalize: true });
      self.postMessage({
        status: "complete",
        result: Array.from(output.data),
      });
    }
  } catch (err) {
    self.postMessage({ status: "error", error: err.message });
  }
};
