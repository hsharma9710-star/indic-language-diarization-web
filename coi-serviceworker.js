/* Makes the page "cross-origin isolated" on static hosts that cannot set COOP/COEP headers (e.g. GitHub Pages),
   which lets ONNX Runtime Web use several threads. Without it the page still works, only slower. */
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
  self.addEventListener("fetch", (e) => {
    const r = e.request;
    if (r.cache === "only-if-cached" && r.mode !== "same-origin") return;
    e.respondWith(fetch(r).then((resp) => {
      if (resp.status === 0) return resp;
      const h = new Headers(resp.headers);
      h.set("Cross-Origin-Embedder-Policy", "require-corp");
      h.set("Cross-Origin-Opener-Policy", "same-origin");
      return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers: h });
    }).catch((err) => { console.error(err); return Response.error(); }));
  });
} else {
  (async () => {
    if (window.crossOriginIsolated || !window.isSecureContext || !("serviceWorker" in navigator)) return;
    try {
      const reg = await navigator.serviceWorker.register(document.currentScript ? document.currentScript.src : "coi-serviceworker.js");
      reg.addEventListener("updatefound", () => window.location.reload());
      if (reg.active && !navigator.serviceWorker.controller) window.location.reload();
    } catch (e) { console.warn("cross-origin isolation not available:", e); }
  })();
}
