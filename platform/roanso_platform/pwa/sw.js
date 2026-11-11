/* Service Worker Offline-First — Un Cuento, Un Abrazo (ROANSO SpA)
 * Permite reproducción nocturna en Modo Avión (cero Wi-Fi / cero pantallas interactivas en dormitorio)
 * con soporte completo para solicitudes HTTP 206 Partial Content (Range) sobre audios MP3 cacheados.
 */
const SHELL_CACHE = "roanso-shell-v1";
const DATA_CACHE = "roanso-showcase-data-v1";
const AUDIO_CACHE = "roanso-offline-audio-v1";

const SHELL_ASSETS = [
  "/",
  "/manifest.webmanifest",
  "/pwa/icon.svg",
  "/pwa/app.css",
  "/pwa/app.js"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  const allowed = new Set([SHELL_CACHE, DATA_CACHE, AUDIO_CACHE]);
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => !allowed.has(k)).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

async function serveCachedAudioWithRange(request, cachedResponse) {
  const rangeHeader = request.headers.get("range");
  if (!rangeHeader) {
    return cachedResponse;
  }
  const buf = await cachedResponse.arrayBuffer();
  const total = buf.byteLength;
  const match = /bytes=(\d+)-(\d*)/.exec(rangeHeader);
  if (!match) {
    return cachedResponse;
  }
  const start = Math.min(parseInt(match[1], 10), Math.max(0, total - 1));
  const end = match[2] ? Math.min(parseInt(match[2], 10), total - 1) : total - 1;
  const slice = buf.slice(start, end + 1);
  return new Response(slice, {
    status: 206,
    statusText: "Partial Content",
    headers: {
      "Content-Type": "audio/mpeg",
      "Accept-Ranges": "bytes",
      "Content-Range": `bytes ${start}-${end}/${total}`,
      "Content-Length": String(slice.byteLength)
    }
  });
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);

  // 1. Audios MP3 (vitrina pública o capítulos familiares guardados para Modo Avión)
  if (url.pathname.startsWith("/v1/showcase/audio/") || url.pathname.startsWith("/v1/media/")) {
    event.respondWith(
      caches.open(AUDIO_CACHE).then(async (cache) => {
        const cleanUrl = url.origin + url.pathname;
        const cached = await cache.match(cleanUrl);
        if (cached) {
          return serveCachedAudioWithRange(req, cached);
        }
        return fetch(req);
      })
    );
    return;
  }

  // 2. Metadatos públicos de vitrina (Stale-While-Revalidate para funcionar sin red)
  if (url.pathname.startsWith("/v1/showcase/") || url.pathname === "/healthz") {
    event.respondWith(
      caches.open(DATA_CACHE).then(async (cache) => {
        try {
          const fresh = await fetch(req);
          if (fresh.ok) {
            cache.put(req, fresh.clone());
          }
          return fresh;
        } catch (_err) {
          const fallback = await cache.match(req);
          if (fallback) return fallback;
          throw _err;
        }
      })
    );
    return;
  }

  // 3. App Shell estático (Cache-First con actualización de respaldo)
  if (SHELL_ASSETS.includes(url.pathname)) {
    event.respondWith(
      caches.match(url.pathname).then((cached) => cached || fetch(req))
    );
  }
});

self.addEventListener("message", (event) => {
  const data = event.data || {};
  if (data.type === "CACHE_AUDIO_OFFLINE" && data.audioUrl) {
    event.waitUntil(
      (async () => {
        const cache = await caches.open(AUDIO_CACHE);
        const cleanKey = new URL(data.audioUrl, self.location.origin);
        const cacheKeyUrl = cleanKey.origin + cleanKey.pathname;
        const headers = data.bearerToken ? { Authorization: `Bearer ${data.bearerToken}` } : {};
        const resp = await fetch(data.audioUrl, { headers });
        if (resp.ok) {
          await cache.put(cacheKeyUrl, resp.clone());
          if (event.source) {
            event.source.postMessage({ type: "AUDIO_CACHED_OK", audioUrl: cacheKeyUrl });
          }
        }
      })()
    );
  }
});
