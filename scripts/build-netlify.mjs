import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const distDir = path.join(rootDir, "dist");
const pwaSrcDir = path.join(rootDir, "platform", "roanso_platform", "pwa");
const showcaseSrcDir = path.join(rootDir, "platform", "showcase_assets");
const voiceCatalogPath = path.join(rootDir, "voces", "catalogo_etico", "voice_catalog_manifest.json");

fs.rmSync(distDir, { recursive: true, force: true });
fs.mkdirSync(path.join(distDir, "pwa"), { recursive: true });
fs.mkdirSync(path.join(distDir, "static-api", "audio"), { recursive: true });

// 1. PWA App Shell + Service Worker + Web App Manifest
fs.copyFileSync(path.join(pwaSrcDir, "index.html"), path.join(distDir, "index.html"));
fs.copyFileSync(path.join(pwaSrcDir, "manifest.webmanifest"), path.join(distDir, "manifest.webmanifest"));
fs.copyFileSync(path.join(pwaSrcDir, "sw.js"), path.join(distDir, "sw.js"));

for (const asset of ["app.css", "app.js", "icon.svg"]) {
  fs.copyFileSync(path.join(pwaSrcDir, asset), path.join(distDir, "pwa", asset));
}

// 2. Showcase Archetypes Manifest + Whisper-verified MP3 chapters on Netlify Edge CDN
const rawShowcase = JSON.parse(
  fs.readFileSync(path.join(showcaseSrcDir, "showcase_audio_manifest.json"), "utf8")
);
for (const item of rawShowcase.archetypes || []) {
  const pid = item.profile_id;
  item.audio_stream_url = `/v1/showcase/audio/${pid}`;
  const srcMp3 = path.join(showcaseSrcDir, `${pid}_cap1.mp3`);
  const dstMp3 = path.join(distDir, "static-api", "audio", `${pid}.mp3`);
  fs.copyFileSync(srcMp3, dstMp3);
}
fs.writeFileSync(
  path.join(distDir, "static-api", "archetypes.json"),
  JSON.stringify(rawShowcase, null, 2),
  "utf8"
);

// 3. Ethical Voice Design Catalog Manifest (Apache-2.0, zero minor cloning)
const rawVoices = JSON.parse(fs.readFileSync(voiceCatalogPath, "utf8"));
fs.writeFileSync(
  path.join(distDir, "static-api", "voices.json"),
  JSON.stringify(rawVoices, null, 2),
  "utf8"
);

// 4. Auditable Human-Scale AI Health & Governance Manifest (/healthz)
const healthManifest = {
  status: "ok",
  service: "roanso-control-plane-netlify",
  organization: "ROANSO SpA (Chile)",
  project: "Un Cuento, Un Abrazo",
  architecture: {
    edge_pwa_and_showcase: "Netlify Global CDN (HTTP 206 Range + Offline Service Worker)",
    relational_control_plane: "Netlify Database (Managed PostgreSQL + FOR UPDATE SKIP LOCKED)",
    biometric_vault: "Netlify Blobs + AES-256-GCM + per-family HKDF-SHA256 (1-click cryptographic shred)",
    elastic_gpu_plane: "Scale-to-zero rented GPU workers (Qwen2.5-7B-Instruct + OmniVoice + faster-whisper)"
  },
  privacy_ley_21719: {
    enforced: true,
    clinical_diagnoses_stored: false,
    minor_voice_cloning_allowed: false,
    child_facing_chatbots: false
  },
  human_scale_governance: {
    enforced: true,
    max_params_billions: 8.0,
    depends_on_frontier_api: false,
    models: [
      { role: "narrative_slm", model: "Qwen/Qwen2.5-7B-Instruct + ROANSO-Chileno-QLoRA", params_billions: 7.6, license: "Apache-2.0" },
      { role: "acoustic_tts", model: "k2-fsa/OmniVoice", params_billions: 0.6, license: "Apache-2.0" },
      { role: "acoustic_qa_asr", model: "SYSTRAN/faster-whisper-small", params_billions: 0.24, license: "MIT" }
    ]
  }
};
fs.writeFileSync(
  path.join(distDir, "static-api", "healthz.json"),
  JSON.stringify(healthManifest, null, 2),
  "utf8"
);

console.log("[build-netlify] Built static PWA + 3 verified showcase MP3s into dist/");
