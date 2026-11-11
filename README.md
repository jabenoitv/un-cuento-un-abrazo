# Un Cuento, Un Abrazo — Open Human-Scale Neuro-Inclusive Storytelling Engine

[![License: Apache 2.0](https://img.shields.io/badge/Code-Apache%202.0-blue.svg)](./LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Corpus%20%26%20Voice%20Specs-CC%20BY%204.0-green.svg)](./docs/OPEN_SPANISH_CORPUS_LEDGER_SPEC.md)
[![Hardware: CERN-OHL-P](https://img.shields.io/badge/E--Ink%20Spec-CERN--OHL--P--2.0-orange.svg)](./docs/EINK_OPEN_HARDWARE_SPEC.md)
[![Models: <=7.6B Local](https://img.shields.io/badge/Max%20Model%20Size-%E2%89%A47.6B%20Local-success.svg)](./motor/motor_local_human_scale.py)
[![Live Demo: Netlify HTTPS](https://img.shields.io/badge/Live%20PWA%20%26%20Showcase-un--cuento--un--abrazo.netlify.app-00C7B7.svg)](https://un-cuento-un-abrazo.netlify.app)

**Un Cuento, Un Abrazo** (*"One Story, One Hug"*, developed in Rancagua, Chile by **ROANSO SpA**) is an open-source, screen-safe Spanish storytelling and nocturnal audio engine built around **Human-Scale AI** principles: **Small local models (`<=7.6B`), Intentional child-privacy guardrails, and Tailored neuro-inclusive narrative mechanics**—with **zero AI chatbots talking to children** and **zero runtime dependency on frontier cloud APIs**.

> **Live Auditable PWA & Real Audio Showcase:** [https://un-cuento-un-abrazo.netlify.app](https://un-cuento-un-abrazo.netlify.app)
> Test the 3 Whisper-verified Chilean Spanish `.mp3` bedtime chapters (`Lucas`, `Mateo`, `Fresia`), 6" E-Ink view, Airplane-Mode Offline `206 Range` caching, Child Amber Bedside Lock (demo PIN `0000`), and the live **Law 21.719 Adult Portal** (`AES-256-GCM` + `HKDF` 1-click voice shred & ARCO portability).

---

## Why Human-Scale AI for Bedtime Storytelling?

Mainstream AI story apps rely on massive cloud frontier models that generate translated English clichés (*"AI slop"*) and moralizing sermons, expose minors to interactive screens/chatbots, collect clinical diagnoses, and charge cloud TTS rates (\$0.40–\$0.60 USD per chapter) that price out Latin American families and public schools.

Bedtime storytelling is **asynchronous and domain-bounded**. That lets us replace extractive cloud stacks with a compact, auditable local pipeline running on a single consumer GPU (`8–16 GB VRAM`):

```mermaid
flowchart LR
    A["Parent Daily Seed + 4 Non-Medical Sensory Sliders"] --> B["Law 21.719 Sanitizer (Strips Clinical/Medical Terms)"]
    B --> C["Local Open-Weights SLM <=7.6B (Qwen2.5-7B / Gemma-2-9B LoRA)"]
    C --> D["Deterministic Symbolic Linter (RAE/INFLESZ + Carol Gray 10.2 + 10-Gram Check)"]
    D --> E["Local OmniVoice / F5-TTS (~0.6B) + -52 dBFS Room Tone + Sleep Ramp"]
    E --> F["Local Whisper ASR (WER <=0.12) + WavLM-SV (95M) Verification"]
    F --> G["Amber Sleep PWA / Offline 6-inch E-Ink + 3-Question Parent Card"]
```

### Core Model Stack & Parameter Footprint (`--auditoria-mozilla`)

| Layer | Open Model / Engine | Parameters | License | Runtime | Frontier API? |
|---|---|---:|---|---|---|
| Narrative & 6-Criteria Reviewer | `Qwen2.5-7B-Instruct` LoRA (trained on audited PD / `CC-BY-4.0` Spanish corpus) | **7.6 B** | Apache-2.0 | Local GPU (`Ollama` / `llama.cpp` Q4_K_M ~5.2 GB VRAM) | **No** |
| Deterministic Safety & Readability | `motor/pipeline_multinino.py` (symbolic RAE syllabifier, INFLESZ, Carol Gray 10.2) | **0 B** | Apache-2.0 | Pure Python CPU / Offline | **No** |
| Nocturnal Speech + Sleep Ramp | `OmniVoice` / `F5-TTS` Spanish LoRA + `-52 dBFS` warm room tone | **0.6 B** | Apache-2.0 / MIT | Local GPU (~2.4 GB VRAM FP16) | **No** |
| Audio-to-Script Intelligibility QA | `Whisper` (`small` / `turbo` local open weights) | **0.81 B** | MIT | Local CPU / GPU | **No** |
| Timbre & Adult Consent Verification | `WavLM-Base-Plus-SV` | **0.095 B** | MIT | Local CPU / GPU | **No** |

---

## Key Neuro-Inclusive & Child-Privacy Innovations

1. **Universal Design Without Clinical Profiling (Chilean Law 21.719):**
   Instead of asking parents to disclose medical diagnoses (*Autism*, *ADHD*), families configure **four non-medical Universal Design axes** (*Language Clarity*, *Nocturnal Structure*, *Acoustic Stimulus*, *Thematic Focus*). Our live intake filter (`sanitizar_onboarding_ley21719`) automatically strips clinical and pharmacological terms before any Story Bible touches disk.
2. **22 Evidence-Backed Narrative Rules (`docs/NEURODIVERSITY_EVIDENCE_22_RULES.md`):**
   - **Literal Clarity (`demo_lucas`):** Short active sentences (`~12.6 words/sentence`, `Szigriszt-Pazos 74.8`), explicit emotional cause-and-effect (*"felt calm because..."*), inline explanation of idioms, predictable step-by-step opening, zero anxious cliffhangers, and constant `0.89x` mono-voice pacing.
   - **15-Second Hook + Physiological Sleep Ramp (`demo_mateo`):** Immediate action anchored in the child's special interest in sentence 1, followed by gradual lexical and acoustic deceleration (`0.96x -> 0.92x -> 0.86x`).
   - **Carol Gray 10.2 & Double Empathy:** Blocks 2nd-person behavioral directives (*"you must behave"*) and AI sermons, modeling mutual two-way understanding rather than demanding neurotypical "camouflaging."
   - **3-Question Parent-Child Morning Card:** Supports shared human conversation—the highest-evidence practice for narrative comprehension and socio-emotional inference.
3. **License-Audited Human Spanish Corpus (`docs/OPEN_SPANISH_CORPUS_LEDGER_SPEC.md`):**
   Excludes synthetic GPT datasets (e.g., TinyStories) and `NC`/`ND` works, using only verified Public Domain works across Chile/US/EU (e.g., Ramón Laval's 1923 Chilean folktales, Horacio Quiroga 1918) and `CC-BY-4.0` stories (Pratham StoryWeaver), enforced by a 10-gram anti-regurgitation check.

---

## 30-Second Quickstart (macOS / Linux / Windows — Python 3.10+ Stdlib)

```bash
# 1. Print the verifiable Human-Scale AI model & parameter manifest
python motor/motor_local_human_scale.py --auditoria-mozilla

# 2. Run the 10-test deterministic, privacy, Carol Gray & INFLESZ verification suite
python -m unittest discover -s tests -v

# 3. Validate the three inclusive Spanish demo archetypes directly
python motor/pipeline_multinino.py demo_lucas motor/capitulos/demo_lucas_cap1.txt
python motor/pipeline_multinino.py demo_mateo motor/capitulos/demo_mateo_cap1.txt
python motor/pipeline_multinino.py demo_fresia motor/capitulos/demo_fresia_cap1.txt

# 4. Launch the zero-dependency showcase + 6" E-Ink simulator + real MP3 audio (http://127.0.0.1:8095)
python app/server.py

# 5. Launch the full Multi-Tenant Cloud Control Plane + AES-256-GCM Voice Vault + Offline Family PWA
python -m pip install -e "./platform[dev]"
python -m pytest platform/tests -v
uvicorn roanso_platform.api:create_app --factory --app-dir platform --port 8080
```

*(Acoustic training and GPU synthesis workers in `platform/` and `voces/` run on consumer or rented GPUs using `omnivoice`, `faster-whisper`, `transformers`, `peft`, `trl`, and `ffmpeg`.)*

---

## Empirical Benchmarks & Verified Showcase Audio

1. **Chilean Neuro-Inclusive SLM Benchmark (`platform/training_data/BENCHMARK_RESULTS.md`):**
   Evaluated across all 3 Universal Design profiles (`demo_lucas`, `demo_mateo`, `demo_fresia`) against `101,968` `SHA-256`-audited Public Domain & `CC-BY-4.0` words (`corpus/licencias_corpus.csv`):
   - **Vanilla `Qwen2.5-7B` (`7.6B` Apache-2.0):** `0%` pass rate (slips neutral-dubbing terms like `autobús` and 1st-person narrator drift).
   - **ROANSO `roanso-chileno:7b` + `SelfHealingStoryEngine`:** **`100%` pass rate (`3/3`)**, `0` non-Chilean dubbing terms, `0` 10-gram corpus collisions, `Szigriszt-Pazos 61.5–68.3`.
2. **Whisper-Verified Nocturnal Showcase Audio (`platform/showcase_assets/SHOWCASE_AUDIO_AUDIT.md`):**
   Synthesized with ethical *Voice Design* catalog voices (`Apache-2.0`, zero minor cloning, zero third-party actor scraping), `-52 dBFS` warm room tone, and `-20 LUFS` mastering:
   - `demo_lucas_cap1.mp3` (`monovoz_estable`, `272.6 s`): **`Whisper WER = 0.0152` (`98.48%` accuracy)**, `-0.24 dB` steady dynamics.
   - `demo_mateo_cap1.mp3` (*Sleep Ramp*, `116.3 s`): **`Whisper WER = 0.0035` (`99.65%` accuracy)**, `+1.48 dB` gentle attenuation.
   - `demo_fresia_cap1.mp3` (*Sleep Ramp*, `230.3 s`): **`Whisper WER = 0.0179` (`98.21%` accuracy)**, `+1.01 dB` gentle attenuation.

---

## Repository Structure

- `platform/` — Production Multi-Tenant Cloud Control Plane (`FastAPI` + `SQLAlchemy 2.0` PostgreSQL/SQLite WAL + `S3/R2`), `AES-256-GCM` + `HKDF` biometric voice vault with 1-click cryptographic shred, `SKIP LOCKED` elastic GPU lease queue, 4-bit NF4 `QLoRA` trainer (`Apache-2.0`), real `.mp3` showcase assets, and Offline-First Family PWA (`Service Worker` + Child Amber Lock Mode).

- `motor/pipeline_multinino.py` — 4-axis Universal Design engine, Law 21.719 clinical sanitizer, RAE phonological syllabifier, INFLESZ readability indices, Carol Gray 10.2 & anti-sermon linter, and 3-question parent card generator.
- `motor/motor_local_human_scale.py` — Local `<=10B` open-weights SLM orchestrator (`Ollama`/`llama.cpp`), frontier API domain blocker, and 10-gram corpus anti-regurgitation verifier.
- `motor/biblias/` & `motor/capitulos/` — Three complete inclusive Spanish archetypes (`demo_lucas`, `demo_mateo`, `demo_fresia`).
- `voces/` — Nocturnal 3-register voice trainer (`WavLM-SV` + `Whisper` + spectral warmth + spoken adult consent check) and `-52 dBFS` continuous room-tone *Sleep Ramp* synthesizer.
- `app/` — Zero-dependency interactive web showcase, Law 21.719 live onboarding demo, and 6" E-Ink hardware simulator.
- `docs/` — Peer-reviewed 22-rule neurodiversity study, Spanish Public Domain / `CC-BY-4.0` corpus governance specification, and `CERN-OHL-P-2.0` 6" E-Ink bedside reader specification.
- `tests/` & `scripts/audit_public_repo.py` — Automated unit test suite and pre-publish zero-PII security gate.

---

## Resumen en Español (Chile y Latinoamérica)

**Un Cuento, Un Abrazo** (impulsado desde Rancagua, Región de O'Higgins, por **ROANSO SpA**) es un motor abierto de narrativa nocturna inclusiva en español que opera con modelos pequeños locales (`<=7.6B`), síntesis vocal cálida en GPU de consumo (`OmniVoice` + *room tone* continuo a `-52 dBFS`), legibilidad fonológica en español (`Szigriszt-Pazos / INFLESZ`) y cumplimiento estricto de privacidad infantil (**Ley 21.719**, cero almacenamiento de diagnósticos clínicos y cero chatbots frente a niños).
