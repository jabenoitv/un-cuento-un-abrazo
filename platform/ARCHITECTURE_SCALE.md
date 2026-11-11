# Arquitectura Productiva a Escala — ROANSO SpA (*"Un Cuento, Un Abrazo"*)

Este módulo implementa la arquitectura profesional desacoplada lista para operar comercialmente sobre **servidores cloud arrendados (VPS CPU)** y **nodos GPU elásticos arrendados por minuto (RunPod / Modal / Hetzner GPU)**, cumpliendo simultáneamente **Ley 21.719 (Chile)** y los requisitos técnicos de **Mozilla Human-Scale AI (`<= 10B` parámetros, cero APIs *frontier*)**.

---

## 1. Topología Desacoplada: Control Plane 24/7 vs. GPU Worker Plane *Scale-to-Zero*

```mermaid
flowchart LR
    PWA["PWA Familiar / Lector E-Ink"] -->|"JWT + PIN Parental (HTTPS)"| API["Control Plane API (FastAPI en VPS US$7-15/mes)"]
    API --> PG[("PostgreSQL 16 (Multi-Tenant + SKIP LOCKED Queue)")]
    API --> R2[("Cloudflare R2 / S3 (MP3 + Bóveda Voz AES-256-GCM)")]
    GPU["Nodos GPU Arrendados (RunPod L4 / 4090 24 GB)"] -->|"Outbound Pull Lease + Heartbeat (X-Worker-Token)"| API
    GPU -->|"1. vLLM Qwen2.5-7B + LoRA (<=3 turnos auto-reparación)"| LINT["Validador Carol Gray + INFLESZ + Anti-10gram"]
    LINT -->|"2. OmniVoice LoRA + Sleep Ramp (-52 dBFS)"| ASR["3. Faster-Whisper QA (WER <= 0.12)"]
    ASR -->|"Upload MP3 + Reporte QA"| API
```

### ¿Por qué esta arquitectura escala sin quemar caja?
1. **Control Plane siempre disponible a costo fijo mínimo (~US\$7–\$15/mes):** Un VPS Linux liviano (2 vCPU / 4 GB RAM en Hetzner, Railway o Fly.io) sirve instantáneamente autenticación, perfiles, catálogo y descarga de audios desde Cloudflare R2 (US\$0 de *egress*).
2. **Nodos GPU arrendados sin puertos expuestos y con *Scale-to-Zero*:** Los contenedores `Dockerfile.gpu-worker` hacen *outbound HTTPS pull* hacia `/v1/worker/leases/claim`. Cuando la cola nocturna queda vacía (`204 No Content`), el contenedor con `--drain-and-exit` se apaga y el gasto GPU baja a **US\$0,00/hora**.
3. **Tolerancia a fallos de GPUs *Spot* (60% más baratas):** Cada trabajo tiene un `lease_expires_at` renovado por *heartbeat*. Si un proveedor corta una instancia *spot*, `requeue_expired_leases()` devuelve automáticamente el trabajo a `QUEUED` y otro nodo lo completa sin intervención humana.

---

## 2. Unit Economics Reales en GPU Arrendada (24 GB VRAM)

| Etapa en Nodo GPU (`NVIDIA L4` / `RTX 4090`) | Tiempo Promedio | Consumo |
|---|---|---|
| Escritura `Qwen2.5-7B-Instruct` + LoRA (`vLLM`) + Linter | ~4,5 s | Abierto (`Apache-2.0`) |
| Síntesis Nocturna `OmniVoice` + *Sleep Ramp* + *Room Tone* | ~17,5 s | Abierto (`Apache-2.0`) |
| Verificación Acústica `faster-whisper` (`WER <= 0.12`) | ~3,0 s | Abierto (`MIT`) |
| **Total por capítulo verificado (~750 palabras / ~6 min audio)** | **~25,0 s GPU** | **~US\$0,0038 (~CLP \$3,7 / cuento)** |

- **Capacidad de 1 nodo GPU arrendado:** ~140 capítulos/hora.
- **1.000 familias activas diarias:** se procesan en ~7 horas-GPU totales (~US\$3,80/noche o ~10 minutos corriendo 40 workers en paralelo).

---

## 3. Fronteras de Seguridad y Privacidad (Ley 21.719)

1. **Cero Diagnósticos Clínicos en Base de Datos:** `ChildProfileCreate` ejecuta `sanitizar_onboarding_ley21719` antes de crear la entidad ORM. Términos médicos (`TEA`, `TDAH`, fármacos) son purgados en memoria; solo se registra el contador anónimo `sanitized_terms_blocked_count`.
2. **Bóveda Biométrica `AES-256-GCM` + Borrado Criptográfico (`HKDF`):**
   - Prohibición estricta de voces de menores.
   - Cada embedding adulto se cifra con `AES-256-GCM` derivando una clave única mediante `HKDF-SHA256(master_key, key_salt, family_id)`.
   - `DELETE /v1/voices/{id}` elimina el objeto en `S3/R2` **y destruye `key_salt_hex`**, haciendo matemáticamente irrecuperable cualquier respaldo frío.
3. **Doble Compuerta Neuro-Inclusiva:** Aunque el nodo GPU ejecuta `SelfHealingStoryEngine`, el Control Plane vuelve a correr `analizar_cuento_multinino` + `verificar_anti_regurgitacion_ngramas` en `/complete` antes de aceptar un capítulo como `READY`.
