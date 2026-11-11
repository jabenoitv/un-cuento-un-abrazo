# Auditoría Acústica Paso 4 — Catálogo Ético de Voces + 3 Cuentos Showcase Reales

## 1. Catálogo Ético de Voces (`OmniVoice` *Voice Design* — `Apache-2.0`, Cero Clonación de Terceros/Menores)

| ID Voz | Descripción | Género | `WER` Semilla | Centroide Espectral | `SHA-256` (`.pt`) |
|---|---|---|---:|---:|---|
| `catalog-chileno-sereno-m1` | Narrador Sureño Sereno (Catálogo Ético es-CL) | `male` | `0.0` | `2287.2 Hz` | `49bc86404b2985` |
| `catalog-chilena-calida-f1` | Narradora Valle Central Cálida (Catálogo Ético es-CL) | `female` | `0.0` | `2212.8 Hz` | `f93572afa678f1` |

## 2. Capítulos `.mp3` Reales Verificados con `faster-whisper` (`-20 LUFS`, Room Tone `-52 dBFS`)

| Arquetipo | Modo Acústico | Voz | Palabras | Duración | Velocidad (`WPM`) | `Whisper WER` | Precisión ASR | Atenuación *Sleep Ramp* | Centroide |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `demo_lucas` | `monovoz_estable` | `catalog-chilena-calida-f1` | 656 | `272.61 s` | `144.4` | **`0.0152`** | **`98.48 %`** | `-0.24 dB` | `2336.4 Hz` |
| `demo_mateo` | `dinamico_a_calmo` | `catalog-chileno-sereno-m1` | 283 | `116.34 s` | `146.0` | **`0.0035`** | **`99.65 %`** | `1.48 dB` | `2400.9 Hz` |
| `demo_fresia` | `dinamico_a_calmo` | `catalog-chilena-calida-f1` | 558 | `230.3 s` | `145.4` | **`0.0179`** | **`98.21 %`** | `1.01 dB` | `2582.8 Hz` |
