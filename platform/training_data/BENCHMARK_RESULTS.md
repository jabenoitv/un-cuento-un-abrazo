# Reporte Empírico Paso 3 — Corpus Abierto Auditado (`SHA-256`) + Kit QLoRA + Benchmark Chileno

## 1. Resumen del Corpus Humano Abierto Ingerido (`licencias_corpus.csv` — 36 columnas)
- **Obras verificadas criptográficamente (`SHA-256`):** `9` (`PDM-1.0` triple jurisdicción CL/US/EU + `CC-BY-4.0` ROANSO)
- **Palabras limpias totales:** `101968` (`0` apariciones residuales de marca registrada)
- **Pasajes nocturnos seguros (`pasajes_nocturnos_seguros.jsonl`):** `497` (filtrados de portadas académicas y violencia folclórica)
- **Ejemplos SFT ChatML (`sft_train_chileno.jsonl` + `sft_val_chileno.jsonl`):** `447 train` / `50 val`
- **Pares DPO Preferencias Chilenas (`dpo_preferencias_chilenas.jsonl`):** `3`

## 2. Benchmark en Vivo (`<= 8B` Open-Weights Local: Vanilla `qwen2.5:7b` vs. `roanso-chileno:7b`)

| Arquetipo | Ejes Sensoriales | Brazo | Aprobado QA + 10-gram | Intentos | Palabras | Szigriszt (INFLESZ) | Léxico No-Chileno | Desvío 1ª Persona | Errores Detectados |
|---|---|---|---|---:|---:|---:|---|---|---:|
| `demo_lucas` | `literal_claro` / `ritual_anticipado` | Vanilla `qwen2.5:7b` | NO | 1 | 167 | 70.3 | autobús | NO | 2 |
| `demo_lucas` | `literal_claro` / `ritual_anticipado` | **ROANSO `roanso-chileno:7b`** | **SI** | **2** | **315** | **68.3** | **0** | **NO** | **0** |
| `demo_mateo` | `agil_visual` / `enganche_rampa_calma` | Vanilla `qwen2.5:7b` | NO | 1 | 188 | 80.0 | 0 | SI | 2 |
| `demo_mateo` | `agil_visual` / `enganche_rampa_calma` | **ROANSO `roanso-chileno:7b`** | **SI** | **1** | **506** | **68.1** | **0** | **NO** | **0** |
| `demo_fresia` | `literario_rico` / `serie_multiverso` | Vanilla `qwen2.5:7b` | NO | 1 | 251 | 66.6 | 0 | SI | 1 |
| `demo_fresia` | `literario_rico` / `serie_multiverso` | **ROANSO `roanso-chileno:7b`** | **SI** | **1** | **299** | **61.5** | **0** | **NO** | **0** |
