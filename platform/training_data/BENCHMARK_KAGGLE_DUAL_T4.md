# Benchmark Empírico en GPU Dual (`Tesla T4 x2`) — Modelos Abiertos `<= 7.6B` ROANSO

- **Infraestructura de evaluación:** `2x Tesla T4` (100% *air-gapped*, `HF_HUB_OFFLINE=1`, cero llamadas a APIs *frontier*, costo `US$0.00`).
- **Corpus retenido (`sft_val_chileno.jsonl`):** `50` pasajes nocturnos chilenos auditados (`SHA-256`, `PDM-1.0` / `CC-BY-4.0`).
- **Escenarios evaluados:** `5` cuentos cotidianos de **Diseño Universal Invisible** (7 compuertas: Ley 21.719 cero etiquetas clínicas, Carol Gray 10.2 cero directivas en 2ª persona, cero moralejas IA, léxico chileno natural, legibilidad Szigriszt-Pazos INFLESZ, cierre calmo y `0` colisiones de 10-gramas contra `497` pasajes del corpus).

## 1. Reducción de Perplejidad en Español Narrativo Chileno (`Held-Out Validation`)

| Modelo Abierto (`Apache-2.0`) | Parámetros | Pico VRAM (`NF4`) | Loss Base | Loss `QLoRA+DPO` | Perplejidad Base | Perplejidad `QLoRA+DPO` | Mejora Perplejidad |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-4B` | `4.0B` | `3790.3 MiB` | `3.4891` | **`2.5356`** | `32.76` | **`12.62`** | **`-61.48%`** |
| `Qwen/Qwen2.5-7B-Instruct` | `7.6B` | `6776.9 MiB` | `2.5853` | **`2.3639`** | `13.27` | **`10.63`** | **`-19.89%`** |

## 2. Cumplimiento de las 7 Compuertas Human-Scale AI (5 Cuentos de Diseño Universal)

| Modelo | Brazo Evaluado | Cuentos Aprobados (`7/7` Compuertas) | Tasa de Aprobación | INFLESZ Medio (Szigriszt-Pazos) | Colisiones 10-Gramas |
|---|---|---:|---:|---:|---:|
| `Qwen/Qwen3-4B` (`4.0B`) | Base Zero-Shot (Sin alinear) | `0/5` | `0.0%` | — | — |
| **`Qwen/Qwen3-4B` (`4.0B`)** | **ROANSO `QLoRA+DPO` + Linter Simbólico** | **`5/5`** | **`100.0%`** | **`70.41`** | **`0`** |
| `Qwen/Qwen2.5-7B-Instruct` (`7.6B`) | Base Zero-Shot (Sin alinear) | `0/5` | `0.0%` | — | — |
| **`Qwen/Qwen2.5-7B-Instruct` (`7.6B`)** | **ROANSO `QLoRA+DPO` + Linter Simbólico** | **`3/5`** | **`60.0%`** | **`68.03`** | **`0`** |

## 3. Detalle por Escenario Cotidiano (`Qwen3-4B` y `Qwen2.5-7B-Instruct`)

| Escenario | Valor Cotidiano | Base Zero-Shot (Fallos) | ROANSO `Qwen3-4B` (Turnos / Palabras / INFLESZ) | ROANSO `Qwen2.5-7B` (Turnos / Palabras / INFLESZ) |
|---|---|---|---|---|
| *El puente de bloques en el recreo* | diferenciar un accidente por entusiasmo de una mala intención | `formato_multiparrafo (14 bloques), falta_frase_cierre_calma` | **PASS** (`1t`, `196w`, `64.22`) | **FAIL** (`4t`, `97w`, `67.97`) |
| *Cinco minutos más con la locomotora azul* | poner un límite amable sin enojarse ni sentirse culpable | `formato_multiparrafo (15 bloques), falta_frase_cierre_calma` | **PASS** (`1t`, `185w`, `72.38`) | **PASS** (`4t`, `162w`, `74.88`) |
| *El sombrero de papel junto al columpio* | distinguir cuándo una broma deja de ser divertida y pedir ayuda con calma | `formato_multiparrafo (18 bloques), ordenes_2a_persona_carol_gray` | **PASS** (`1t`, `159w`, `73.19`) | **FAIL** (`4t`, `95w`, `57.13`) |
| *La carpa de frazadas en tarde de lluvia* | adaptarse paso a paso cuando un panorama cambia por la lluvia | `formato_multiparrafo (12 bloques), lexico_no_chileno:autobús` | **PASS** (`1t`, `176w`, `68.7`) | **PASS** (`1t`, `204w`, `63.36`) |
| *La carrera de barquitos en el estero* | manejar la frustración al llegar segundo y valorar la amistad | `formato_multiparrafo (13 bloques), ordenes_2a_persona_carol_gray` | **PASS** (`1t`, `206w`, `73.57`) | **PASS** (`2t`, `204w`, `76.81`) |
