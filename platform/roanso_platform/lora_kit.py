# -*- coding: utf-8 -*-
"""Constructor de Datasets SFT/DPO Chilenos Auditados + Verificador Pre-Entrenamiento LoRA (Paso 3).

Prepara los artefactos listos para entrenar un adaptador `QLoRA` (`Apache-2.0`) en un servidor GPU
arrendado (`NVIDIA L4 24 GB` / `RTX 4090` / `T4 16 GB`, costo ~US$0,50–$1,20 por corrida):
1. Verifica criptográficamente (`SHA-256`) que cada texto en `corpus_limpio/` coincida con su fila
   aprobada en `licencias_corpus.csv` antes de construir el dataset.
2. Genera `sft_train_chileno.jsonl` y `sft_val_chileno.jsonl` en formato `messages` (ChatML estándar
   compatible con `TRL SFTTrainer`, `Unsloth` y `Axolotl`).
3. Genera `dpo_preferencias_chilenas.jsonl` (pares `chosen` vs. `rejected`) enseñando explícitamente:
   - Español de Chile afectivo (`chaqueta`, `conversar`, `porotos`) sobre doblaje neutro (`chamarra`, `platicar`).
   - Descripción empática Carol Gray 10.2 sobre órdenes en 2ª persona (`tú debes portarte bien`).
   - Valores mostrados con acciones concretas sobre moralejas sintéticas (`aprendió una valiosa lección`).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .corpus_pipeline import sha256_bytes, validate_ledger_row


def verify_corpus_integrity_against_ledger(corpus_dir: Path) -> List[Dict[str, str]]:
    """Bloquea cualquier entrenamiento si falta `licencias_corpus.csv` o no calza algún `SHA-256`."""
    ledger_path = corpus_dir / "licencias_corpus.csv"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Falta ledger obligatorio: {ledger_path}")

    with ledger_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("El ledger `licencias_corpus.csv` está vacío.")

    txt_by_sha: Dict[str, Path] = {}
    for txt_file in corpus_dir.glob("*.txt"):
        txt_by_sha[sha256_bytes(txt_file.read_bytes())] = txt_file

    verified: List[Dict[str, str]] = []
    for row in rows:
        ok, errs = validate_ledger_row(row)
        if not ok:
            raise PermissionError(f"Obra {row.get('id')} rechazada por gobernanza legal: {errs}")
        expected_sha = row["sha256_archivo"]
        if expected_sha not in txt_by_sha:
            raise ValueError(f"Hash SHA-256 no verificado en disco para obra {row['id']} ({row['titulo']}).")
        verified.append(row)
    return verified


DPO_CHILEAN_PREFERENCE_SEEDS: List[Dict[str, str]] = [
    {
        "prompt": (
            "Escribe un párrafo nocturno en 3ª persona sobre Mateo abrigándose para mirar las estrellas "
            "junto al estero en español de Chile."
        ),
        "chosen": (
            "Mateo se puso su chaqueta de lana tibia y salió despacito al corredor de madera mientras la "
            "perrita Canela caminaba a su lado. Se sintió tranquilo porque el murmullo parejo del estero "
            "sonaba suave bajo las primeras estrellas del sur."
        ),
        "rejected": (
            "Papá y yo salimos rápido. Me puse mi chamarra gruesa junto a la alberca para platicar un "
            "rato, y así aprendí una valiosa lección sobre obedecer."
        ),
    },
    {
        "prompt": (
            "Narra con lenguaje claro y directo cómo Lucas resuelve un ruido molesto en la estación "
            "explicando las intenciones de su amigo sin órdenes en 2ª persona ni sermones."
        ),
        "chosen": (
            "Lucas sintió los hombros apretados porque dos palancas juntas hacen un clic metálico fuerte. "
            "Levantó una mano con calma y explicó que ese ruido le molestaba en los oídos. Su amigo Tomás "
            "no quería incomodarlo, solo estaba muy entusiasmado por encender la luz verde; lo entendió al "
            "tiro, eso quiere decir de inmediato, y acordaron bajar una sola palanca por turno."
        ),
        "rejected": (
            "Tú debes aguantar el ruido sin quejarte y tienes que portarte bien con tus amigos. La "
            "moraleja de esta historia es que siempre debes adaptarte a los demás."
        ),
    },
    {
        "prompt": (
            "Escribe el cierre de rampa de sueño para Fresia en Caleta Niebla sin cliffhangers angustiantes."
        ),
        "chosen": (
            "El faro de Caleta Niebla giró lento proyectando su luz color miel sobre los botes amarrados. "
            "Fresia dejó su cuaderno azul en el velador, respiró hondo escuchando la brisa suave y cerró "
            "los ojos confiada pensando en el pan tibio de mañana porque en casa te queremos mucho."
        ),
        "rejected": (
            "De pronto una sombra misteriosa cruzó el muelle oscuro y nadie sabía qué peligro les "
            "esperaba. ¿Lograrán salvarse? Continuará..."
        ),
    },
    {
        "prompt": (
            "Narra una escena cotidiana de amistad en el patio del colegio donde Sofía descubre las "
            "verdaderas intenciones de un compañero sin usar etiquetas clínicas ni moralejas."
        ),
        "chosen": (
            "Sofía miró su torre de bloques de madera recién caída sobre la banca del patio. Al ver las "
            "mejillas coloradas y las manos abiertas de Martín buscando los bloques bajo la mesa, notó que "
            "había pasado a llevar la banca sin querer al correr tras la pelota, así que respiró lento y "
            "le propuso armar juntos el puente más firme."
        ),
        "rejected": (
            "Como Sofía tiene un diagnóstico sensorial especial, el terapeuta dice que tú debes controlar "
            "tu enojo y perdonar siempre para ser una niña buena."
        ),
    },
    {
        "prompt": (
            "Describe cómo Gaspar pone un límite amable cuando un amigo le pide prestado su cuaderno de "
            "trenes favoritos antes de terminar de dibujar."
        ),
        "chosen": (
            "Gaspar sostuvo su cuaderno contra el chaleco azul y le dijo con voz tranquila a Julián que "
            "primero quería terminar de pintar las ruedas de la locomotora y que en cinco minutos podrían "
            "mirar todas las páginas juntos en la alfombra."
        ),
        "rejected": (
            "¡Eres un niño egoísta si no prestas tus juguetes en este mismo instante! Todos los niños "
            "buenos regalan sus cosas sin decir que no."
        ),
    },
    {
        "prompt": (
            "Explica de manera natural una expresión figurada dentro de un cuento chileno tranquilo para "
            "que cualquier niño siga la historia paso a paso."
        ),
        "chosen": (
            "El abuelo miró las nubes grises sobre la cordillera y dijo que se venía el agua a cántaros, "
            "eso significa que pronto caería una lluvia tupida y pareja sobre el techo de tejuelas, por lo "
            "que entraron la leña seca al corredor."
        ),
        "rejected": (
            "El abuelo habló en sentido figurado con puras metáforas irónicas que nadie entendió y después "
            "se burló diciendo que era pan comido."
        ),
    },
    {
        "prompt": (
            "Narra un cambio pequeño en el plan de la tarde mostrando calma y anticipación familiar en "
            "español chileno cálido."
        ),
        "chosen": (
            "Como la llovizna mojó la vereda de la plaza, mamá avisó temprano que hoy tomarían once en la "
            "cocina tibia con pan amasado y después armarían la carpa de frazadas junto al sofá, manteniendo "
            "el mismo orden tranquilo antes del cuento de la noche."
        ),
        "rejected": (
            "Todo se canceló de sorpresa en el último segundo mientras sonaban bocinas estridentes en el "
            "autobús rumbo a la acera mojada."
        ),
    },
    {
        "prompt": (
            "Escribe un pasaje breve sobre pedir ayuda a un adulto de confianza cuando una broma en el "
            "recreo deja de sentirse divertida."
        ),
        "chosen": (
            "Cuando las risas alrededor del columpio empezaron a sonar demasiado fuertes, Emilia sintió el "
            "pecho apretado, caminó tranquila hacia la profesora del patio y le contó con palabras simples "
            "que prefería cambiar a un juego donde todos se rieran juntos y nadie quedara incómodo."
        ),
        "rejected": (
            "En el tapiz mágico del reino encantado brilló un destello mágico que mágicamente resolvió "
            "todo con una valiosa enseñanza moral."
        ),
    },
]


def build_sft_and_dpo_datasets(corpus_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Construye los datasets JSONL auditados listos para subir o montar en un nodo GPU arrendado."""
    verified_rows = verify_corpus_integrity_against_ledger(corpus_dir)
    allowed_ids = {r["id"] for r in verified_rows if r["uso_permitido"] == "entrenamiento"}

    passages_file = corpus_dir / "pasajes_nocturnos_seguros.jsonl"
    if not passages_file.exists():
        raise FileNotFoundError(f"Falta {passages_file}; ejecute primero `ingest_verified_open_corpus`.")

    sft_records: List[Dict[str, Any]] = []
    for line in passages_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item["work_id"] not in allowed_ids:
            continue
        sys_msg = (
            "Eres un narrador chileno y sudamericano cálido de cuentos nocturnos infantiles ('Un cuento, un abrazo'). "
            "Escribes en 3ª persona serena, sin clichés de IA, sin moralejas explícitas y con vocabulario natural."
        )
        usr_msg = (
            f"Escribe un pasaje narrativo sereno y descriptivo apto para lectura antes de dormir "
            f"(referencia de cadencia humana: {item['autor']}, idioma {item['idioma']})."
        )
        sft_records.append(
            {
                "source_work_id": item["work_id"],
                "passage_sha256": item["sha256"],
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg},
                    {"role": "assistant", "content": item["texto"]},
                ],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_idx = max(1, int(len(sft_records) * 0.9))
    train_rows = sft_records[:split_idx]
    val_rows = sft_records[split_idx:] or sft_records[:2]

    train_path = output_dir / "sft_train_chileno.jsonl"
    val_path = output_dir / "sft_val_chileno.jsonl"
    dpo_path = output_dir / "dpo_preferencias_chilenas.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with dpo_path.open("w", encoding="utf-8") as f:
        for pair in DPO_CHILEAN_PREFERENCE_SEEDS:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    modelfile_path = output_dir / "Modelfile.roanso-chileno"
    modelfile_path.write_text(
        "FROM qwen2.5:7b\n"
        "PARAMETER temperature 0.55\n"
        "PARAMETER num_ctx 4096\n"
        "SYSTEM \"\"\"Eres el narrador nocturno chileno de ROANSO ('Un cuento, un abrazo'). "
        "Narras siempre en 3ª persona cálida y prosa continua de un solo párrafo, en español de Chile "
        "natural (chaqueta, chaleco, conversar, frazada, vereda, estero, despacito), jamás usas "
        "mexicanismos ni doblaje neutro (chamarra, platicar, alberca, frijoles, autobús, zumo), "
        "jamás incluyes etiquetas clínicas ni diagnósticas, muestras valores cotidianos describiendo "
        "emociones e intenciones reales de los personajes sin ordenar en 2ª persona ni dar moralejas, "
        "explicas cualquier expresión figurada con sencillez paso a paso y cierras en calma profunda "
        "para acompañar el sueño familiar.\"\"\"\n",
        encoding="utf-8",
    )

    return {
        "verified_works": len(verified_rows),
        "sft_train_examples": len(train_rows),
        "sft_val_examples": len(val_rows),
        "dpo_preference_pairs": len(DPO_CHILEAN_PREFERENCE_SEEDS),
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "dpo_jsonl": str(dpo_path),
        "ollama_modelfile": str(modelfile_path),
    }
