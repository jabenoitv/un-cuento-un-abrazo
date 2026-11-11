# -*- coding: utf-8 -*-
"""Suite de Pruebas Automatizadas Paso 3 — Gobernanza Corpus Libre, Léxico Chileno y Kit QLoRA."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from roanso_platform.corpus_pipeline import (
    LEDGER_COLUMNS_V1,
    extract_bedtime_safe_passages,
    extract_wikisource_clean_prose,
    sha256_bytes,
    strip_gutenberg_boilerplate_and_trademark,
    validate_ledger_row,
)
from roanso_platform.domain_bridge import (
    analizar_cuento_multinino,
    build_bible_dict_from_child,
)
from roanso_platform.lora_kit import (
    build_sft_and_dpo_datasets,
    verify_corpus_integrity_against_ledger,
)
from training.train_qlora_rented_gpu import run_qlora_training


def _valid_base_row() -> dict:
    row = {col: "NA" for col in LEDGER_COLUMNS_V1}
    row.update(
        {
            "id": "CL-PD-TEST",
            "titulo": "Cuentos populares en Chile",
            "autor": "Ramón A. Laval",
            "autor_muerte": "1929",
            "licencia_spdx": "PDM-1.0",
            "pd_chile": "si",
            "pd_eeuu": "si",
            "pd_ue": "si",
            "idioma": "es-CL",
            "palabras": "1200",
            "uso_permitido": "entrenamiento",
            "sintetico": "no",
            "fecha_publicacion_original": "1923",
            "tipo_procedencia": "dominio_publico",
            "evidencia_autoria": "Edición 1923",
        }
    )
    return row


def test_ledger_validator_enforces_commercial_and_copyright_rules() -> None:
    ok, errs = validate_ledger_row(_valid_base_row())
    assert ok is True
    assert errs == []

    # 1. Bloquea licencias no comerciales o copyleft virales (NC, ND, BY-SA, GFDL)
    for bad_spdx in ("CC-BY-NC-4.0", "CC-BY-ND-4.0", "CC-BY-SA-4.0", "GFDL-1.3"):
        r = _valid_base_row()
        r["licencia_spdx"] = bad_spdx
        ok_bad, errs_bad = validate_ledger_row(r)
        assert ok_bad is False
        assert any("incompatible" in e.lower() for e in errs_bad)

    # 2. Bloquea datasets sintéticos (ej. TinyStories / ajibawa)
    r_syn = _valid_base_row()
    r_syn["sintetico"] = "si"
    ok_syn, errs_syn = validate_ledger_row(r_syn)
    assert ok_syn is False
    assert any("sintética" in e.lower() for e in errs_syn)

    # 3. Bloquea autores en disputa o aún protegidos en Chile (Mistral, Schkolnik, Pino Saavedra, Walsh)
    for blocked_author in ("Gabriela Mistral", "Saúl Schkolnik", "Yolando Pino Saavedra", "María Elena Walsh"):
        r_aut = _valid_base_row()
        r_aut["autor"] = blocked_author
        ok_aut, _ = validate_ledger_row(r_aut)
        assert ok_aut is False

    # 4. Exige triple jurisdicción (Chile + EE.UU. + UE) para dominio público
    r_pd = _valid_base_row()
    r_pd["pd_chile"] = "no"
    ok_pd, errs_pd = validate_ledger_row(r_pd)
    assert ok_pd is False
    assert any("pd_chile" in e for e in errs_pd)

    # 5. Obras >= 2022 exigen evidencia de autoría humana explícita (§11.3)
    r_2025 = _valid_base_row()
    r_2025["tipo_procedencia"] = "propio_equipo"
    r_2025["licencia_spdx"] = "CC-BY-4.0"
    r_2025["fecha_publicacion_original"] = "2025"
    r_2025["evidencia_autoria"] = ""
    ok_2025, _ = validate_ledger_row(r_2025)
    assert ok_2025 is False


def test_gutenberg_boilerplate_and_trademark_completely_removed() -> None:
    raw = (
        "The Project Gutenberg eBook of Cuentos Populares\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK CUENTOS POPULARES ***\n"
        "Produced by Distributed Proofreaders for Project Gutenberg.\n\n"
        "Había una vez en un valle tranquilo de Rancagua un viejo molino de agua tibia.\n"
        "Los pajaritos cantaban suave al caer el sol sobre los álamos.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK CUENTOS POPULARES ***\n"
        "Updated editions will replace the previous one--the old editions on Gutenberg.\n"
    )
    cleaned = strip_gutenberg_boilerplate_and_trademark(raw)
    assert "gutenberg" not in cleaned.lower()
    assert "molino de agua tibia" in cleaned


def test_wikisource_prose_and_bedtime_violence_filter() -> None:
    html = (
        '<div class="mw-parser-output">'
        '<table class="ws-noexport"><tr><td>Navegación edición moderna</td></tr></table>'
        "<p>En el bosque sureño caminaba despacito una pequeña cierva junto al estero cristalino y tibio, "
        "escuchando el canto suave de los grillos mientras las hojas de arrayán brillaban bajo la luna tranquila "
        "y el aire olía a tierra húmeda y miel silvestre en completa calma nocturna para descansar muy bien "
        "toda la noche junto a su mamá cierva en el claro tibio del sur.</p>"
        "<p>De pronto salió un cazador con su escopeta llena de sangre y disparó un balazo terrible.</p>"
        "</div>"
    )
    prose = extract_wikisource_clean_prose(html)
    assert "Navegación" not in prose
    passages = extract_bedtime_safe_passages(
        clean_text=prose,
        work_id="LAT-PD-TEST",
        titulo="Cuento prueba",
        autor="Horacio Quiroga",
        idioma="es",
        min_words=45,
        max_words=180,
    )
    assert len(passages) == 1
    assert "escopeta" not in passages[0]["texto"].lower()
    assert "estero cristalino" in passages[0]["texto"]


def test_chilean_lexical_guard_and_third_person_linter() -> None:
    biblia = build_bible_dict_from_child(
        alias="Mateo",
        age_years=6,
        axes={
            "lenguaje": "agil_visual",
            "estructura": "enganche_rampa_calma",
            "estimulo_sonoro": "dinamico_a_calmo",
            "foco_tematico": "misiones_hiperfoco",
        },
        special_interests=["relojes de arena"],
        valores=["compartir turnos"],
        temas_evitar=[],
        closing_phrase="te queremos mucho",
        extra_bible={"palabras_minimas": 40, "palabras_maximas": 300},
    )

    bad_story = (
        "Papá y yo salimos con Mateo mirando relojes de arena. Mateo se puso su chamarra roja junto a la "
        "alberca para platicar un rato y tomar zumo con popote mientras la Canela caminaba suave porque "
        "en casa te queremos mucho."
    )
    qa_bad = analizar_cuento_multinino(bad_story, biblia)
    assert qa_bad["valido"] is False
    joined_errs = " | ".join(qa_bad["errores"])
    assert "chamarra" in joined_errs
    assert "alberca" in joined_errs
    assert "platicar" in joined_errs
    assert "1ª persona" in joined_errs

    good_story = (
        "Mateo giró su reloj de arena dorada en el corredor de madera mientras la Canela caminaba tranquila "
        "a su lado. Se puso su chaqueta tibia cerca de la piscina cubierta para conversar bajito y tomar "
        "jugo con bombilla, sintiéndose muy seguro porque el agua sonaba suave y en casa te queremos mucho."
    )
    qa_good = analizar_cuento_multinino(good_story, biblia)
    assert qa_good["valido"] is True, qa_good["errores"]


def test_qlora_preflight_blocks_tampered_sha256_and_non_apache_models(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    dataset_dir = tmp_path / "ds"

    clean_bytes = (
        "En las colinas suaves del valle central caminaba tranquilo el pequeño pastor escuchando el "
        "agua clara del canal mientras las estrellas se encendían una por una en silencio."
    ).encode("utf-8")
    txt_file = corpus_dir / "obra_limpia.txt"
    txt_file.write_bytes(clean_bytes)

    row = _valid_base_row()
    row["sha256_archivo"] = sha256_bytes(clean_bytes)

    with (corpus_dir / "licencias_corpus.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS_V1)
        writer.writeheader()
        writer.writerow(row)

    (corpus_dir / "pasajes_nocturnos_seguros.jsonl").write_text(
        json.dumps(
            {
                "work_id": row["id"],
                "titulo": row["titulo"],
                "autor": row["autor"],
                "idioma": "es-CL",
                "palabras": 28,
                "sha256": sha256_bytes(clean_bytes),
                "texto": clean_bytes.decode("utf-8"),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # 1. Pasa verificación intacta
    verified = verify_corpus_integrity_against_ledger(corpus_dir)
    assert len(verified) == 1
    stats = build_sft_and_dpo_datasets(corpus_dir, dataset_dir)
    assert stats["sft_train_examples"] == 1
    assert stats["dpo_preference_pairs"] >= 3

    # 2. Bloquea modelo base con licencia no Apache-2.0 (ej. Qwen2.5-3B research license)
    bad_args = argparse.Namespace(
        base_model="Qwen/Qwen2.5-3B-Instruct",
        corpus_dir=str(corpus_dir),
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        lora_r=8,
        lora_alpha=16,
        verify_only=True,
    )
    with pytest.raises(PermissionError, match="Apache-2.0"):
        run_qlora_training(bad_args)

    # 3. Bloquea archivo adulterado en disco (SHA-256 mismatch)
    txt_file.write_bytes(clean_bytes + b" texto inyectado sin auditar")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_corpus_integrity_against_ledger(corpus_dir)
