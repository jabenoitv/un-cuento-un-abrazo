# -*- coding: utf-8 -*-
"""Pruebas Automatizadas del Pipeline de Entrenamiento en Kaggle GPU (`T4 x2`)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import wave
from pathlib import Path

import pytest

from roanso_platform.corpus_pipeline import LEDGER_COLUMNS_V1, sha256_bytes
from training.kaggle_orchestrator import (
    assert_safe_roanso_slug,
    load_kaggle_credentials,
    stage_slm_kaggle_bundle,
    stage_tts_kaggle_bundle,
)
from training.kaggle_qlora_kernel import (
    compute_kaggle_gpu_profile,
    run_kaggle_qlora_pipeline,
)
from training.kaggle_tts_voice_kernel import run_kaggle_voice_preflight_or_train


def _write_mini_verified_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    prose = (
        "En la tarde tranquila de Rancagua, Martín compartió sus barquitos de papel junto al estero "
        "mientras su amiga Sofía entendía sus intenciones con calma y el sol bajaba suave."
    ).encode("utf-8")
    (corpus_dir / "obra_cl.txt").write_bytes(prose)

    row = {col: "NA" for col in LEDGER_COLUMNS_V1}
    row.update(
        {
            "id": "CL-PD-KAGGLE-01",
            "titulo": "Cuentos populares en Chile",
            "autor": "Ramón A. Laval",
            "autor_muerte": "1929",
            "licencia_spdx": "PDM-1.0",
            "pd_chile": "si",
            "pd_eeuu": "si",
            "pd_ue": "si",
            "idioma": "es-CL",
            "palabras": "30",
            "uso_permitido": "entrenamiento",
            "sintetico": "no",
            "fecha_publicacion_original": "1923",
            "tipo_procedencia": "dominio_publico",
            "evidencia_autoria": "Edición 1923",
            "sha256_archivo": sha256_bytes(prose),
        }
    )
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
                "palabras": 30,
                "sha256": sha256_bytes(prose),
                "texto": prose.decode("utf-8"),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_clean_sine_wav(wav_path: Path, sample_rate: int = 24000, seconds: float = 1.0, amp: float = 0.25) -> None:
    n_frames = int(sample_rate * seconds)
    frames = bytearray()
    for i in range(n_frames):
        val = int(32767 * amp * math.sin(2.0 * math.pi * 220.0 * (i / sample_rate)))
        frames.extend(struct.pack("<h", val))
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))


def test_01_slug_fence_protects_non_roanso_kaggle_projects() -> None:
    assert assert_safe_roanso_slug("roanso-cuentos-chileno-sft-v1") == "roanso-cuentos-chileno-sft-v1"
    for protected_other_slug in ("proyecto-externo-industrial", "corpus-ajeno-privado", "otro-proyecto"):
        with pytest.raises(PermissionError, match="roanso-"):
            assert_safe_roanso_slug(protected_other_slug)


def test_02_kgat_bearer_credentials_bridge_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    fake_json = tmp_path / "kaggle.json"
    fake_json.write_text(
        json.dumps({"username": "demo_roanso_org", "key": "KGAT_00000000000000000000000000000000"}),
        encoding="utf-8",
    )
    creds = load_kaggle_credentials(kaggle_json_path=fake_json)
    assert creds.username == "demo_roanso_org"
    assert creds.is_bearer_kgat is True
    assert os.environ.get("KAGGLE_API_TOKEN") == "KGAT_00000000000000000000000000000000"


def test_03_stage_slm_bundle_and_t4_x2_preflight(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    staging_dir = tmp_path / "staging"
    _write_mini_verified_corpus(corpus_dir)

    staged = stage_slm_kaggle_bundle(
        corpus_dir=corpus_dir,
        staging_root=staging_dir,
        kaggle_username="roanso_lab",
    )
    assert staged["verified_works"] == 1
    assert staged["dpo_preference_pairs"] == 8

    ds_meta = json.loads((Path(staged["dataset_dir"]) / "dataset-metadata.json").read_text(encoding="utf-8"))
    kn_meta = json.loads((Path(staged["kernel_dir"]) / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert ds_meta["isPrivate"] is True
    assert kn_meta["is_private"] is True
    assert kn_meta["enable_gpu"] is True
    assert kn_meta["enable_internet"] is False
    assert kn_meta["dataset_sources"] == ["roanso_lab/roanso-corpus-chileno-sft"]
    assert kn_meta["model_sources"] == ["qwen-lm/qwen2.5/transformers/7b-instruct/1"]

    # Verifica preflight del kernel QLoRA sobre el dataset empaquetado
    preflight = run_kaggle_qlora_pipeline(
        argparse.Namespace(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            dataset_dir=staged["dataset_dir"],
            output_dir=str(tmp_path / "out"),
            epochs=2,
            batch_size=2,
            grad_accum=4,
            lora_r=16,
            lora_alpha=32,
            run_dpo=True,
            verify_only=True,
        )
    )
    assert preflight["license"] == "Apache-2.0"
    assert preflight["dpo_rows"] == 8

    # Verifica perfil hardware Tesla T4 x2 (sm_75 -> fp16=True, device_map={"": 0}, eval en cuda:1)
    t4_profile = compute_kaggle_gpu_profile(has_cuda=True, major_cc=7, device_count=2)
    assert t4_profile["fp16"] is True
    assert t4_profile["bf16"] is False
    assert t4_profile["device_map"] == {"": 0}
    assert t4_profile["secondary_eval_device"] == "cuda:1"


def test_04_private_voice_actor_kernel_enforces_adult_commercial_contract(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging_tts"
    tts_staged = stage_tts_kaggle_bundle(
        staging_root=staging_dir,
        kaggle_username="roanso_lab",
        private_voice_dataset_slug="roanso-voces-estudio-chile-v1",
    )
    kn_meta = json.loads((Path(tts_staged["kernel_dir"]) / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert kn_meta["is_private"] is True
    assert kn_meta["enable_gpu"] is True

    voice_ds = tmp_path / "voice_ds"
    voice_ds.mkdir()
    _write_clean_sine_wav(voice_ds / "toma_01.wav")
    contract_hash = hashlib.sha256(b"Contrato Exclusivo Actor Chileno ROANSO SpA").hexdigest()

    # 1. Bloquea voz de menor
    (voice_ds / "actor_voice_license.json").write_text(
        json.dumps(
            {
                "actor_id": "test_minor",
                "is_minor": True,
                "adult_actor_verified": False,
                "exclusive_children_commercial_license": True,
                "signed_contract_sha256": contract_hash,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="adultas"):
        run_kaggle_voice_preflight_or_train(
            argparse.Namespace(
                voice_dataset_dir=str(voice_ds),
                tts_backbone="canopylabs/orpheus-3b-0.1-ft",
                output_dir=str(tmp_path / "tts_out"),
                verify_only=True,
            )
        )

    # 2. Aprueba contrato comercial adulto exclusivo con toma WAV limpia
    (voice_ds / "actor_voice_license.json").write_text(
        json.dumps(
            {
                "actor_id": "actor_cl_femenino_01",
                "locale": "es-CL",
                "is_minor": False,
                "adult_actor_verified": True,
                "exclusive_children_commercial_license": True,
                "signed_contract_sha256": contract_hash,
            }
        ),
        encoding="utf-8",
    )
    report = run_kaggle_voice_preflight_or_train(
        argparse.Namespace(
            voice_dataset_dir=str(voice_ds),
            tts_backbone="canopylabs/orpheus-3b-0.1-ft",
            output_dir=str(tmp_path / "tts_out"),
            verify_only=True,
        )
    )
    assert report["acoustic_audit"]["studio_ready_for_commercial_lora"] is True
    assert report["backbone_license"] == "Apache-2.0"
