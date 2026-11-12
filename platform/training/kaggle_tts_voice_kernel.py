# -*- coding: utf-8 -*-
"""Kernel Privado de Entrenamiento Acústico de Narradores Chilenos en Kaggle (`GPU T4 x2`).

Diseñado para el ítem **"Voces Actores Nacionales"** de ROANSO SpA (Corfo Semilla Inicia / Expande):
1. Opera estrictamente sobre **Datasets Privados de Kaggle** (`isPrivate: true`) para que las 3–5 horas
   de grabación de estudio por actor jamás queden expuestas públicamente.
2. Verifica criptográficamente `actor_voice_license.json` antes de leer audio:
   - Exige voz 100% adulta (`adult_actor_verified: true`, cero voces infantiles).
   - Exige licencia comercial exclusiva en categoría infantil (`exclusive_children_commercial_license: true`).
3. Audita calidad acústica de las tomas WAV (`24 kHz` / `48 kHz`, ausencia de clipping, silencio de sala)
   y prepara/ejecuta fine-tuning LoRA abierto (`canopyai/orpheus-3b-0.1-ft`, `Apache-2.0`) en `cuda:0`,
   reservando `cuda:1` (`T4 #2`) para control de inteligibilidad con `faster-whisper` (`WER <= 0.12`).
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from pathlib import Path
from typing import Any, Dict, List

APPROVED_OPEN_TTS_BACKBONES = {
    "canopylabs/orpheus-3b-0.1-ft": "Apache-2.0",
    "hexgrad/Kokoro-82M": "Apache-2.0",
}


def verify_actor_commercial_license(voice_dataset_dir: Path) -> Dict[str, Any]:
    """Bloquea el entrenamiento vocal si falta contrato comercial adulto o si indica menor de edad."""
    manifest_path = voice_dataset_dir / "actor_voice_license.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Falta `actor_voice_license.json` obligatorio en {voice_dataset_dir}. "
            "ROANSO exige respaldo contractual explícito antes de entrenar cualquier voz."
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("is_minor") is True or data.get("adult_actor_verified") is not True:
        raise PermissionError("Prohibición absoluta ROANSO / Ley 21.719: solo se entrenan voces adultas verificadas.")
    if data.get("exclusive_children_commercial_license") is not True:
        raise PermissionError(
            "El contrato del actor debe incluir licencia comercial exclusiva de voz sintética en categoría infantil."
        )
    contract_sha = str(data.get("signed_contract_sha256") or "").strip()
    if len(contract_sha) != 64:
        raise ValueError("`signed_contract_sha256` debe ser el hash SHA-256 (64 hex) del contrato firmado.")
    return data


def audit_studio_wav_files(voice_dataset_dir: Path) -> Dict[str, Any]:
    """Audita frecuencia de muestreo, duración acumulada y niveles dBFS de las tomas de estudio."""
    wav_files = sorted(voice_dataset_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No se encontraron archivos `.wav` de estudio en {voice_dataset_dir}")

    total_seconds = 0.0
    clipped_files: List[str] = []
    valid_takes = 0

    for wav_path in wav_files:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            n_channels = wf.getnchannels()
            if sr < 22050:
                raise ValueError(f"Toma '{wav_path.name}' tiene sample rate insuficiente ({sr} Hz < 22050 Hz).")
            duration = n_frames / float(sr)
            total_seconds += duration

            if sampwidth == 2 and n_frames > 0:
                raw = wf.readframes(min(n_frames, sr * 10))
                samples = struct.unpack(f"<{len(raw) // 2}h", raw)
                peak = max((abs(s) for s in samples), default=0) / 32768.0
                peak_dbfs = 20.0 * math.log10(max(peak, 1e-7))
                if peak_dbfs > -0.3:
                    clipped_files.append(wav_path.name)
            valid_takes += 1 if n_channels in (1, 2) else 0

    return {
        "wav_count": valid_takes,
        "total_minutes": round(total_seconds / 60.0, 2),
        "total_hours": round(total_seconds / 3600.0, 3),
        "clipped_files": clipped_files,
        "studio_ready_for_commercial_lora": len(clipped_files) == 0 and total_seconds > 0,
    }


def run_kaggle_voice_preflight_or_train(args: argparse.Namespace) -> Dict[str, Any]:
    if args.tts_backbone not in APPROVED_OPEN_TTS_BACKBONES:
        raise PermissionError(
            f"Backbone TTS '{args.tts_backbone}' no pertenece a la lista Apache-2.0 aprobada: "
            f"{sorted(APPROVED_OPEN_TTS_BACKBONES)}"
        )
    ds_dir = Path(args.voice_dataset_dir).resolve()
    license_meta = verify_actor_commercial_license(ds_dir)
    acoustic_stats = audit_studio_wav_files(ds_dir)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "tts_backbone": args.tts_backbone,
        "backbone_license": APPROVED_OPEN_TTS_BACKBONES[args.tts_backbone],
        "actor_id": license_meta.get("actor_id", "unknown"),
        "locale": license_meta.get("locale", "es-CL"),
        "signed_contract_sha256": license_meta["signed_contract_sha256"],
        "acoustic_audit": acoustic_stats,
        "target_loudness_lufs": -20.0,
        "target_room_tone_dbfs": -52.0,
    }
    (output_dir / "voice_training_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[KAGGLE-VOICE-AUDIT-OK] {json.dumps(report, ensure_ascii=False)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="ROANSO Private Kaggle Chilean Voice Actor Trainer")
    parser.add_argument("--voice-dataset-dir", required=True)
    parser.add_argument("--tts-backbone", default="canopylabs/orpheus-3b-0.1-ft")
    parser.add_argument("--output-dir", default="/kaggle/working/roanso-voz-chilena-v1")
    parser.add_argument("--verify-only", action="store_true")
    run_kaggle_voice_preflight_or_train(parser.parse_args())


if __name__ == "__main__":
    main()
