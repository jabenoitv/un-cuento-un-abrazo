# -*- coding: utf-8 -*-
"""Nocturnal Storytelling Synthesizer with Sleep Ramp, Cosine Crossfades & Room Tone — ROANSO SpA.

Eliminates two acoustic failure modes common in generic TTS pipelines:
1. Instead of splitting on internal commas and concatenating digital silence (`np.zeros`—which
   triggers abrupt noise-gate drops that startle sound-sensitive sleepers), groups 1-2 complete
   breathable sentences, applies 35 ms cosine fades, and mixes continuous warm brown room tone
   at `-52 dBFS`.
2. Adapts tempo, pause cadence, and final attenuation to the child's `estimulo_sonoro` axis:
   - `monovoz_estable`: constant `0.89x` pace, single steady vocal embedding, uniform volume.
   - `dinamico_a_calmo` / `multivoz_personajes` (*Sleep Ramp*): physiological deceleration from
     `0.96x` (curious opening) -> `0.92x` (body) -> `0.86x` (sleep whisper closing).

Usage:
  python voces/sintetizar_cuento_nocturno.py <nino_id> <script.txt> <output.mp3> [narrator_slug]
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VOCES_DIR = Path(__file__).resolve().parent
MOTOR_DIR = VOCES_DIR.parent / "motor"
PERFILES_NARRADOR_DIR = VOCES_DIR / "perfiles_narrador"
OMNIVOICE_DIR = Path(os.environ.get("OMNIVOICE_DIR", str(VOCES_DIR / "omnivoice_assets")))
ROOM_TONE_DEFAULT = OMNIVOICE_DIR / "room_tone_estable.wav"
SR = 24000

sys.path.insert(0, str(MOTOR_DIR))
from pipeline_multinino import cargar_biblia, normalizar_perfil_sensorial  # noqa: E402


def agrupar_oraciones_respirables(texto: str, max_chars: int = 260) -> List[str]:
    """Groups 1 to 2 complete sentences without breaking on internal commas."""
    oraciones = [s.strip() for s in re.findall(r"[^.!?…]+[.!?…]*", texto.strip()) if s.strip()]
    grupos: List[str] = []
    actual = ""
    for oracion in oraciones:
        if not actual:
            actual = oracion
        elif len(actual) + 1 + len(oracion) <= max_chars:
            actual = f"{actual} {oracion}"
        else:
            grupos.append(actual)
            actual = oracion
    if actual:
        grupos.append(actual)
    return grupos


def resolver_prompts_narrador(narrador_slug: str) -> Dict[str, Path]:
    """Resolves `.pt` embeddings for opening, body, and sleep closing registers."""
    carpeta = PERFILES_NARRADOR_DIR / narrador_slug
    base = carpeta / "desarrollo_sereno.pt"
    if not base.exists():
        fallback = OMNIVOICE_DIR / "default_narrator.pt"
        base = fallback
    return {
        "inicio": (carpeta / "inicio_curioso.pt") if (carpeta / "inicio_curioso.pt").exists() else base,
        "desarrollo": base,
        "cierre": (carpeta / "cierre_sueno.pt") if (carpeta / "cierre_sueno.pt").exists() else base,
    }


def generar_room_tone_calido(num_samples: int, sr: int = SR, nivel_dbfs: float = -52.0) -> np.ndarray:
    """Loads steady room tone or synthesizes warm brown noise at -52 dBFS."""
    target_rms = 10.0 ** (nivel_dbfs / 20.0)
    if ROOM_TONE_DEFAULT.exists():
        rt, rt_sr = sf.read(str(ROOM_TONE_DEFAULT))
        if rt.ndim > 1:
            rt = rt.mean(axis=1)
        rt = np.asarray(rt, dtype=np.float32)
        if rt_sr == sr and len(rt) > sr:
            reps = int(np.ceil(num_samples / len(rt))) + 1
            looped = np.tile(rt, reps)[:num_samples]
            rms_actual = float(np.sqrt(np.mean(looped ** 2)) + 1e-9)
            return looped * (target_rms / rms_actual)

    rng = np.random.default_rng(19)
    white = rng.standard_normal(num_samples).astype(np.float32)
    brown = np.cumsum(white)
    brown -= np.mean(brown)
    rms_actual = float(np.sqrt(np.mean(brown ** 2)) + 1e-9)
    return (brown * (target_rms / rms_actual)).astype(np.float32)


def unir_con_crossfade_y_room_tone(
    tramos_y_pausas: List[Tuple[np.ndarray, float]],
    sr: int = SR,
    fade_ms: float = 35.0,
) -> np.ndarray:
    """Applies 35 ms cosine fades per breath group and mixes continuous -52 dBFS room tone."""
    fade_n = int(sr * (fade_ms / 1000.0))
    partes: List[np.ndarray] = []

    for idx, (audio, pausa_s) in enumerate(tramos_y_pausas):
        seg = np.asarray(audio, dtype=np.float32).copy()
        if len(seg) > 2 * fade_n:
            ventana_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_n, dtype=np.float32)))
            ventana_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_n, dtype=np.float32)))
            seg[:fade_n] *= ventana_in
            seg[-fade_n:] *= ventana_out
        partes.append(seg)
        if idx < len(tramos_y_pausas) - 1 and pausa_s > 0:
            partes.append(np.zeros(int(sr * pausa_s), dtype=np.float32))

    partes.append(np.zeros(int(sr * 1.5), dtype=np.float32))
    voz_completa = np.concatenate(partes)
    piso = generar_room_tone_calido(len(voz_completa), sr=sr, nivel_dbfs=-52.0)
    return np.clip(voz_completa + piso, -0.98, 0.98)


def parametros_tramo(progreso: float, estimulo_sonoro: str) -> Tuple[str, float, float, float]:
    """Returns (voice_stage, speed, pause_seconds, linear_gain) for progress in [0..1]."""
    if estimulo_sonoro == "monovoz_estable":
        return ("desarrollo", 0.89, 0.85, 1.0)
    if progreso < 0.28:
        return ("inicio", 0.96, 0.58, 1.0)
    if progreso < 0.72:
        return ("desarrollo", 0.92, 0.78, 0.96)
    aten = 0.88 if progreso < 0.92 else 0.80
    return ("cierre", 0.86, 1.12, aten)


def sintetizar_cuento(nino_id: str, guion_path: Path, salida_path: Path, narrador_slug: str = "narrador_sur") -> None:
    import torch
    from omnivoice import OmniVoice, VoiceClonePrompt

    biblia = cargar_biblia(nino_id)
    perfil = normalizar_perfil_sensorial(biblia)
    estimulo = perfil["estimulo_sonoro"]
    texto = guion_path.read_text(encoding="utf-8").strip()
    bloques = agrupar_oraciones_respirables(texto)

    prompts_paths = resolver_prompts_narrador(narrador_slug)
    prompts_cargados = {k: VoiceClonePrompt.load(str(p)) for k, p in prompts_paths.items()}

    tts = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    tramos: List[Tuple[np.ndarray, float]] = []
    total = max(1, len(bloques))
    for i, bloque in enumerate(bloques):
        progreso = i / max(1, total - 1)
        etapa, speed, pausa_s, gain = parametros_tramo(progreso, estimulo)
        audio_seg = tts.generate(
            text=bloque,
            language="es",
            voice_clone_prompt=prompts_cargados[etapa],
            speed=speed,
        )[0]
        tramos.append((np.asarray(audio_seg, dtype=np.float32) * gain, pausa_s))

    mezcla = unir_con_crossfade_y_room_tone(tramos, sr=SR)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_tmp = Path(tmp.name)
    try:
        sf.write(str(wav_tmp), mezcla, SR)
        af_master = "highpass=f=70,lowpass=f=10000,loudnorm=I=-20:TP=-2.0:LRA=6"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_tmp), "-af", af_master, "-b:a", "128k", str(salida_path)],
            check=True,
        )
    finally:
        if wav_tmp.exists():
            wav_tmp.unlink()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python sintetizar_cuento_nocturno.py <nino_id> <script.txt> <output.mp3> [narrator_slug]")
        sys.exit(1)
    sintetizar_cuento(
        sys.argv[1],
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        narrador_slug=sys.argv[4] if len(sys.argv) >= 5 else "narrador_sur",
    )
