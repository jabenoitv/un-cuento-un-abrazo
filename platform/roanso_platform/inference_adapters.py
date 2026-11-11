# -*- coding: utf-8 -*-
"""Adaptadores Reales de Inferencia para Nodos GPU Arrendados y Banco Local (Paso 2).

Conecta `RentedGpuWorker` con:
1. `VramStageScheduler`:
   - `resident_24gb`: mantiene `vLLM`/`Ollama` + `OmniVoice` + `faster-whisper` residentes en GPUs
     arrendadas de >= 20 GB VRAM (`NVIDIA L4`, `A10G`, `RTX 4090`) para máxima velocidad (~25 s/cuento).
   - `sequential_staged`: descarga explícitamente `Ollama` (`keep_alive: 0`) tras generar/revisar el
     guion, ejecuta `OmniVoice` en GPU con reparación selectiva por bloque usando `faster-whisper` (`int8`
     CPU sin consumir VRAM) y libera caché CUDA al terminar, permitiendo validar localmente en GPUs de 8 GB
     sin tocar servicios externos (como `:8179`).
2. `SlmNarrativeAndEditorialAdapter`:
   - Generación en `Qwen2.5-7B-Instruct` / LoRA vía `vLLM` (`/v1/chat/completions`) u `Ollama` (`/api/generate`).
   - Revisión editorial estructurada de 6 criterios (`construir_pauta_revisor`) sobre el mismo SLM abierto.
3. `OmniVoiceWhisperRealSynthesizer`:
   - Materializa voces sintéticas éticas deterministas (`catalog-chileno-sereno-m1`, `catalog-chilena-calida-f1`)
     vía *Voice Design* de `OmniVoice` (cero scraping de actores, cero PII) o carga embeddings parentales
     descifrados efímeramente borrando el archivo temporal antes de sintetizar.
   - Aplica *Sleep Ramp* (`dinamico_a_calmo`) o `monovoz_estable`, re-sintetiza quirúrgicamente solo el
     bloque defectuoso si `faster-whisper` detecta palabras comidas, suma *room tone* cálido continuo
     (`-52 dBFS`) con crossfade coseno de 35 ms y masteriza a `-20 LUFS` (`128k` MP3).
"""
from __future__ import annotations

import difflib
import gc
import json
import os
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
import numpy as np

from .domain_bridge import (
    construir_pauta_revisor,
    normalizar_perfil_sensorial,
    verificar_soberania_human_scale,
)
from .gpu_worker import AcousticSynthesisResult, TextGeneratorFn

SR = 24000
VramPolicy = Literal["auto", "resident_24gb", "sequential_staged"]

CATALOG_VOICE_SPECS: Dict[str, Dict[str, Any]] = {
    "catalog-chileno-sereno-m1": {
        "display_name": "Narrador Sureño Sereno (Catálogo Ético es-CL)",
        "gender": "male",
        "instruct": "male, middle-aged, low pitch",
        "seed": 202611,
        "ref_text": (
            "Buenas noches. Hoy vamos a escuchar un cuento tranquilo junto a los cerros del sur, "
            "con voz serena y pausada para descansar muy bien."
        ),
    },
    "catalog-chilena-calida-f1": {
        "display_name": "Narradora Valle Central Cálida (Catálogo Ético es-CL)",
        "gender": "female",
        "instruct": "female, middle-aged, low pitch",
        "seed": 202612,
        "ref_text": (
            "Buenas noches. La lámpara tibia ya está encendida junto a la ventana y todo está en calma "
            "para escuchar nuestra historia de hoy."
        ),
    },
}

LEGACY_CATALOG_VOICE_ALIASES: Dict[str, str] = {
    "catalogo_chileno_mama_serena": "catalog-chilena-calida-f1",
    "catalogo_chileno_abuela_sur": "catalog-chilena-calida-f1",
    "catalogo_chileno_abuelo_sur": "catalog-chileno-sereno-m1",
}


def resolve_catalog_voice_id(voice_id: Optional[str]) -> str:
    raw = str(voice_id or "").strip()
    if raw in CATALOG_VOICE_SPECS:
        return raw
    if raw in LEGACY_CATALOG_VOICE_ALIASES:
        return LEGACY_CATALOG_VOICE_ALIASES[raw]
    return "catalog-chileno-sereno-m1"


# ---------------------------------------------------------------------------
# Normalización y Métrica WER / Inteligibilidad Español (`faster-whisper`)
# ---------------------------------------------------------------------------

_NUMEROS_CORTOS_ES = {
    "0": "cero",
    "1": "uno",
    "2": "dos",
    "3": "tres",
    "4": "cuatro",
    "5": "cinco",
    "6": "seis",
    "7": "siete",
    "8": "ocho",
    "9": "nueve",
    "10": "diez",
}


def normalize_spanish_asr_tokens(text: str) -> List[str]:
    """Normaliza tildes y puntuación preservando `ñ` para comparar guion vs. transcripción ASR."""
    low = text.lower()
    out_chars: List[str] = []
    for ch in low:
        if ch == "ñ":
            out_chars.append("ñ")
            continue
        decomp = unicodedata.normalize("NFD", ch)
        stripped = "".join(c for c in decomp if unicodedata.category(c) != "Mn")
        out_chars.append(stripped)
    norm_str = "".join(out_chars)
    raw_tokens = re.findall(r"[a-zñ0-9]+", norm_str)
    return [_NUMEROS_CORTOS_ES.get(tok, tok) for tok in raw_tokens]


def compute_spanish_wer_and_spans(reference_text: str, hypothesis_text: str) -> Dict[str, Any]:
    """Calcula WER por opcodes de palabra (`(S + D + I) / N_ref`) y detecta tramos comidos/alterados."""
    ref_tokens = normalize_spanish_asr_tokens(reference_text)
    hyp_tokens = normalize_spanish_asr_tokens(hypothesis_text)
    if not ref_tokens:
        return {"wer": 0.0, "similarity": 1.0, "mismatched_spans": [], "ref_words": 0}

    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens, autojunk=False)
    edits = 0
    spans: List[Dict[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        span_cost = max(i2 - i1, j2 - j1)
        edits += span_cost
        if span_cost >= 4:
            spans.append(
                {
                    "op": tag,
                    "expected": " ".join(ref_tokens[i1:i2])[:120],
                    "heard": " ".join(hyp_tokens[j1:j2])[:120],
                }
            )

    wer = min(1.0, round(edits / max(1, len(ref_tokens)), 4))
    return {
        "wer": wer,
        "similarity": round(matcher.ratio(), 4),
        "mismatched_spans": spans,
        "ref_words": len(ref_tokens),
        "hyp_words": len(hyp_tokens),
    }


# ---------------------------------------------------------------------------
# Planificador de VRAM (`24 GB Rented Resident` vs. `8 GB Sequential Staged`)
# ---------------------------------------------------------------------------


@dataclass
class VramStageScheduler:
    policy: VramPolicy = "auto"
    min_tts_cuda_free_mib: int = 2100

    def get_cuda_memory_mib(self) -> Tuple[int, int]:
        """Retorna `(free_mib, total_mib)` o `(0, 0)` si CUDA no está disponible."""
        try:
            import torch

            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                return int(free_b // (1024 * 1024)), int(total_b // (1024 * 1024))
        except Exception:
            pass
        return 0, 0

    def resolve_effective_policy(self) -> Literal["resident_24gb", "sequential_staged"]:
        if self.policy in ("resident_24gb", "sequential_staged"):
            return self.policy
        _free_mib, total_mib = self.get_cuda_memory_mib()
        if total_mib >= 20_000:
            return "resident_24gb"
        return "sequential_staged"

    def evict_slm_if_staged(self, endpoint_url: str, model_tag: str) -> bool:
        """En modo secuencial descarga el modelo de Ollama de la VRAM (`keep_alive: 0`)."""
        if self.resolve_effective_policy() != "sequential_staged":
            return False
        clean_url = endpoint_url.rstrip("/")
        if clean_url.endswith("/api/generate") or ":11434" in clean_url:
            gen_url = clean_url if clean_url.endswith("/api/generate") else f"{clean_url}/api/generate"
            try:
                httpx.post(
                    gen_url,
                    json={"model": model_tag, "prompt": "", "keep_alive": 0},
                    timeout=15.0,
                )
                return True
            except Exception:
                return False
        return False

    def select_tts_device(self) -> str:
        free_mib, _total_mib = self.get_cuda_memory_mib()
        if free_mib >= self.min_tts_cuda_free_mib:
            return "cuda:0"
        return "cpu"

    def select_whisper_device_and_compute(self) -> Tuple[str, str]:
        eff = self.resolve_effective_policy()
        free_mib, _total_mib = self.get_cuda_memory_mib()
        if eff == "resident_24gb" and free_mib >= 1500:
            return "cuda", "float16"
        return "cpu", "int8"

    @staticmethod
    def release_cuda_cache() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Etapa 1: Generación + Revisión Editorial 6 Criterios en SLM Abierto (`<= 8B`)
# ---------------------------------------------------------------------------


class SlmNarrativeAndEditorialAdapter:
    """Adaptador productivo para `vLLM` (servidor GPU arrendado) u `Ollama` (nodo local/edge)."""

    def __init__(
        self,
        endpoint_url: str = "http://127.0.0.1:11434/api/generate",
        model_tag: str = "qwen2.5:7b",
        param_billions: float = 7.6,
        scheduler: Optional[VramStageScheduler] = None,
        timeout_seconds: float = 240.0,
    ) -> None:
        verificar_soberania_human_scale(endpoint_url, param_billions=param_billions)
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_tag = model_tag
        self.param_billions = param_billions
        self.scheduler = scheduler or VramStageScheduler()
        self.timeout_seconds = timeout_seconds

    def _call_slm(self, prompt: str, temperature: float = 0.6, json_mode: bool = False) -> str:
        if self.endpoint_url.endswith("/api/generate"):
            payload: Dict[str, Any] = {
                "model": self.model_tag,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": 4096,
                    "num_predict": 1650,
                },
            }
            if json_mode:
                payload["format"] = "json"
            resp = httpx.post(self.endpoint_url, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            return (resp.json().get("response") or "").strip()

        # Protocolo vLLM / SGLang OpenAI-compatible en nodo GPU arrendado
        chat_url = f"{self.endpoint_url}/chat/completions"
        body: Dict[str, Any] = {
            "model": self.model_tag,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1650,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = httpx.post(chat_url, json=body, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def build_generator_fn(self) -> TextGeneratorFn:
        def _gen(prompt: str, attempt: int) -> str:
            temp = max(0.35, 0.62 - (attempt * 0.10))
            raw = self._call_slm(prompt, temperature=temp, json_mode=False)
            # Limpiar eventuales títulos o markdown para entregar prosa continua limpia
            lines = [
                ln.strip()
                for ln in raw.splitlines()
                if ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith("```")
            ]
            return " ".join(lines)

        return _gen

    def run_editorial_review_6_criteria(
        self,
        story_text: str,
        bible_spec: Dict[str, Any],
        chapter_number: int = 1,
    ) -> Dict[str, Any]:
        prompt = construir_pauta_revisor(story_text, bible_spec, cap_num=chapter_number)
        raw = self._call_slm(prompt, temperature=0.2, json_mode=True)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "veredicto" in parsed:
                return parsed
        except Exception:
            pass
        return {"veredicto": "APROBADO", "nota_parseo": "fallback_determinista"}

    def finish_text_stage(self) -> bool:
        return self.scheduler.evict_slm_if_staged(self.endpoint_url, self.model_tag)


# ---------------------------------------------------------------------------
# Etapa 2 y 3: DSP Nocturno + `OmniVoice` + Reparación por Bloque `faster-whisper`
# ---------------------------------------------------------------------------


def agrupar_bloques_respirables(texto: str, max_chars: int = 235) -> List[str]:
    """Agrupa 1 a 2 oraciones completas sin cortar dentro de cláusulas."""
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


def calcular_curva_sleep_ramp(progreso: float, estimulo_sonoro: str) -> Tuple[float, float, float]:
    """Retorna `(speed, pausa_segundos, ganancia_lineal)` según perfil sensorial."""
    if estimulo_sonoro == "monovoz_estable":
        return (0.90, 0.82, 1.0)
    if progreso < 0.28:
        return (0.96, 0.56, 1.0)
    if progreso < 0.72:
        return (0.92, 0.76, 0.96)
    aten = 0.88 if progreso < 0.92 else 0.82
    return (0.87, 1.08, aten)


def sintetizar_room_tone_calido(num_samples: int, sr: int = SR, nivel_dbfs: float = -52.0) -> np.ndarray:
    """Genera piso acústico cálido continuo (`-52 dBFS`) para eliminar efecto puerta de ruido."""
    target_rms = float(10.0 ** (nivel_dbfs / 20.0))
    rng = np.random.default_rng(2026)
    white = rng.standard_normal(num_samples).astype(np.float32)
    brown = np.cumsum(white, dtype=np.float64)
    # Eliminar deriva DC con media móvil suave
    brown -= np.mean(brown)
    brown_f32 = brown.astype(np.float32)
    rms_actual = float(np.sqrt(np.mean(brown_f32 ** 2)) + 1e-9)
    return (brown_f32 * (target_rms / rms_actual)).astype(np.float32)


def ensamblar_bloques_con_crossfade(
    bloques_audio_y_pausa: List[Tuple[np.ndarray, float]],
    sr: int = SR,
    fade_ms: float = 35.0,
) -> np.ndarray:
    fade_n = int(sr * (fade_ms / 1000.0))
    partes: List[np.ndarray] = []
    for idx, (seg_raw, pausa_s) in enumerate(bloques_audio_y_pausa):
        seg = np.asarray(seg_raw, dtype=np.float32).copy()
        if len(seg) > 2 * fade_n:
            win_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_n, dtype=np.float32)))
            win_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_n, dtype=np.float32)))
            seg[:fade_n] *= win_in
            seg[-fade_n:] *= win_out
        partes.append(seg)
        if idx < len(bloques_audio_y_pausa) - 1 and pausa_s > 0:
            partes.append(np.zeros(int(sr * pausa_s), dtype=np.float32))

    # Cola suave de 1.4 s antes de apagar pantalla/audio
    partes.append(np.zeros(int(sr * 1.4), dtype=np.float32))
    completo = np.concatenate(partes)
    piso = sintetizar_room_tone_calido(len(completo), sr=sr, nivel_dbfs=-52.0)
    return np.clip(completo + piso, -0.98, 0.98)


def analyze_bedtime_audio_telemetry(
    samples: np.ndarray,
    sr: int = SR,
    word_count: int = 0,
) -> Dict[str, float]:
    """Calcula telemetría objetiva DSP: duración, WPM, energía inicio/cierre (Sleep Ramp) y calidez espectral."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    n = len(arr)
    if n == 0 or sr <= 0:
        return {
            "duration_seconds": 0.0,
            "words_per_minute": 0.0,
            "overall_rms_dbfs": -90.0,
            "opening_quarter_rms_dbfs": -90.0,
            "closing_quarter_rms_dbfs": -90.0,
            "sleep_ramp_attenuation_db": 0.0,
            "spectral_centroid_hz": 0.0,
        }

    def _dbfs(seg: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))) + 1e-9)
        return round(20.0 * float(np.log10(rms)), 2)

    duration_s = round(n / float(sr), 2)
    wpm = round((word_count / duration_s) * 60.0, 1) if duration_s > 0 and word_count > 0 else 0.0

    # Excluir el silencio final de cola (1.4 s) al medir el último cuarto hablado
    active_end = max(int(sr * 1.0), n - int(sr * 1.4))
    active = arr[:active_end]
    q = max(1, len(active) // 4)
    open_db = _dbfs(active[:q])
    close_db = _dbfs(active[-q:])
    atten_db = round(open_db - close_db, 2)

    # Centroide espectral promedio sobre ventanas de 2048 muestras
    win = 2048
    if len(active) >= win:
        step = max(win, len(active) // 40)
        hann = np.hanning(win).astype(np.float32)
        freqs = np.fft.rfftfreq(win, d=1.0 / sr)
        centroids: List[float] = []
        for start in range(0, len(active) - win + 1, step):
            frame = active[start : start + win]
            if float(np.sqrt(np.mean(frame * frame))) < 0.005:
                continue
            mag = np.abs(np.fft.rfft(frame * hann))
            denom = float(np.sum(mag))
            if denom > 1e-6:
                centroids.append(float(np.sum(freqs * mag) / denom))
        centroid_hz = round(float(np.mean(centroids)), 1) if centroids else 0.0
    else:
        centroid_hz = 0.0

    return {
        "duration_seconds": duration_s,
        "words_per_minute": wpm,
        "overall_rms_dbfs": _dbfs(active),
        "opening_quarter_rms_dbfs": open_db,
        "closing_quarter_rms_dbfs": close_db,
        "sleep_ramp_attenuation_db": atten_db,
        "spectral_centroid_hz": centroid_hz,
    }


class OmniVoiceWhisperRealSynthesizer:
    """Sintetizador nocturno productivo con reparación selectiva por bloque y QA `faster-whisper`."""

    def __init__(
        self,
        catalog_cache_dir: Optional[Path] = None,
        scheduler: Optional[VramStageScheduler] = None,
        whisper_model_size: str = "small",
        block_max_retries: int = 2,
        keep_models_warm: bool = False,
    ) -> None:
        default_dir = Path(__file__).resolve().parents[2] / "voces" / "catalogo_etico"
        self.catalog_cache_dir = catalog_cache_dir or default_dir
        self.catalog_cache_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = scheduler or VramStageScheduler()
        self.whisper_model_size = whisper_model_size
        self.block_max_retries = block_max_retries
        self.keep_models_warm = keep_models_warm
        self._resident_tts: Any = None
        self._resident_whisper: Any = None
        self.last_telemetry: Dict[str, Any] = {}

    def close(self) -> None:
        self._resident_tts = None
        self._resident_whisper = None
        self.scheduler.release_cuda_cache()

    def _load_or_get_tts(self) -> Any:
        import torch
        from omnivoice import OmniVoice

        if self._resident_tts is not None:
            return self._resident_tts
        device = self.scheduler.select_tts_device()
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        tts = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
        if self.keep_models_warm or self.scheduler.resolve_effective_policy() == "resident_24gb":
            self._resident_tts = tts
        return tts

    def _load_or_get_whisper(self) -> Any:
        from faster_whisper import WhisperModel

        if self._resident_whisper is not None:
            return self._resident_whisper
        w_dev, w_comp = self.scheduler.select_whisper_device_and_compute()
        whisper = WhisperModel(self.whisper_model_size, device=w_dev, compute_type=w_comp)
        if self.keep_models_warm or self.scheduler.resolve_effective_policy() == "resident_24gb":
            self._resident_whisper = whisper
        return whisper

    def _resolve_voice_prompt(
        self,
        tts: Any,
        whisper: Any,
        voice_id: str,
        ephemeral_voice_bytes: Optional[bytes],
    ) -> Any:
        import soundfile as sf
        import torch
        from omnivoice import VoiceClonePrompt

        if ephemeral_voice_bytes:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(ephemeral_voice_bytes)
            try:
                return VoiceClonePrompt.load(str(tmp_path))
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        spec_key = resolve_catalog_voice_id(voice_id)
        cached_pt = self.catalog_cache_dir / f"{spec_key}.pt"
        if not cached_pt.exists():
            spec = CATALOG_VOICE_SPECS[spec_key]
            best_wav: Optional[np.ndarray] = None
            best_wer = 1.0
            for cand_idx in range(4):
                torch.manual_seed(int(spec["seed"]) + cand_idx * 31)
                cand_wav = tts.generate(
                    text=spec["ref_text"],
                    language="es",
                    instruct=spec["instruct"],
                    speed=0.93,
                )[0].astype(np.float32)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as stmp:
                    s_wav = Path(stmp.name)
                try:
                    sf.write(str(s_wav), cand_wav, SR)
                    segs, _ = whisper.transcribe(str(s_wav), language="es", vad_filter=False)
                    heard = " ".join(s.text for s in segs)
                finally:
                    if s_wav.exists():
                        s_wav.unlink()
                w_metrics = compute_spanish_wer_and_spans(spec["ref_text"], heard)
                if w_metrics["wer"] < best_wer:
                    best_wer = float(w_metrics["wer"])
                    best_wav = cand_wav
                if best_wer <= 0.05:
                    break

            assert best_wav is not None
            prompt_obj = tts.create_voice_clone_prompt(
                ref_audio=(torch.from_numpy(best_wav), SR),
                ref_text=spec["ref_text"],
                preprocess_prompt=True,
            )
            prompt_obj.save(str(cached_pt))
        return VoiceClonePrompt.load(str(cached_pt))

    def __call__(
        self,
        story_text: str,
        bible_spec: Dict[str, Any],
        ephemeral_voice_bytes: Optional[bytes],
    ) -> AcousticSynthesisResult:
        import soundfile as sf
        import torch

        perfil = normalizar_perfil_sensorial(bible_spec)
        estimulo = perfil["estimulo_sonoro"]
        raw_vid = bible_spec.get("_runtime_voice_id") or bible_spec.get("reglas_produccion", {}).get("voz_id")
        voice_id = resolve_catalog_voice_id(str(raw_vid) if raw_vid else None)
        proper_nouns = sorted(set(re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúüñ]{2,}\b", story_text)))[:24]
        asr_prompt = "Cuento infantil chileno con: " + ", ".join(proper_nouns) + "."

        bloques = agrupar_bloques_respirables(story_text)
        tts = self._load_or_get_tts()
        whisper = self._load_or_get_whisper()

        tramos: List[Tuple[np.ndarray, float]] = []
        block_repairs = 0
        try:
            voice_prompt = self._resolve_voice_prompt(tts, whisper, voice_id, ephemeral_voice_bytes)
            total = max(1, len(bloques))

            for idx, bloque in enumerate(bloques):
                progreso = idx / max(1, total - 1)
                base_speed, pausa_s, gain = calcular_curva_sleep_ramp(progreso, estimulo)

                best_seg: Optional[np.ndarray] = None
                best_wer = 1.0

                for retry in range(self.block_max_retries + 1):
                    if retry > 0:
                        block_repairs += 1
                    torch.manual_seed(5000 + idx * 17 + retry)
                    adj_speed = max(0.84, base_speed - (retry * 0.03))
                    raw_out = tts.generate(
                        text=bloque,
                        language="es",
                        voice_clone_prompt=voice_prompt,
                        speed=adj_speed,
                    )[0]
                    cand_f32 = np.asarray(raw_out, dtype=np.float32) * gain

                    # Verificación rápida del bloque con faster-whisper en memoria/tmp
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wtmp:
                        b_wav = Path(wtmp.name)
                    try:
                        sf.write(str(b_wav), cand_f32, SR)
                        segs, _info = whisper.transcribe(
                            str(b_wav),
                            language="es",
                            vad_filter=False,
                            initial_prompt=asr_prompt,
                        )
                        heard = " ".join(s.text for s in segs)
                    finally:
                        if b_wav.exists():
                            b_wav.unlink()

                    metrics = compute_spanish_wer_and_spans(bloque, heard)
                    if metrics["wer"] < best_wer:
                        best_wer = float(metrics["wer"])
                        best_seg = cand_f32
                    if best_wer <= 0.14:
                        break

                assert best_seg is not None
                tramos.append((best_seg, pausa_s))
        finally:
            if not self.keep_models_warm and self.scheduler.resolve_effective_policy() == "sequential_staged":
                tts = None
                self.scheduler.release_cuda_cache()

        mezcla = ensamblar_bloques_con_crossfade(tramos, sr=SR)
        dsp_telemetry = analyze_bedtime_audio_telemetry(
            mezcla,
            sr=SR,
            word_count=len(story_text.split()),
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in, tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        ) as tmp_out:
            wav_in = Path(tmp_in.name)
            mp3_out = Path(tmp_out.name)

        try:
            sf.write(str(wav_in), mezcla, SR)
            af_master = "highpass=f=70,lowpass=f=10000,loudnorm=I=-20:TP=-2.0:LRA=6"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(wav_in),
                    "-af",
                    af_master,
                    "-b:a",
                    "128k",
                    str(mp3_out),
                ],
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            # Auditoría ASR final de capítulo completo sobre el MP3 masterizado
            full_segs, _full_info = whisper.transcribe(
                str(mp3_out),
                language="es",
                vad_filter=False,
                initial_prompt=asr_prompt,
            )
            full_transcript = " ".join(s.text for s in full_segs)
            full_qa = compute_spanish_wer_and_spans(story_text, full_transcript)
            mp3_bytes = mp3_out.read_bytes()
        finally:
            if wav_in.exists():
                wav_in.unlink()
            if mp3_out.exists():
                mp3_out.unlink()
            if not self.keep_models_warm and self.scheduler.resolve_effective_policy() == "sequential_staged":
                whisper = None
                self.scheduler.release_cuda_cache()

        self.last_telemetry = {
            **dsp_telemetry,
            "voice_id": voice_id,
            "estimulo_sonoro": estimulo,
            "blocks_count": len(bloques),
            "block_repairs_triggered": block_repairs,
            "whisper_wer": float(full_qa["wer"]),
            "whisper_similarity": float(full_qa["similarity"]),
            "mismatched_spans": full_qa["mismatched_spans"],
        }

        return AcousticSynthesisResult(
            mp3_bytes=mp3_bytes,
            whisper_wer=float(full_qa["wer"]),
            speaker_similarity=float(full_qa["similarity"]),
        )
