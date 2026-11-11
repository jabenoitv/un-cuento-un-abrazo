# -*- coding: utf-8 -*-
"""Tests Unitarios de Adaptadores de Inferencia GPU, Planificador VRAM y DSP Nocturno (Paso 2)."""
from __future__ import annotations

import numpy as np

from roanso_platform.inference_adapters import (
    SR,
    SlmNarrativeAndEditorialAdapter,
    VramStageScheduler,
    agrupar_bloques_respirables,
    calcular_curva_sleep_ramp,
    compute_spanish_wer_and_spans,
    ensamblar_bloques_con_crossfade,
    normalize_spanish_asr_tokens,
)


def test_01_spanish_asr_normalization_preserves_enie_and_digits() -> None:
    tokens = normalize_spanish_asr_tokens("¡Mañana el niño verá 4 trenes en Estación Quillay!")
    assert tokens == ["mañana", "el", "niño", "vera", "cuatro", "trenes", "en", "estacion", "quillay"]


def test_02_spanish_wer_zero_on_punctuation_and_catches_swallowed_spans() -> None:
    ref = "Buenas noches, Lucas. El tren número 4 descansó tranquilo en su cochera tibia."
    hyp_ok = "buenas noches lucas el tren numero cuatro descanso tranquilo en su cochera tibia"
    res_ok = compute_spanish_wer_and_spans(ref, hyp_ok)
    assert res_ok["wer"] == 0.0
    assert res_ok["mismatched_spans"] == []

    # Simulamos 5 palabras comidas por el TTS en medio del párrafo
    hyp_bad = "buenas noches lucas en su cochera tibia"
    res_bad = compute_spanish_wer_and_spans(ref, hyp_bad)
    assert res_bad["wer"] > 0.35
    assert len(res_bad["mismatched_spans"]) == 1
    assert "tren numero cuatro" in res_bad["mismatched_spans"][0]["expected"]


def test_03_vram_scheduler_24gb_resident_vs_8gb_sequential(monkeypatch) -> None:
    sched_auto = VramStageScheduler(policy="auto")

    # Simular nodo GPU arrendado NVIDIA L4 24 GB
    monkeypatch.setattr(sched_auto, "get_cuda_memory_mib", lambda: (21500, 24576))
    assert sched_auto.resolve_effective_policy() == "resident_24gb"
    assert sched_auto.select_tts_device() == "cuda:0"
    assert sched_auto.select_whisper_device_and_compute() == ("cuda", "float16")
    # En 24 GB no descarga el SLM entre trabajos
    assert sched_auto.evict_slm_if_staged("http://127.0.0.1:11434/api/generate", "qwen2.5:7b") is False

    # Simular banco local RTX 5060 8 GB
    monkeypatch.setattr(sched_auto, "get_cuda_memory_mib", lambda: (6800, 8151))
    assert sched_auto.resolve_effective_policy() == "sequential_staged"
    assert sched_auto.select_tts_device() == "cuda:0"
    assert sched_auto.select_whisper_device_and_compute() == ("cpu", "int8")


def test_04_sleep_ramp_vs_monovoz_estable_and_crossfade_dsp() -> None:
    # Monovoz estable (TEA / alta sensibilidad): velocidad y ganancia constantes
    assert calcular_curva_sleep_ramp(0.0, "monovoz_estable") == (0.90, 0.82, 1.0)
    assert calcular_curva_sleep_ramp(0.95, "monovoz_estable") == (0.90, 0.82, 1.0)

    # Sleep Ramp (dinamico_a_calmo): desacelera y suaviza ganancia hacia el final
    s_ini, p_ini, g_ini = calcular_curva_sleep_ramp(0.10, "dinamico_a_calmo")
    s_fin, p_fin, g_fin = calcular_curva_sleep_ramp(0.96, "dinamico_a_calmo")
    assert s_fin < s_ini
    assert p_fin > p_ini
    assert g_fin < g_ini

    bloques = agrupar_bloques_respirables(
        "Primera oración corta. Segunda oración corta. Tercera oración un poco más larga para probar el corte limpio.",
        max_chars=60,
    )
    assert len(bloques) >= 2

    t = np.linspace(0, 0.5, int(SR * 0.5), endpoint=False, dtype=np.float32)
    onda1 = 0.25 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    onda2 = 0.20 * np.sin(2 * np.pi * 196 * t).astype(np.float32)
    mezcla = ensamblar_bloques_con_crossfade([(onda1, 0.6), (onda2, 0.6)], sr=SR)
    assert mezcla.dtype == np.float32
    assert float(np.max(np.abs(mezcla))) <= 0.98
    # En el tramo de pausa intermedia no hay silencio digital cero absoluto gracias al room tone -52 dBFS
    mitad_pausa = mezcla[int(SR * 0.65) : int(SR * 0.95)]
    assert float(np.sqrt(np.mean(mitad_pausa ** 2))) > 1e-4


def test_05_slm_adapter_cleans_markdown_and_blocks_frontier() -> None:
    adapter = SlmNarrativeAndEditorialAdapter(
        endpoint_url="http://127.0.0.1:11434/api/generate",
        model_tag="qwen2.5:7b",
        param_billions=7.6,
    )
    adapter._call_slm = lambda prompt, temperature=0.6, json_mode=False: "# Título\nPrimera frase.\n\nSegunda frase."  # type: ignore[method-assign]
    gen_fn = adapter.build_generator_fn()
    assert gen_fn("hola", 0) == "Primera frase. Segunda frase."
