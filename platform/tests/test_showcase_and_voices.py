# -*- coding: utf-8 -*-
"""Suite de Pruebas Automatizadas Paso 4 — Catálogo Ético de Voces, Telemetría DSP y Vitrina Pública."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from roanso_platform.inference_adapters import (
    SR,
    analyze_bedtime_audio_telemetry,
    calcular_curva_sleep_ramp,
    ensamblar_bloques_con_crossfade,
    resolve_catalog_voice_id,
)


def test_01_legacy_and_canonical_catalog_voice_resolution() -> None:
    assert resolve_catalog_voice_id("catalog-chilena-calida-f1") == "catalog-chilena-calida-f1"
    assert resolve_catalog_voice_id("catalog-chileno-sereno-m1") == "catalog-chileno-sereno-m1"
    assert resolve_catalog_voice_id("catalogo_chileno_mama_serena") == "catalog-chilena-calida-f1"
    assert resolve_catalog_voice_id("catalogo_chileno_abuela_sur") == "catalog-chilena-calida-f1"
    assert resolve_catalog_voice_id("catalogo_chileno_abuelo_sur") == "catalog-chileno-sereno-m1"
    assert resolve_catalog_voice_id("unknown_id") == "catalog-chileno-sereno-m1"


def test_02_dsp_telemetry_proves_sleep_ramp_attenuation_vs_monovoz() -> None:
    t = np.linspace(0, 1.2, int(SR * 1.2), endpoint=False, dtype=np.float32)
    warm_tone = 0.28 * np.sin(2.0 * np.pi * 240.0 * t).astype(np.float32)

    # Construir 5 bloques con Sleep Ramp vs 5 bloques con Monovoz Estable
    ramp_blocks = []
    mono_blocks = []
    for i in range(5):
        prog = i / 4.0
        _sp_r, pause_r, gain_r = calcular_curva_sleep_ramp(prog, "dinamico_a_calmo")
        _sp_m, pause_m, gain_m = calcular_curva_sleep_ramp(prog, "monovoz_estable")
        ramp_blocks.append((warm_tone * gain_r, pause_r))
        mono_blocks.append((warm_tone * gain_m, pause_m))

    mix_ramp = ensamblar_bloques_con_crossfade(ramp_blocks, sr=SR)
    mix_mono = ensamblar_bloques_con_crossfade(mono_blocks, sr=SR)

    tel_ramp = analyze_bedtime_audio_telemetry(mix_ramp, sr=SR, word_count=40)
    tel_mono = analyze_bedtime_audio_telemetry(mix_mono, sr=SR, word_count=40)

    assert tel_ramp["sleep_ramp_attenuation_db"] >= 1.2
    assert abs(tel_mono["sleep_ramp_attenuation_db"]) < 0.6
    assert 150.0 <= tel_ramp["spectral_centroid_hz"] <= 1200.0


def test_03_verified_ethical_voice_catalog_and_showcase_endpoints(platform_env) -> None:
    _settings, _app, client, _storage_dir = platform_env

    # 1. Catálogo ético público
    v_res = client.get("/v1/showcase/voices")
    assert v_res.status_code == 200
    v_body = v_res.json()
    assert v_body["governance"]["zero_minor_voice_cloning"] is True
    assert v_body["governance"]["zero_third_party_actor_scraping"] is True
    voice_ids = {v["id"] for v in v_body["voices"]}
    assert {"catalog-chileno-sereno-m1", "catalog-chilena-calida-f1"}.issubset(voice_ids)

    # 2. Arquetipos públicos verificados con Whisper WER <= 0.12
    a_res = client.get("/v1/showcase/archetypes")
    assert a_res.status_code == 200
    archetypes = a_res.json()["archetypes"]
    assert len(archetypes) == 3
    by_id = {a["profile_id"]: a for a in archetypes}
    assert set(by_id.keys()) == {"demo_lucas", "demo_mateo", "demo_fresia"}

    for pid, item in by_id.items():
        assert item["acoustic_qa"]["whisper_wer"] <= 0.12, f"{pid} WER alto"
        assert item["acoustic_qa"]["duration_seconds"] >= 60.0
        assert item["mp3_size_bytes"] > 500_000

    # 3. Streaming MP3 completo + soporte HTTP 206 Range para PWA / HTML5 Audio
    full_audio = client.get("/v1/showcase/audio/demo_mateo")
    assert full_audio.status_code == 200
    assert full_audio.headers["content-type"] == "audio/mpeg"
    assert full_audio.headers["accept-ranges"] == "bytes"
    assert len(full_audio.content) == by_id["demo_mateo"]["mp3_size_bytes"]

    partial = client.get("/v1/showcase/audio/demo_mateo", headers={"Range": "bytes=0-4095"})
    assert partial.status_code == 206
    assert len(partial.content) == 4096
    assert partial.headers["content-range"].startswith("bytes 0-4095/")

    # 4. Bloquea path traversal o IDs fuera de arquetipos públicos
    bad = client.get("/v1/showcase/audio/familia_privada")
    assert bad.status_code == 400
