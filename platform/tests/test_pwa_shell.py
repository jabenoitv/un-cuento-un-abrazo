# -*- coding: utf-8 -*-
"""Suite de Pruebas Automatizadas Paso 5 — PWA Offline-First, Modo Niño Ámbar y Streaming Familiar."""
from __future__ import annotations

import base64

from roanso_platform.gpu_worker import (
    AcousticSynthesisResult,
    RentedGpuWorker,
    SelfHealingStoryEngine,
)


def test_01_pwa_shell_manifest_and_service_worker(platform_env) -> None:
    _settings, _app, client, _storage_dir = platform_env

    root = client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers["content-type"]
    html = root.text
    assert "/manifest.webmanifest" in html
    assert "/sw.js" in html or "app.js" in html
    assert "Cero Chatbots" in html
    assert "googleapis.com" not in html  # Cero rastreadores o fuentes externas

    mf = client.get("/manifest.webmanifest")
    assert mf.status_code == 200
    mf_json = mf.json()
    assert mf_json["display"] == "standalone"
    assert mf_json["lang"] == "es-CL"
    assert mf_json["theme_color"] == "#060402"

    sw = client.get("/sw.js")
    assert sw.status_code == 200
    assert sw.headers.get("service-worker-allowed") == "/"
    assert "CACHE_AUDIO_OFFLINE" in sw.text
    assert "206" in sw.text

    for asset in ("app.css", "app.js", "icon.svg"):
        r = client.get(f"/pwa/{asset}")
        assert r.status_code == 200

    assert client.get("/pwa/nonexistent.js").status_code == 404


def test_02_list_child_chapters_and_query_token_audio_streaming(platform_env) -> None:
    settings, _app, client, _storage_dir = platform_env

    reg = client.post(
        "/v1/auth/register",
        json={
            "family_name": "Familia Rojas",
            "full_name": "Paula Rojas",
            "email": "paula@ejemplo.cl",
            "password": "ClaveSegura2026!",
            "parental_pin": "2580",
            "plan_tier": "familiar_voces",
            "accepted_ley21719_notice": True,
        },
    )
    jwt_a = reg.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {jwt_a}"}

    ch = client.post(
        "/v1/children",
        headers=headers_a,
        json={
            "alias": "Mateo",
            "age_years": 6,
            "perfil_sensorial": {
                "lenguaje": "agil_visual",
                "estructura": "enganche_rampa_calma",
                "estimulo_sonoro": "dinamico_a_calmo",
                "foco_tematico": "misiones_hiperfoco",
            },
            "intereses_especiales": ["barcos de madera"],
            "valores_trabajar": ["compartir turnos"],
            "closing_phrase": "te queremos mucho",
        },
    ).json()

    client.post(
        "/v1/jobs",
        headers=headers_a,
        json={"child_id": ch["id"], "parent_seed": "Hoy armaron un barco de madera junto al estero."},
    )

    valid_story = (
        "Mateo miró los barcos de madera brillar junto al estero mientras la Canela caminaba suave a su lado. "
        * 18
        + "Mateo se sintió tranquilo porque todo quedó ordenado para mañana y en casa te queremos mucho."
    )

    worker = RentedGpuWorker(
        control_plane_url="http://testserver",
        worker_token=settings.worker_shared_token,
        worker_id="gpu-pwa-test",
        slm_endpoint="http://127.0.0.1:11434/api/generate",
        slm_model_tag="roanso-chileno:7b",
        slm_params_billions=7.6,
        story_engine=SelfHealingStoryEngine(lambda _p, _a: valid_story),
        audio_synthesizer=lambda _t, _b, _v: AcousticSynthesisResult(b"ID3_MOCK_MP3", 0.01, 0.98),
        http_client=client,
    )
    assert worker.drain_queue() == 1

    chapters = client.get(f"/v1/children/{ch['id']}/chapters", headers=headers_a)
    assert chapters.status_code == 200
    items = chapters.json()
    assert len(items) == 1
    media_url = items[0]["audio_url"]
    assert media_url.startswith("/v1/media/audio/")

    # Reproducción vía ?token= para <audio src="..."> en PWA móvil
    stream_ok = client.get(f"{media_url}?token={jwt_a}")
    assert stream_ok.status_code == 200
    assert stream_ok.content == b"ID3_MOCK_MP3"

    # Otra familia no puede reproducirlo ni siquiera con su propio ?token=
    reg_b = client.post(
        "/v1/auth/register",
        json={
            "family_name": "Familia Otra",
            "full_name": "Pedro Otra",
            "email": "pedro@ejemplo.cl",
            "password": "ClaveSegura2026!",
            "parental_pin": "1111",
            "plan_tier": "esencial_digital",
            "accepted_ley21719_notice": True,
        },
    )
    jwt_b = reg_b.json()["access_token"]
    stream_forbidden = client.get(f"{media_url}?token={jwt_b}")
    assert stream_forbidden.status_code == 403
