# -*- coding: utf-8 -*-
"""Suite E2E de Arquitectura Cloud + GPU Arrendada — ROANSO Platform v1.0."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from roanso_platform.api import create_app
from roanso_platform.config import PlatformSettings
from roanso_platform.gpu_worker import (
    AcousticSynthesisResult,
    RentedGpuWorker,
    SelfHealingStoryEngine,
)
from roanso_platform.models import StoryJob


def _build_valid_lucas_story(closing_phrase: str = "en casa te queremos mucho") -> str:
    """Genera un cuento válido para perfil literal_claro + ritual_anticipado (>=550 palabras, frases cortas)."""
    sentences = [
        "Buenas noches, Lucas.",
        "Esta noche escucharemos un cuento tranquilo en la estación de Quillay Alto.",
        "Primero te cuento lo que pasará hoy para que todo sea claro y sin sorpresas.",
        "Lucas revisará el tablero de madera de los trenes junto a la perrita Canela.",
        "Después conversará con su amigo Tomás para turnarse con calma en las palancas.",
        "Al final dejará el tren azul descansando tibio dentro de su cochera.",
        "En la estación de Quillay Alto ya encendieron tres faroles de luz ámbar.",
        "Lucas lleva puesta su gorra azul de encargado de horarios.",
        "Camina despacio por el andén de piedra con su linterna tibia.",
        "A su lado avanza la Canela moviendo la cola en silencio sobre las tablas.",
        "A Lucas le gusta mucho mirar el tablero grande de pino.",
        "En ese tablero cada tren tiene su hora exacta y su tarjeta de color.",
        "El tren número cuatro es el tren azul que reparte las cartas del valle.",
        "Ese tren llega siempre a las ocho con diez minutos desde los cerros suaves.",
        "Esta tarde vino de visita su vecino Tomás para conocer las vías.",
        "Tomás hablaba rápido porque sentía mucha curiosidad por las ruedas brillantes.",
        "Tomás quiso bajar dos palancas de bronce juntas para ver encenderse los focos.",
        "Lucas sintió los hombros tensos porque dos palancas juntas suenan muy fuerte.",
        "Además ese golpe doble confunde la aguja del riel principal.",
        "Lucas levantó una mano tranquila y explicó con claridad lo que sentía.",
        "Contó que ese sonido metálico fuerte le molestaba bastante en los oídos.",
        "Tomás soltó las palancas y escuchó con respeto a su amigo.",
        "Lo entendió al tiro, eso quiere decir que lo comprendió de inmediato y con cariño.",
        "Tomás explicó que estaba apurado porque deseaba ver brillar la lámpara verde.",
        "Entonces ambos buscaron un acuerdo sencillo donde los dos estuvieran cómodos.",
        "Primero Lucas revisó la vía usando su libreta cuadriculada y su lápiz grafito.",
        "Todo el riel estaba despejado y seguro bajo la brisa fresca del sur.",
        "Luego Lucas hizo una seña suave con su linterna amarilla desde el andén.",
        "Con esa señal clara, Tomás bajó despacio una sola palanca de bronce.",
        "La lámpara redonda cambió a color verde con un clic suavecito y parejo.",
        "Lucas se sintió aliviado porque el sonido fue suave y todo siguió su orden.",
        "A lo lejos apareció el faro tibio del tren azul avanzando lento entre los árboles.",
        "Las ruedas frenaron con un susurro largo frente al banco de madera.",
        "Don Anselmo saludó desde la cabina con una sonrisa serena y amable.",
        "Entregó cuatro sacos de tela llenos de cartas y dibujos del pueblo vecino.",
        "Lucas revisó cada etiqueta numerada y puso una marca limpia en su cuaderno.",
        "Tomás ayudó a ordenar los paquetes chicos dentro del canasto de mimbre.",
        "Trabajar por turnos resultó entretenido y ninguno tuvo que cambiar su forma de ser.",
        "Después el tren azul entró despacito a la cochera techada para descansar.",
        "Don Anselmo apagó la caldera tibia y cerró las puertas grandes de cedro.",
        "En el cielo despejado brillaban las primeras estrellas quietas sobre el valle.",
        "Tomás se despidió contento prometiendo volver mañana por la tarde a saludar.",
        "Lucas regresó a su casa de madera junto a la fiel perrita Canela.",
        "La Canela se acostó sobre la alfombra tejida y suspiró hondo y parejo.",
        "Lucas guardó su gorra azul en la repisa ordenada junto a la ventana.",
        "Se puso su pijama suave de algodón tibio y apagó la lámpara del velador.",
        "Afuera solo se escuchaba el murmullo lento del estero bajando entre las piedras.",
        "Lucas respiró profundo sintiendo sus hombros livianos y su cama muy calentita.",
        f"Cerró los ojos tranquilo esperando el día de mañana porque {closing_phrase}.",
    ]
    return " ".join(sentences)


@pytest.fixture()
def platform_env(tmp_path: Path):
    db_file = tmp_path / "test_roanso.db"
    storage_dir = tmp_path / "object_storage"
    master_key_b64 = base64.urlsafe_b64encode(b"0123456789abcdef0123456789abcdef").decode("ascii")
    settings = PlatformSettings(
        environment="test",
        database_url=f"sqlite:///{db_file.as_posix()}",
        jwt_secret_key="test-super-secret-jwt-key-roanso-cloud-2026",
        voice_vault_master_key_b64=master_key_b64,
        worker_shared_token="worker-secret-token-runpod-99",
        lease_ttl_seconds=60,
        max_job_attempts=3,
        storage_backend="local",
        local_storage_dir=storage_dir,
        rented_gpu_hourly_usd=0.54,
    )
    app = create_app(settings)
    client = TestClient(app)
    return settings, app, client, storage_dir


def test_01_healthz_and_human_scale_manifest(platform_env) -> None:
    _settings, _app, client, _storage_dir = platform_env
    res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["queue"]["scale_to_zero_idle"] is True
    assert body["human_scale_governance"]["depends_on_frontier_api"] is False
    assert body["human_scale_governance"]["max_params_billions"] <= 10.0


def test_02_register_pin_and_ley21719_onboarding_sanitizer(platform_env) -> None:
    _settings, app, client, _storage_dir = platform_env

    reg = client.post(
        "/v1/auth/register",
        json={
            "family_name": "Familia Valdivia",
            "full_name": "Carolina Valdivia",
            "email": "carolina@ejemplo.cl",
            "password": "PasswordSegura2026!",
            "parental_pin": "4821",
            "plan_tier": "familiar_voces",
            "accepted_ley21719_notice": True,
        },
    )
    assert reg.status_code == 201
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # PIN incorrecto bloqueado (protege panel parental frente al niño)
    bad_pin = client.post("/v1/auth/verify-pin", headers=headers, json={"parental_pin": "0000"})
    assert bad_pin.status_code == 403
    ok_pin = client.post("/v1/auth/verify-pin", headers=headers, json={"parental_pin": "4821"})
    assert ok_pin.status_code == 200
    assert ok_pin.json()["unlocked"] is True

    # Intento de ingresar diagnósticos y fármacos (TEA, TDAH, risperidona) en onboarding
    child_res = client.post(
        "/v1/children",
        headers=headers,
        json={
            "alias": "Lucas",
            "age_years": 7,
            "perfil_sensorial": {
                "lenguaje": "literal_claro",
                "estructura": "ritual_anticipado",
                "estimulo_sonoro": "monovoz_estable",
                "foco_tematico": "interes_profundo",
            },
            "intereses_especiales": ["trenes eléctricos con TEA", "mapas de estaciones"],
            "temas_evitar": ["ruidos fuertes", "psiquiatra y risperidona"],
            "notas_libres_padre": "Fue diagnosticado con autismo y TDAH el año pasado, prefiere rutinas suaves.",
            "closing_phrase": "en casa te queremos mucho",
        },
    )
    assert child_res.status_code == 201
    child_data = child_res.json()
    assert child_data["sanitized_terms_blocked_count"] >= 4

    # Inspección forense directa a SQL: CERO términos clínicos almacenados en ninguna columna
    with app.state.engine.connect() as conn:
        raw_row = str(conn.execute(text("SELECT * FROM child_profiles")).fetchall()).lower()
    for forbidden in ("autismo", "tea", "tdah", "risperidona", "psiquiatra"):
        assert forbidden not in raw_row


def test_03_aes256_gcm_voice_vault_and_cryptographic_shred(platform_env) -> None:
    _settings, _app, client, storage_dir = platform_env
    reg = client.post(
        "/v1/auth/register",
        json={
            "family_name": "Familia Soto",
            "full_name": "Andrés Soto",
            "email": "andres@ejemplo.cl",
            "password": "PasswordSegura2026!",
            "parental_pin": "1984",
            "plan_tier": "familiar_voces",
            "accepted_ley21719_notice": True,
        },
    )
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    # Prohibición absoluta de clonar voz de menor
    minor_attempt = client.post(
        "/v1/voices/parental",
        headers=headers,
        json={
            "display_name": "Voz Niño",
            "adult_full_name": "Andrés Soto",
            "spoken_consent_transcript": "Yo Andrés Soto autorizo a ROANSO a usar mi voz para cuentos.",
            "is_minor_voice_declaration_false": False,
            "embedding_b64": base64.b64encode(b"FAKE_PT_TENSOR_SECRET_VOICE_DATA_12345").decode("ascii"),
        },
    )
    assert minor_attempt.status_code == 422

    # Enrolamiento válido adulto
    secret_pt = b"PYTORCH_SPEAKER_EMBEDDING_SECRET_VECTOR_CHILE_987654321"
    enroll = client.post(
        "/v1/voices/parental",
        headers=headers,
        json={
            "display_name": "Papá Andrés Nocturno",
            "adult_full_name": "Andrés Soto",
            "spoken_consent_transcript": "Yo Andrés Soto autorizo expresamente a ROANSO SpA a procesar mi voz para mis hijos.",
            "is_minor_voice_declaration_false": True,
            "embedding_b64": base64.b64encode(secret_pt).decode("ascii"),
        },
    )
    assert enroll.status_code == 201
    voice_id = enroll.json()["id"]

    # Verificar que en disco/S3 jamás existe el vector en texto claro (cifrado AES-256-GCM real)
    enc_files = list(storage_dir.rglob("*.aes256gcm"))
    assert len(enc_files) == 1
    cipher_on_disk = enc_files[0].read_bytes()
    assert secret_pt not in cipher_on_disk

    # Revocación y destrucción criptográfica en 1 clic (Ley 21.719)
    rev = client.delete(f"/v1/voices/{voice_id}", headers=headers)
    assert rev.status_code == 200
    assert rev.json()["status"] == "REVOKED_SHREDDED"
    assert len(list(storage_dir.rglob("*.aes256gcm"))) == 0


def test_04_rented_gpu_worker_self_healing_pipeline_end_to_end(platform_env) -> None:
    settings, app, client, _storage_dir = platform_env

    reg = client.post(
        "/v1/auth/register",
        json={
            "family_name": "Familia Morales",
            "full_name": "Elena Morales",
            "email": "elena@ejemplo.cl",
            "password": "PasswordSegura2026!",
            "parental_pin": "7744",
            "plan_tier": "familiar_voces",
            "accepted_ley21719_notice": True,
        },
    )
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    secret_pt = b"ELENA_MATERNAL_WARM_VOICE_EMBEDDING_PT_BYTES_2026"
    voice_res = client.post(
        "/v1/voices/parental",
        headers=headers,
        json={
            "display_name": "Mamá Elena Serenidad",
            "adult_full_name": "Elena Morales",
            "spoken_consent_transcript": "Yo Elena Morales autorizo a ROANSO SpA a usar mi voz materna en los cuentos.",
            "is_minor_voice_declaration_false": True,
            "embedding_b64": base64.b64encode(secret_pt).decode("ascii"),
        },
    )
    voice_id = voice_res.json()["id"]

    child_res = client.post(
        "/v1/children",
        headers=headers,
        json={
            "alias": "Lucas",
            "age_years": 7,
            "perfil_sensorial": {
                "lenguaje": "literal_claro",
                "estructura": "ritual_anticipado",
                "estimulo_sonoro": "monovoz_estable",
                "foco_tematico": "interes_profundo",
            },
            "intereses_especiales": ["trenes azules", "tableros de horarios"],
            "valores_trabajar": ["pedir turnos con calma"],
            "closing_phrase": "en casa te queremos mucho",
            "preferred_voice_id": voice_id,
        },
    )
    child_id = child_res.json()["id"]

    job_res = client.post(
        "/v1/jobs",
        headers=headers,
        json={
            "child_id": child_id,
            "parent_seed": "Hoy en el patio su primo quería apretar todos los botones juntos y acordaron turnarse.",
            "bridge_place": "Estación Quillay Alto",
            "bridge_character": "Don Anselmo",
        },
    )
    assert job_res.status_code == 202
    job_id = job_res.json()["id"]

    # Simular Nodo GPU Arrendado (RunPod NVIDIA L4) conectado por HTTP al Control Plane:
    # Turno 0 produce un sermón prohibido ("aprendió una valiosa lección") -> SelfHealingStoryEngine
    # detecta el error determinista y en Turno 1 entrega el capítulo limpio.
    clean_story = _build_valid_lucas_story("en casa te queremos mucho")
    prompts_seen: list[str] = []
    decrypted_voice_received: list[bytes] = []

    def mock_slm_7b(prompt: str, attempt: int) -> str:
        prompts_seen.append(prompt)
        if attempt == 0:
            return clean_story + " Y así Lucas aprendió una valiosa lección para siempre."
        return clean_story

    def mock_omnivoice_and_whisper(
        text_in: str,
        bible_spec: Dict[str, Any],
        ephemeral_voice: Optional[bytes],
    ) -> AcousticSynthesisResult:
        assert ephemeral_voice == secret_pt
        decrypted_voice_received.append(ephemeral_voice or b"")
        assert bible_spec["perfil_sensorial"]["estimulo_sonoro"] == "monovoz_estable"
        return AcousticSynthesisResult(
            mp3_bytes=b"ID3_FAKE_MP3_STREAM_WITH_MINUS_52DBFS_ROOM_TONE_" + text_in[:40].encode("utf-8"),
            whisper_wer=0.038,
            speaker_similarity=0.91,
        )

    worker = RentedGpuWorker(
        control_plane_url="http://testserver",
        worker_token=settings.worker_shared_token,
        worker_id="runpod-l4-worker-01",
        gpu_model="NVIDIA-L4-24GB",
        slm_endpoint="http://127.0.0.1:8000/v1",
        slm_model_tag="Qwen2.5-7B-Instruct-ROANSO-LoRA",
        slm_params_billions=7.6,
        story_engine=SelfHealingStoryEngine(mock_slm_7b, max_repair_turns=3),
        audio_synthesizer=mock_omnivoice_and_whisper,
        http_client=client,
    )

    processed = worker.drain_queue()
    assert processed == 1
    assert len(prompts_seen) == 2
    assert "CORRECCIÓN OBLIGATORIA" in prompts_seen[1]
    assert len(decrypted_voice_received) == 1

    # Cola ahora está vacía -> Scale-to-Zero activado
    assert client.get("/healthz").json()["queue"]["scale_to_zero_idle"] is True

    # Familia consulta el trabajo y descarga el capítulo verificado + guía de desayuno
    job_done = client.get(f"/v1/jobs/{job_id}", headers=headers).json()
    assert job_done["status"] == "READY"
    assert job_done["slm_params_billions"] == 7.6
    assert job_done["gpu_seconds_used"] > 0

    chap = client.get(f"/v1/children/{child_id}/chapters/1", headers=headers)
    assert chap.status_code == 200
    chap_body = chap.json()
    assert chap_body["readability"]["szigriszt_pazos"] >= 65.0
    assert chap_body["acoustic_qa"]["whisper_wer"] == 0.038
    assert len(chap_body["parent_guide"]["preguntas"]) == 3
    assert chap_body["qa_report"]["editorial_review"]["self_healing_turns"] == 2

    # Descarga autenticada del stream MP3
    audio_dl = client.get(chap_body["audio_url"], headers=headers)
    assert audio_dl.status_code == 200
    assert audio_dl.content.startswith(b"ID3_FAKE_MP3_STREAM")


def test_05_spot_gpu_crash_orphan_lease_recovery(platform_env) -> None:
    settings, app, client, _storage_dir = platform_env
    reg = client.post(
        "/v1/auth/register",
        json={
            "family_name": "Familia Díaz",
            "full_name": "Tomás Díaz",
            "email": "tomas@ejemplo.cl",
            "password": "PasswordSegura2026!",
            "parental_pin": "1234",
            "accepted_ley21719_notice": True,
        },
    )
    headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    child_id = client.post(
        "/v1/children",
        headers=headers,
        json={
            "alias": "Lucas",
            "age_years": 7,
            "perfil_sensorial": {
                "lenguaje": "literal_claro",
                "estructura": "ritual_anticipado",
                "estimulo_sonoro": "monovoz_estable",
                "foco_tematico": "interes_profundo",
            },
        },
    ).json()["id"]
    job_id = client.post(
        "/v1/jobs",
        headers=headers,
        json={"child_id": child_id, "parent_seed": "Hoy ordenamos los vagones azules antes de cenar."},
    ).json()["id"]

    # Nodo Spot #1 reclama el lease y luego "muere" (forzamos vencimiento de lease_expires_at)
    worker_headers = {"X-Worker-Token": settings.worker_shared_token}
    claim1 = client.post(
        "/v1/worker/leases/claim",
        headers=worker_headers,
        json={"worker_id": "spot-dead-node-01", "gpu_model": "RTX-4090"},
    )
    assert claim1.status_code == 200

    with app.state.session_factory() as db:
        stalled = db.get(StoryJob, job_id)
        assert stalled is not None
        stalled.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        db.commit()

    # Nodo Spot #2 entra, el Control Plane recupera el lease huérfano automáticamente y lo entrega a #2
    claim2 = client.post(
        "/v1/worker/leases/claim",
        headers=worker_headers,
        json={"worker_id": "spot-healthy-node-02", "gpu_model": "NVIDIA-L4-24GB"},
    )
    assert claim2.status_code == 200
    assert claim2.json()["job_id"] == job_id


def test_06_blocks_frontier_api_and_high_wer_submissions(platform_env) -> None:
    settings, _app, client, _storage_dir = platform_env
    worker_headers = {"X-Worker-Token": settings.worker_shared_token}

    # Bloquea nodo que intente usar OpenAI API
    bad_frontier = client.post(
        "/v1/worker/leases/claim",
        headers=worker_headers,
        json={
            "worker_id": "rogue-worker",
            "inference_endpoint": "https://api.openai.com/v1",
            "slm_params_billions": 7.6,
        },
    )
    assert bad_frontier.status_code == 422

    # Bloquea modelo gigante > 10B
    bad_size = client.post(
        "/v1/worker/leases/claim",
        headers=worker_headers,
        json={
            "worker_id": "rogue-worker-70b",
            "inference_endpoint": "http://127.0.0.1:8000/v1",
            "slm_params_billions": 70.0,
        },
    )
    assert bad_size.status_code == 422


def test_07_multi_tenant_isolation_and_arco_erasure(platform_env) -> None:
    _settings, _app, client, _storage_dir = platform_env
    fam_a = client.post(
        "/v1/auth/register",
        json={
            "family_name": "Familia A",
            "full_name": "Mamá A",
            "email": "a@ejemplo.cl",
            "password": "PasswordSegura2026!",
            "parental_pin": "1111",
            "accepted_ley21719_notice": True,
        },
    ).json()
    fam_b = client.post(
        "/v1/auth/register",
        json={
            "family_name": "Familia B",
            "full_name": "Papá B",
            "email": "b@ejemplo.cl",
            "password": "PasswordSegura2026!",
            "parental_pin": "2222",
            "accepted_ley21719_notice": True,
        },
    ).json()

    headers_a = {"Authorization": f"Bearer {fam_a['access_token']}"}
    headers_b = {"Authorization": f"Bearer {fam_b['access_token']}"}

    child_a = client.post(
        "/v1/children",
        headers=headers_a,
        json={"alias": "Sofi", "age_years": 6},
    ).json()

    # Familia B no ve los perfiles de Familia A ni puede encolarle capítulos
    assert client.get("/v1/children", headers=headers_b).json() == []
    cross_job = client.post(
        "/v1/jobs",
        headers=headers_b,
        json={"child_id": child_a["id"], "parent_seed": "Intento cruzado entre tenants."},
    )
    assert cross_job.status_code == 404

    # Portabilidad ARCO+ y Derecho al Olvido completo de Familia A
    exp = client.get("/v1/families/me/export", headers=headers_a)
    assert exp.status_code == 200
    assert exp.json()["clinical_data_stored"] is False

    erased = client.delete("/v1/families/me", headers=headers_a)
    assert erased.status_code == 200
    assert client.get("/v1/children", headers=headers_a).status_code == 401
