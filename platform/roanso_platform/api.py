# -*- coding: utf-8 -*-
"""Control Plane API (FastAPI) para Servidor VPS Arrendado + Nodos GPU Elásticos."""

import base64
import hmac
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, Generator, List, Optional

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import PlatformSettings, get_settings_from_env
from .database import build_engine, init_db_and_seed_catalog
from .domain_bridge import STACK_MODELOS_HUMAN_SCALE
from .models import (
    ChildProfile,
    ConsentAuditLog,
    FamilyTenant,
    ParentUser,
    StoryChapter,
    StoryJob,
    VoiceProfile,
)
from .queue_service import DistributedJobQueueService
from .schemas import (
    AuthTokenResponse,
    ChildProfileCreate,
    ChildProfileResponse,
    FamilyRegisterRequest,
    LoginRequest,
    ParentalVoiceEnrollRequest,
    SensoryAxesDTO,
    StoryJobEnqueueRequest,
    StoryJobResponse,
    VerifyPinRequest,
    VoiceProfileResponse,
    WorkerClaimRequest,
    WorkerCompleteJobRequest,
    WorkerFailJobRequest,
    WorkerHeartbeatRequest,
    WorkerLeasePayload,
    WorkerUploadAudioRequest,
)
from .security import (
    VoiceVaultCrypto,
    create_access_token,
    decode_access_token,
    hash_secret_pbkdf2,
    sha256_hex,
    verify_secret_pbkdf2,
)
from .storage import build_storage_adapter

bearer_scheme = HTTPBearer(auto_error=False)


def create_app(settings: Optional[PlatformSettings] = None) -> FastAPI:
    cfg = settings or get_settings_from_env()
    engine = build_engine(cfg.database_url)
    session_factory = init_db_and_seed_catalog(engine)
    storage = build_storage_adapter(cfg)
    vault = VoiceVaultCrypto(cfg.master_key_bytes)
    queue_svc = DistributedJobQueueService(settings=cfg, storage=storage, vault=vault)

    app = FastAPI(
        title="ROANSO SpA — Un Cuento, Un Abrazo (Cloud Control Plane)",
        version="1.0.0",
        description=(
            "API multi-tenant desacoplada para correr en VPS liviano con PostgreSQL + S3/R2 "
            "y despachar síntesis narrativa/vocal a nodos GPU arrendados bajo demanda."
        ),
    )
    app.state.settings = cfg
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.storage = storage
    app.state.vault = vault
    app.state.queue_svc = queue_svc

    def get_db() -> Generator[Session, None, None]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    def get_current_parent(
        creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)],
        db: Annotated[Session, Depends(get_db)],
    ) -> ParentUser:
        if not creds or creds.scheme.lower() != "bearer":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token Bearer requerido.")
        try:
            claims = decode_access_token(creds.credentials, cfg.jwt_secret_key)
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"JWT inválido: {exc}") from exc

        user = db.scalar(
            select(ParentUser).where(
                ParentUser.id == claims.get("sub"),
                ParentUser.family_id == claims.get("fam"),
            )
        )
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Cuenta familiar no activa.")
        return user

    def require_worker_token(
        x_worker_token: Annotated[Optional[str], Header()] = None,
    ) -> None:
        if not x_worker_token or not hmac.compare_digest(x_worker_token, cfg.worker_shared_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-Worker-Token inválido.")

    # ------------------------------------------------------------------
    # Salud, Observabilidad y Manifiesto Human-Scale AI
    # ------------------------------------------------------------------
    @app.get("/healthz", tags=["observability"])
    def healthz(db: Annotated[Session, Depends(get_db)]) -> Dict[str, Any]:
        queued_count = db.scalar(select(func.count(StoryJob.id)).where(StoryJob.status == "QUEUED")) or 0
        leased_count = db.scalar(select(func.count(StoryJob.id)).where(StoryJob.status.like("LEASED_%"))) or 0
        return {
            "status": "ok",
            "service": "roanso-control-plane",
            "environment": cfg.environment,
            "database_dialect": engine.dialect.name,
            "storage_backend": cfg.storage_backend,
            "queue": {
                "queued_jobs": int(queued_count),
                "active_gpu_leases": int(leased_count),
                "scale_to_zero_idle": (queued_count == 0 and leased_count == 0),
            },
            "human_scale_governance": {
                "enforced": cfg.enforce_human_scale_ceiling,
                "max_params_billions": cfg.max_model_params_billions,
                "depends_on_frontier_api": False,
                "models": STACK_MODELOS_HUMAN_SCALE,
            },
        }

    # ------------------------------------------------------------------
    # Autenticación Familiar + PIN Parental
    # ------------------------------------------------------------------
    @app.post("/v1/auth/register", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED, tags=["auth"])
    def register_family(
        payload: FamilyRegisterRequest,
        db: Annotated[Session, Depends(get_db)],
    ) -> AuthTokenResponse:
        existing = db.scalar(select(ParentUser).where(ParentUser.email == payload.email.lower().strip()))
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="El correo ya está registrado.")

        tenant = FamilyTenant(
            family_name=payload.family_name.strip(),
            plan_tier=payload.plan_tier,
        )
        db.add(tenant)
        db.flush()

        parent = ParentUser(
            family_id=tenant.id,
            email=payload.email.lower().strip(),
            full_name=payload.full_name.strip(),
            password_hash=hash_secret_pbkdf2(payload.password),
            parental_pin_hash=hash_secret_pbkdf2(payload.parental_pin),
        )
        db.add(parent)
        db.flush()

        consent_text = (
            f"Acepto el tratamiento mínimo no clínico bajo Ley 21.719 para la familia {tenant.family_name} "
            f"({parent.email})."
        )
        db.add(
            ConsentAuditLog(
                family_id=tenant.id,
                parent_user_id=parent.id,
                consent_type="ley21719_onboarding",
                statement_sha256=sha256_hex(consent_text),
                metadata_json={"plan_tier": tenant.plan_tier},
            )
        )
        db.commit()

        token = create_access_token(
            subject_user_id=parent.id,
            family_id=tenant.id,
            secret_key=cfg.jwt_secret_key,
            expire_minutes=cfg.jwt_expire_minutes,
        )
        return AuthTokenResponse(
            access_token=token,
            family_id=tenant.id,
            parent_user_id=parent.id,
            plan_tier=tenant.plan_tier,
        )

    @app.post("/v1/auth/login", response_model=AuthTokenResponse, tags=["auth"])
    def login_family(
        payload: LoginRequest,
        db: Annotated[Session, Depends(get_db)],
    ) -> AuthTokenResponse:
        parent = db.scalar(select(ParentUser).where(ParentUser.email == payload.email.lower().strip()))
        if not parent or not verify_secret_pbkdf2(payload.password, parent.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciales inválidas.")
        tenant = db.scalar(select(FamilyTenant).where(FamilyTenant.id == parent.family_id))
        token = create_access_token(
            subject_user_id=parent.id,
            family_id=parent.family_id,
            secret_key=cfg.jwt_secret_key,
            expire_minutes=cfg.jwt_expire_minutes,
        )
        return AuthTokenResponse(
            access_token=token,
            family_id=parent.family_id,
            parent_user_id=parent.id,
            plan_tier=tenant.plan_tier if tenant else "esencial_digital",
        )

    @app.post("/v1/auth/verify-pin", tags=["auth"])
    def verify_parental_pin(
        payload: VerifyPinRequest,
        parent: Annotated[ParentUser, Depends(get_current_parent)],
    ) -> Dict[str, Any]:
        if not verify_secret_pbkdf2(payload.parental_pin, parent.parental_pin_hash):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="PIN parental incorrecto.")
        elevated = create_access_token(
            subject_user_id=parent.id,
            family_id=parent.family_id,
            secret_key=cfg.jwt_secret_key,
            expire_minutes=30,
            scope="parental_unlocked",
        )
        return {"unlocked": True, "elevated_token": elevated, "expires_in_minutes": 30}

    # ------------------------------------------------------------------
    # Perfiles Infantiles (Sanitización Automática Ley 21.719)
    # ------------------------------------------------------------------
    def _to_child_response(child: ChildProfile) -> ChildProfileResponse:
        return ChildProfileResponse(
            id=child.id,
            alias=child.alias,
            age_years=child.age_years,
            perfil_sensorial=SensoryAxesDTO(
                lenguaje=child.axis_lenguaje,
                estructura=child.axis_estructura,
                estimulo_sonoro=child.axis_estimulo_sonoro,
                foco_tematico=child.axis_foco_tematico,
            ),
            intereses_especiales=child.special_interests_json,
            valores_trabajar=child.valores_json,
            temas_evitar=child.temas_evitar_json,
            closing_phrase=child.closing_phrase,
            preferred_voice_id=child.preferred_voice_id,
            sanitized_terms_blocked_count=child.sanitized_terms_blocked_count,
            created_at=child.created_at,
        )

    @app.post(
        "/v1/children",
        response_model=ChildProfileResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["children"],
    )
    def create_child_profile(
        payload: ChildProfileCreate,
        parent: Annotated[ParentUser, Depends(get_current_parent)],
        db: Annotated[Session, Depends(get_db)],
    ) -> ChildProfileResponse:
        child = ChildProfile(
            family_id=parent.family_id,
            alias=payload.alias,
            age_years=payload.age_years,
            axis_lenguaje=payload.perfil_sensorial.lenguaje,
            axis_estructura=payload.perfil_sensorial.estructura,
            axis_estimulo_sonoro=payload.perfil_sensorial.estimulo_sonoro,
            axis_foco_tematico=payload.perfil_sensorial.foco_tematico,
            special_interests_json=payload.intereses_especiales,
            valores_json=payload.valores_trabajar,
            temas_evitar_json=payload.temas_evitar,
            closing_phrase=payload.closing_phrase,
            preferred_voice_id=payload.preferred_voice_id,
            bible_json={
                "mascota": {"nombre": payload.mascota_nombre, "especie": payload.mascota_especie},
                "notas_contexto_no_clinicas": payload.notas_libres_padre,
            },
            sanitized_terms_blocked_count=payload.sanitized_terms_blocked_count,
        )
        db.add(child)
        db.commit()
        db.refresh(child)
        return _to_child_response(child)

    @app.get("/v1/children", response_model=List[ChildProfileResponse], tags=["children"])
    def list_children(
        parent: Annotated[ParentUser, Depends(get_current_parent)],
        db: Annotated[Session, Depends(get_db)],
    ) -> List[ChildProfileResponse]:
        rows = db.scalars(
            select(ChildProfile)
            .where(ChildProfile.family_id == parent.family_id)
            .order_by(ChildProfile.created_at.asc())
        ).all()
        return [_to_child_response(c) for c in rows]

    # ------------------------------------------------------------------
    # Bóveda Biométrica AES-256-GCM y Revocación Criptográfica en 1 Clic
    # ------------------------------------------------------------------
    @app.get("/v1/voices", response_model=List[VoiceProfileResponse], tags=["voices"])
    def list_available_voices(
        parent: Annotated[ParentUser, Depends(get_current_parent)],
        db: Annotated[Session, Depends(get_db)],
    ) -> List[VoiceProfileResponse]:
        voices = db.scalars(
            select(VoiceProfile).where(
                (VoiceProfile.is_catalog.is_(True)) | (VoiceProfile.family_id == parent.family_id)
            )
        ).all()
        return [
            VoiceProfileResponse(
                id=v.id,
                display_name=v.display_name,
                is_catalog=v.is_catalog,
                locale=v.locale,
                status=v.status,
                created_at=v.created_at,
                revoked_at=v.revoked_at,
            )
            for v in voices
        ]

    @app.post(
        "/v1/voices/parental",
        response_model=VoiceProfileResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["voices"],
    )
    def enroll_parental_voice(
        payload: ParentalVoiceEnrollRequest,
        parent: Annotated[ParentUser, Depends(get_current_parent)],
        db: Annotated[Session, Depends(get_db)],
    ) -> VoiceProfileResponse:
        try:
            raw_pt = base64.b64decode(payload.embedding_b64.encode("ascii"), validate=True)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Base64 inválido: {exc}") from exc

        encrypted = vault.encrypt_embedding(plaintext_bytes=raw_pt, family_id=parent.family_id)

        consent_log = ConsentAuditLog(
            family_id=parent.family_id,
            parent_user_id=parent.id,
            consent_type="biometric_voice_clone",
            statement_sha256=sha256_hex(payload.spoken_consent_transcript),
            metadata_json={
                "adult_full_name": payload.adult_full_name,
                "embedding_plaintext_sha256": encrypted.plaintext_sha256,
            },
        )
        db.add(consent_log)
        db.flush()

        voice = VoiceProfile(
            family_id=parent.family_id,
            display_name=payload.display_name,
            is_catalog=False,
            locale="es-CL",
            key_salt_hex=encrypted.key_salt_hex,
            plaintext_sha256=encrypted.plaintext_sha256,
            consent_log_id=consent_log.id,
            status="ACTIVE",
        )
        db.add(voice)
        db.flush()

        storage_key = f"vault/{parent.family_id}/{voice.id}.aes256gcm"
        storage.put_bytes(storage_key, encrypted.ciphertext_blob, content_type="application/octet-stream")
        voice.storage_key = storage_key
        db.commit()
        db.refresh(voice)

        return VoiceProfileResponse(
            id=voice.id,
            display_name=voice.display_name,
            is_catalog=voice.is_catalog,
            locale=voice.locale,
            status=voice.status,
            created_at=voice.created_at,
            revoked_at=voice.revoked_at,
        )

    @app.delete("/v1/voices/{voice_id}", tags=["voices"])
    def revoke_and_shred_parental_voice(
        voice_id: str,
        parent: Annotated[ParentUser, Depends(get_current_parent)],
        db: Annotated[Session, Depends(get_db)],
    ) -> Dict[str, Any]:
        voice = db.scalar(
            select(VoiceProfile).where(
                VoiceProfile.id == voice_id,
                VoiceProfile.family_id == parent.family_id,
                VoiceProfile.is_catalog.is_(False),
            )
        )
        if not voice:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voz parental no encontrada.")

        if voice.storage_key:
            storage.delete_key(voice.storage_key)

        # Destrucción criptográfica del salt HKDF + limpieza de referencia
        voice.key_salt_hex = ""
        voice.storage_key = ""
        voice.status = "REVOKED_SHREDDED"
        voice.revoked_at = datetime.now(timezone.utc)

        db.add(
            ConsentAuditLog(
                family_id=parent.family_id,
                parent_user_id=parent.id,
                consent_type="voice_revocation",
                statement_sha256=sha256_hex(f"revoke:{voice.id}"),
                metadata_json={"voice_id": voice.id, "shredded": True},
            )
        )
        db.commit()
        return {"revoked": True, "voice_id": voice.id, "status": voice.status}

    # ------------------------------------------------------------------
    # Encolado Familiar y Lectura de Capítulos Verificados
    # ------------------------------------------------------------------
    def _to_job_response(job: StoryJob) -> StoryJobResponse:
        return StoryJobResponse(
            id=job.id,
            child_id=job.child_id,
            chapter_number=job.chapter_number,
            status=job.status,
            stage_detail=job.stage_detail,
            attempts=job.attempts,
            leased_by_worker_id=job.leased_by_worker_id,
            slm_model_tag=job.slm_model_tag,
            slm_params_billions=job.slm_params_billions,
            gpu_seconds_used=job.gpu_seconds_used,
            estimated_cost_usd=job.estimated_cost_usd,
            last_error=job.last_error,
            created_at=job.created_at,
            completed_at=job.completed_at,
        )

    @app.post(
        "/v1/jobs",
        response_model=StoryJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["stories"],
    )
    def enqueue_story(
        payload: StoryJobEnqueueRequest,
        parent: Annotated[ParentUser, Depends(get_current_parent)],
        db: Annotated[Session, Depends(get_db)],
    ) -> StoryJobResponse:
        child = db.scalar(
            select(ChildProfile).where(
                ChildProfile.id == payload.child_id,
                ChildProfile.family_id == parent.family_id,
            )
        )
        if not child:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Perfil infantil no encontrado.")
        try:
            job = queue_svc.enqueue_story_job(
                db=db,
                family_id=parent.family_id,
                child=child,
                parent_seed=payload.parent_seed,
                bridge_place=payload.bridge_place,
                bridge_character=payload.bridge_character,
                requested_voice_id=payload.voice_id,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        return _to_job_response(job)

    @app.get("/v1/jobs/{job_id}", response_model=StoryJobResponse, tags=["stories"])
    def get_job_status(
        job_id: str,
        parent: Annotated[ParentUser, Depends(get_current_parent)],
        db: Annotated[Session, Depends(get_db)],
    ) -> StoryJobResponse:
        job = db.scalar(
            select(StoryJob).where(StoryJob.id == job_id, StoryJob.family_id == parent.family_id)
        )
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trabajo no encontrado.")
        return _to_job_response(job)

    def _serialize_chapter(child: ChildProfile, chapter: StoryChapter) -> Dict[str, Any]:
        return {
            "chapter_id": chapter.id,
            "child_id": child.id,
            "child_alias": child.alias,
            "chapter_number": chapter.chapter_number,
            "title": chapter.title,
            "story_text": chapter.story_text,
            "word_count": chapter.word_count,
            "readability": {
                "szigriszt_pazos": chapter.szigriszt_score,
                "fernandez_huerta": chapter.fernandez_huerta_score,
            },
            "acoustic_qa": {
                "whisper_wer": chapter.whisper_wer,
                "speaker_similarity": chapter.speaker_similarity,
            },
            "audio_url": storage.get_read_url(
                chapter.audio_storage_key,
                expires_in=cfg.presigned_url_ttl_seconds,
            ),
            "parent_guide": chapter.parent_guide_json,
            "qa_report": chapter.qa_report_json,
            "created_at": chapter.created_at.isoformat(),
        }

    @app.get("/v1/children/{child_id}/chapters", tags=["stories"])
    def list_verified_chapters(
        child_id: str,
        parent: Annotated[ParentUser, Depends(get_current_parent)],
        db: Annotated[Session, Depends(get_db)],
    ) -> List[Dict[str, Any]]:
        child = db.scalar(
            select(ChildProfile).where(
                ChildProfile.id == child_id,
                ChildProfile.family_id == parent.family_id,
            )
        )
        if not child:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Perfil infantil no encontrado.")
        chapters = db.scalars(
            select(StoryChapter)
            .where(StoryChapter.child_id == child.id)
            .order_by(StoryChapter.chapter_number.desc())
        ).all()
        return [_serialize_chapter(child, ch) for ch in chapters]

    @app.get("/v1/children/{child_id}/chapters/{chapter_number}", tags=["stories"])
    def get_verified_chapter(
        child_id: str,
        chapter_number: int,
        parent: Annotated[ParentUser, Depends(get_current_parent)],
        db: Annotated[Session, Depends(get_db)],
    ) -> Dict[str, Any]:
        child = db.scalar(
            select(ChildProfile).where(
                ChildProfile.id == child_id,
                ChildProfile.family_id == parent.family_id,
            )
        )
        if not child:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Perfil infantil no encontrado.")
        chapter = db.scalar(
            select(StoryChapter).where(
                StoryChapter.child_id == child.id,
                StoryChapter.chapter_number == chapter_number,
            )
        )
        if not chapter:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Capítulo aún no disponible.")
        return _serialize_chapter(child, chapter)

    @app.get("/v1/media/{storage_key:path}", tags=["media"])
    def download_local_media(
        storage_key: str,
        creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)],
        db: Annotated[Session, Depends(get_db)],
        token: Optional[str] = None,
    ) -> Response:
        raw_jwt = creds.credentials if (creds and creds.scheme.lower() == "bearer") else token
        if not raw_jwt:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token Bearer requerido.")
        try:
            claims = decode_access_token(raw_jwt, cfg.jwt_secret_key)
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"JWT inválido: {exc}") from exc
        parent = db.scalar(
            select(ParentUser).where(
                ParentUser.id == claims.get("sub"),
                ParentUser.family_id == claims.get("fam"),
            )
        )
        if not parent:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Cuenta familiar no activa.")
        if not storage_key.startswith(f"audio/{parent.family_id}/"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Acceso denegado al objeto.")
        try:
            raw = storage.get_bytes(storage_key)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return Response(content=raw, media_type="audio/mpeg", headers={"Accept-Ranges": "bytes"})

    # ------------------------------------------------------------------
    # Portabilidad y Derecho al Olvido Total (ARCO+ Ley 21.719)
    # ------------------------------------------------------------------
    @app.get("/v1/families/me/export", tags=["privacy-ley21719"])
    def export_family_data(
        parent: Annotated[ParentUser, Depends(get_current_parent)],
        db: Annotated[Session, Depends(get_db)],
    ) -> Dict[str, Any]:
        tenant = db.scalar(select(FamilyTenant).where(FamilyTenant.id == parent.family_id))
        children = db.scalars(select(ChildProfile).where(ChildProfile.family_id == parent.family_id)).all()
        consents = db.scalars(select(ConsentAuditLog).where(ConsentAuditLog.family_id == parent.family_id)).all()
        return {
            "law_compliance": "Chile Ley 21.719 — Portabilidad ARCO+",
            "clinical_data_stored": False,
            "family": {
                "id": tenant.id if tenant else parent.family_id,
                "family_name": tenant.family_name if tenant else "",
                "plan_tier": tenant.plan_tier if tenant else "",
            },
            "children": [_to_child_response(c).model_dump(mode="json") for c in children],
            "consent_ledger": [
                {
                    "id": log.id,
                    "consent_type": log.consent_type,
                    "statement_sha256": log.statement_sha256,
                    "created_at": log.created_at.isoformat(),
                }
                for log in consents
            ],
        }

    @app.delete("/v1/families/me", tags=["privacy-ley21719"])
    def erase_entire_family_tenant(
        parent: Annotated[ParentUser, Depends(get_current_parent)],
        db: Annotated[Session, Depends(get_db)],
    ) -> Dict[str, Any]:
        voices = db.scalars(
            select(VoiceProfile).where(
                VoiceProfile.family_id == parent.family_id,
                VoiceProfile.is_catalog.is_(False),
            )
        ).all()
        for v in voices:
            if v.storage_key:
                storage.delete_key(v.storage_key)

        chapters = db.scalars(
            select(StoryChapter)
            .join(ChildProfile, StoryChapter.child_id == ChildProfile.id)
            .where(ChildProfile.family_id == parent.family_id)
        ).all()
        for ch in chapters:
            storage.delete_key(ch.audio_storage_key)

        tenant = db.scalar(select(FamilyTenant).where(FamilyTenant.id == parent.family_id))
        if tenant:
            db.delete(tenant)
            db.commit()
        return {"erased": True, "family_id": parent.family_id}

    # ------------------------------------------------------------------
    # Endpoints M2M para Nodos GPU Arrendados (RunPod / Modal / Hetzner)
    # ------------------------------------------------------------------
    @app.post(
        "/v1/worker/leases/claim",
        response_model=Optional[WorkerLeasePayload],
        dependencies=[Depends(require_worker_token)],
        tags=["gpu-worker-plane"],
    )
    def worker_claim_lease(
        req: WorkerClaimRequest,
        response: Response,
        db: Annotated[Session, Depends(get_db)],
    ) -> Any:
        lease = queue_svc.claim_next_job(db=db, req=req)
        if lease is None:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        return lease

    @app.post(
        "/v1/worker/leases/{job_id}/heartbeat",
        dependencies=[Depends(require_worker_token)],
        tags=["gpu-worker-plane"],
    )
    def worker_heartbeat(
        job_id: str,
        req: WorkerHeartbeatRequest,
        db: Annotated[Session, Depends(get_db)],
    ) -> Dict[str, Any]:
        try:
            exp = queue_svc.heartbeat_lease(db=db, job_id=job_id, req=req)
        except LookupError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return {"ok": True, "lease_expires_at": exp.isoformat()}

    @app.post(
        "/v1/worker/leases/{job_id}/upload-audio",
        dependencies=[Depends(require_worker_token)],
        tags=["gpu-worker-plane"],
    )
    def worker_upload_audio(
        job_id: str,
        req: WorkerUploadAudioRequest,
        db: Annotated[Session, Depends(get_db)],
    ) -> Dict[str, str]:
        try:
            job = queue_svc._verify_active_lease(db, job_id, req.worker_id, req.lease_token)
            mp3_bytes = base64.b64decode(req.audio_mp3_b64.encode("ascii"), validate=True)
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        key = f"audio/{job.family_id}/{job.child_id}/cap_{job.chapter_number:04d}.mp3"
        storage.put_bytes(key, mp3_bytes, content_type="audio/mpeg")
        return {"audio_storage_key": key}

    @app.post(
        "/v1/worker/leases/{job_id}/complete",
        dependencies=[Depends(require_worker_token)],
        tags=["gpu-worker-plane"],
    )
    def worker_complete_job(
        job_id: str,
        req: WorkerCompleteJobRequest,
        db: Annotated[Session, Depends(get_db)],
    ) -> Dict[str, Any]:
        try:
            chapter = queue_svc.complete_job(db=db, job_id=job_id, req=req)
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        return {
            "ok": True,
            "chapter_id": chapter.id,
            "szigriszt_score": chapter.szigriszt_score,
            "word_count": chapter.word_count,
        }

    @app.post(
        "/v1/worker/leases/{job_id}/fail",
        dependencies=[Depends(require_worker_token)],
        tags=["gpu-worker-plane"],
    )
    def worker_fail_job(
        job_id: str,
        req: WorkerFailJobRequest,
        db: Annotated[Session, Depends(get_db)],
    ) -> Dict[str, Any]:
        try:
            job = queue_svc.fail_job(db=db, job_id=job_id, req=req)
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return {"ok": True, "job_id": job.id, "status": job.status}

    # ------------------------------------------------------------------
    # Vitrina Pública Abierta (`CC-BY-4.0` Showcase + Catálogo Ético)
    # ------------------------------------------------------------------
    showcase_dir = Path(__file__).resolve().parents[1] / "showcase_assets"
    catalog_dir = Path(__file__).resolve().parents[2] / "voces" / "catalogo_etico"

    @app.get("/v1/showcase/voices", tags=["public-showcase"])
    def get_public_ethical_voice_catalog() -> Dict[str, Any]:
        manifest_file = catalog_dir / "voice_catalog_manifest.json"
        if manifest_file.exists():
            return json.loads(manifest_file.read_text(encoding="utf-8"))
        return {
            "schema_version": "1.0",
            "governance": {
                "ley_21719_compliant": True,
                "zero_minor_voice_cloning": True,
                "zero_third_party_actor_scraping": True,
                "commercial_license": "Apache-2.0",
            },
            "voices": [],
        }

    @app.get("/v1/showcase/archetypes", tags=["public-showcase"])
    def get_public_showcase_archetypes() -> Dict[str, Any]:
        manifest_file = showcase_dir / "showcase_audio_manifest.json"
        if not manifest_file.exists():
            return {"schema_version": "1.0", "archetypes": []}
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        for item in data.get("archetypes", []):
            pid = item.get("profile_id", "")
            item["audio_stream_url"] = f"/v1/showcase/audio/{pid}"
        return data

    @app.get("/v1/showcase/audio/{profile_id}", tags=["public-showcase"])
    def stream_public_showcase_audio(profile_id: str, request: Request) -> Response:
        if not re.fullmatch(r"demo_[a-z0-9_]+", profile_id):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ID de arquetipo inválido.")
        mp3_path = showcase_dir / f"{profile_id}_cap1.mp3"
        if not mp3_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio showcase aún no generado.")

        audio_bytes = mp3_path.read_bytes()
        total_len = len(audio_bytes)
        base_headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=86400",
        }
        range_header = request.headers.get("range")
        if range_header:
            m = re.match(r"bytes=(\d+)-(\d*)", range_header.strip())
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else (total_len - 1)
                start = max(0, min(start, total_len - 1))
                end = max(start, min(end, total_len - 1))
                chunk = audio_bytes[start : end + 1]
                base_headers["Content-Range"] = f"bytes {start}-{end}/{total_len}"
                base_headers["Content-Length"] = str(len(chunk))
                return Response(
                    content=chunk,
                    status_code=status.HTTP_206_PARTIAL_CONTENT,
                    media_type="audio/mpeg",
                    headers=base_headers,
                )

        base_headers["Content-Length"] = str(total_len)
        return Response(content=audio_bytes, media_type="audio/mpeg", headers=base_headers)

    # ------------------------------------------------------------------
    # PWA Familiar Offline-First (App Shell + Service Worker + Manifest)
    # ------------------------------------------------------------------
    pwa_dir = Path(__file__).resolve().parent / "pwa"
    pwa_mime_map = {
        "app.css": "text/css; charset=utf-8",
        "app.js": "application/javascript; charset=utf-8",
        "icon.svg": "image/svg+xml",
    }

    @app.get("/", include_in_schema=False)
    def serve_pwa_shell() -> Response:
        return Response(
            content=(pwa_dir / "index.html").read_bytes(),
            media_type="text/html; charset=utf-8",
        )

    @app.get("/manifest.webmanifest", include_in_schema=False)
    def serve_pwa_manifest() -> Response:
        return Response(
            content=(pwa_dir / "manifest.webmanifest").read_bytes(),
            media_type="application/manifest+json; charset=utf-8",
        )

    @app.get("/sw.js", include_in_schema=False)
    def serve_pwa_service_worker() -> Response:
        return Response(
            content=(pwa_dir / "sw.js").read_bytes(),
            media_type="application/javascript; charset=utf-8",
            headers={"Service-Worker-Allowed": "/"},
        )

    @app.get("/pwa/{asset_name}", include_in_schema=False)
    def serve_pwa_asset(asset_name: str) -> Response:
        if asset_name not in pwa_mime_map:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recurso PWA no encontrado.")
        return Response(
            content=(pwa_dir / asset_name).read_bytes(),
            media_type=pwa_mime_map[asset_name],
        )

    return app
