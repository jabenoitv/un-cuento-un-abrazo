# -*- coding: utf-8 -*-
"""Coordinador Transaccional de Cola para Nodos GPU Arrendados (`SKIP LOCKED` + Lease Recovery)."""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import PlatformSettings
from .domain_bridge import (
    analizar_cuento_multinino,
    build_bible_dict_from_child,
    construir_prompt_escritor_local,
    generar_guia_padres,
    verificar_anti_regurgitacion_ngramas,
)
from .models import ChildProfile, StoryChapter, StoryJob, VoiceProfile
from .schemas import (
    WorkerClaimRequest,
    WorkerCompleteJobRequest,
    WorkerFailJobRequest,
    WorkerHeartbeatRequest,
    WorkerLeasePayload,
)
from .security import VoiceVaultCrypto
from .storage import ObjectStorageAdapter


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class DistributedJobQueueService:
    """Gestiona el ciclo de vida de trabajos entre el Control Plane y GPUs efímeras arrendadas."""

    def __init__(
        self,
        settings: PlatformSettings,
        storage: ObjectStorageAdapter,
        vault: VoiceVaultCrypto,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.vault = vault

    def requeue_expired_leases(self, db: Session) -> int:
        """Si una GPU spot arrendada cae, recupera automáticamente sus trabajos vencidos."""
        now = _now_utc()
        active_leased = db.scalars(
            select(StoryJob).where(StoryJob.status.like("LEASED_%"))
        ).all()
        recovered = 0
        for job in active_leased:
            exp = _as_utc(job.lease_expires_at)
            if exp is not None and exp < now:
                if job.attempts >= job.max_attempts:
                    job.status = "FAILED_PERMANENT"
                    job.stage_detail = "lease_expired_max_attempts"
                    job.last_error = "El nodo GPU expiró el lease tras agotar reintentos máximos."
                else:
                    job.status = "QUEUED"
                    job.stage_detail = "requeued_after_spot_preemption"
                    job.leased_by_worker_id = None
                    job.lease_token = None
                    job.lease_expires_at = None
                recovered += 1
        if recovered:
            db.commit()
        return recovered

    def enqueue_story_job(
        self,
        db: Session,
        family_id: str,
        child: ChildProfile,
        parent_seed: str,
        bridge_place: str = "",
        bridge_character: str = "",
        requested_voice_id: Optional[str] = None,
    ) -> StoryJob:
        voice_id = requested_voice_id or child.preferred_voice_id or "catalog-chileno-sereno-m1"
        voice = db.scalar(select(VoiceProfile).where(VoiceProfile.id == voice_id))
        if not voice or voice.status != "ACTIVE":
            raise PermissionError("El perfil de voz seleccionado no existe o fue revocado criptográficamente.")
        if not voice.is_catalog and voice.family_id != family_id:
            raise PermissionError("Aislamiento multi-tenant: no puede usar una voz privada de otra familia.")

        max_chap = db.scalar(
            select(func.max(StoryJob.chapter_number)).where(StoryJob.child_id == child.id)
        )
        next_chap = int(max_chap or 0) + 1

        job = StoryJob(
            family_id=family_id,
            child_id=child.id,
            voice_id=voice.id,
            chapter_number=next_chap,
            parent_seed=parent_seed,
            bridge_place=bridge_place,
            bridge_character=bridge_character,
            status="QUEUED",
            stage_detail="queued_for_gpu_worker",
            max_attempts=self.settings.max_job_attempts,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    def claim_next_job(self, db: Session, req: WorkerClaimRequest) -> Optional[WorkerLeasePayload]:
        """Entrega un lease exclusivo a un nodo GPU arrendado (con `SKIP LOCKED` en PostgreSQL)."""
        self.requeue_expired_leases(db)

        stmt = (
            select(StoryJob)
            .where(StoryJob.status.in_(["QUEUED", "FAILED_RETRYABLE"]))
            .order_by(StoryJob.created_at.asc())
            .limit(1)
        )
        dialect = db.bind.dialect.name if db.bind is not None else "sqlite"
        if dialect == "postgresql":
            stmt = stmt.with_for_update(skip_locked=True)

        job = db.scalar(stmt)
        if not job:
            return None

        child = db.scalar(select(ChildProfile).where(ChildProfile.id == job.child_id))
        if not child:
            job.status = "FAILED_PERMANENT"
            job.last_error = "Perfil infantil eliminado antes de procesar."
            db.commit()
            return None

        voice_id = job.voice_id or "catalog-chileno-sereno-m1"
        voice = db.scalar(select(VoiceProfile).where(VoiceProfile.id == voice_id))
        if not voice or voice.status != "ACTIVE":
            job.status = "FAILED_PERMANENT"
            job.last_error = "Voz revocada antes de iniciar síntesis (Ley 21.719)."
            db.commit()
            return None

        ephemeral_b64: Optional[str] = None
        voice_mode = "catalog"
        if not voice.is_catalog:
            voice_mode = "parental_ephemeral"
            cipher_blob = self.storage.get_bytes(voice.storage_key)
            raw_pt = self.vault.decrypt_embedding(
                ciphertext_blob=cipher_blob,
                family_id=job.family_id,
                key_salt_hex=voice.key_salt_hex,
            )
            ephemeral_b64 = self.vault.encode_ephemeral_worker_b64(raw_pt)

        axes = {
            "lenguaje": child.axis_lenguaje,
            "estructura": child.axis_estructura,
            "estimulo_sonoro": child.axis_estimulo_sonoro,
            "foco_tematico": child.axis_foco_tematico,
        }
        bible_dict = build_bible_dict_from_child(
            alias=child.alias,
            age_years=child.age_years,
            axes=axes,
            special_interests=child.special_interests_json,
            valores=child.valores_json,
            temas_evitar=child.temas_evitar_json,
            closing_phrase=child.closing_phrase,
            extra_bible=child.bible_json,
        )
        seed_enriched = job.parent_seed
        if job.bridge_place or job.bridge_character:
            seed_enriched = (
                f"{job.parent_seed} (Guiño opcional de universo: {job.bridge_place} {job.bridge_character})"
            ).strip()
        writer_prompt = construir_prompt_escritor_local(bible_dict, seed_enriched)

        now = _now_utc()
        lease_token = secrets.token_hex(24)
        expires_at = now + timedelta(seconds=self.settings.lease_ttl_seconds)

        job.status = "LEASED_WRITING"
        job.stage_detail = f"leased_by:{req.gpu_model}"
        job.attempts += 1
        job.leased_by_worker_id = req.worker_id
        job.lease_token = lease_token
        job.lease_expires_at = expires_at
        db.commit()

        return WorkerLeasePayload(
            job_id=job.id,
            lease_token=lease_token,
            lease_expires_at=expires_at,
            chapter_number=job.chapter_number,
            parent_seed=job.parent_seed,
            bridge_place=job.bridge_place,
            bridge_character=job.bridge_character,
            writer_prompt=writer_prompt,
            bible_spec=bible_dict,
            voice_mode=voice_mode,  # type: ignore[arg-type]
            voice_id=voice.id,
            ephemeral_voice_embedding_b64=ephemeral_b64,
        )

    def _verify_active_lease(self, db: Session, job_id: str, worker_id: str, lease_token: str) -> StoryJob:
        job = db.scalar(select(StoryJob).where(StoryJob.id == job_id))
        if not job:
            raise LookupError("Trabajo no encontrado.")
        if job.leased_by_worker_id != worker_id or job.lease_token != lease_token:
            raise PermissionError("Lease inválido o reasignado a otro nodo GPU.")
        exp = _as_utc(job.lease_expires_at)
        if exp is not None and exp < _now_utc():
            raise PermissionError("Lease expirado.")
        return job

    def heartbeat_lease(self, db: Session, job_id: str, req: WorkerHeartbeatRequest) -> datetime:
        job = self._verify_active_lease(db, job_id, req.worker_id, req.lease_token)
        new_exp = _now_utc() + timedelta(seconds=self.settings.lease_ttl_seconds)
        job.lease_expires_at = new_exp
        job.stage_detail = req.stage_detail
        if "synth" in req.stage_detail.lower():
            job.status = "LEASED_SYNTHESIZING"
        elif "verif" in req.stage_detail.lower() or "whisper" in req.stage_detail.lower():
            job.status = "LEASED_VERIFYING"
        db.commit()
        return new_exp

    def complete_job(self, db: Session, job_id: str, req: WorkerCompleteJobRequest) -> StoryChapter:
        """Doble barrera de seguridad: el Control Plane re-audita el cuento antes de publicarlo al niño."""
        job = self._verify_active_lease(db, job_id, req.worker_id, req.lease_token)
        child = db.scalar(select(ChildProfile).where(ChildProfile.id == job.child_id))
        if not child:
            raise LookupError("Perfil infantil no encontrado.")

        if not self.storage.exists(req.audio_storage_key):
            raise ValueError(f"El archivo de audio '{req.audio_storage_key}' no existe en el Object Storage.")

        axes = {
            "lenguaje": child.axis_lenguaje,
            "estructura": child.axis_estructura,
            "estimulo_sonoro": child.axis_estimulo_sonoro,
            "foco_tematico": child.axis_foco_tematico,
        }
        bible_dict = build_bible_dict_from_child(
            alias=child.alias,
            age_years=child.age_years,
            axes=axes,
            special_interests=child.special_interests_json,
            valores=child.valores_json,
            temas_evitar=child.temas_evitar_json,
            closing_phrase=child.closing_phrase,
            extra_bible=child.bible_json,
        )

        qa = analizar_cuento_multinino(req.story_text, bible_dict)
        ok_ngram, collisions = verificar_anti_regurgitacion_ngramas(req.story_text)
        if not ok_ngram:
            qa["valido"] = False
            qa["errores"].extend([f"Anti-regurgitación: {c}" for c in collisions])

        if not qa["valido"]:
            raise ValueError(
                f"Control Plane bloqueó publicación por incumplir reglas neuro-inclusivas: {qa['errores']}"
            )

        parent_guide = generar_guia_padres(req.story_text, bible_dict, req.title)
        cost_usd = round((req.gpu_seconds_used / 3600.0) * self.settings.rented_gpu_hourly_usd, 6)

        chapter = StoryChapter(
            job_id=job.id,
            child_id=child.id,
            chapter_number=job.chapter_number,
            title=req.title,
            story_text=req.story_text,
            word_count=int(qa["legibilidad"]["palabras"]),
            szigriszt_score=float(qa["legibilidad"]["szigriszt_pazos"]),
            fernandez_huerta_score=float(qa["legibilidad"]["fernandez_huerta"]),
            whisper_wer=req.whisper_wer,
            speaker_similarity=req.speaker_similarity,
            audio_storage_key=req.audio_storage_key,
            parent_guide_json=parent_guide,
            qa_report_json={
                "linter": qa,
                "editorial_review": req.editorial_review,
                "anti_regurgitation_clean": True,
                "slm_model_tag": req.slm_model_tag,
                "slm_params_billions": req.slm_params_billions,
                "gpu_seconds_used": req.gpu_seconds_used,
                "estimated_cost_usd": cost_usd,
            },
        )
        db.add(chapter)

        job.status = "READY"
        job.stage_detail = "verified_and_published"
        job.slm_model_tag = req.slm_model_tag
        job.slm_params_billions = req.slm_params_billions
        job.gpu_seconds_used = req.gpu_seconds_used
        job.estimated_cost_usd = cost_usd
        job.lease_token = None
        job.lease_expires_at = None
        job.completed_at = _now_utc()
        job.last_error = ""
        db.commit()
        db.refresh(chapter)
        return chapter

    def fail_job(self, db: Session, job_id: str, req: WorkerFailJobRequest) -> StoryJob:
        job = self._verify_active_lease(db, job_id, req.worker_id, req.lease_token)
        job.last_error = req.error_message
        job.lease_token = None
        job.lease_expires_at = None
        if req.retryable and job.attempts < job.max_attempts:
            job.status = "FAILED_RETRYABLE"
            job.stage_detail = f"retry_scheduled_{job.attempts}_of_{job.max_attempts}"
        else:
            job.status = "FAILED_PERMANENT"
            job.stage_detail = "failed_permanently"
        db.commit()
        db.refresh(job)
        return job
