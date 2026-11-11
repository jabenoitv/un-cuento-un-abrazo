# -*- coding: utf-8 -*-
"""Modelos Relacionales SQLAlchemy 2.0 (PostgreSQL en Producción / SQLite WAL en Dev/CI)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class FamilyTenant(Base):
    """Espacio aislado por familia (multi-tenant estricto)."""

    __tablename__ = "family_tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    family_name: Mapped[str] = mapped_column(String(120), nullable=False)
    plan_tier: Mapped[str] = mapped_column(String(40), nullable=False, default="esencial_digital")
    country_code: Mapped[str] = mapped_column(String(4), nullable=False, default="CL")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    parents: Mapped[List["ParentUser"]] = relationship(back_populates="family", cascade="all, delete-orphan")
    children: Mapped[List["ChildProfile"]] = relationship(back_populates="family", cascade="all, delete-orphan")
    voices: Mapped[List["VoiceProfile"]] = relationship(back_populates="family", cascade="all, delete-orphan")
    consents: Mapped[List["ConsentAuditLog"]] = relationship(back_populates="family", cascade="all, delete-orphan")


class ParentUser(Base):
    """Cuenta autenticada del adulto responsable + PIN parental separado del modo noche infantil."""

    __tablename__ = "parent_users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    family_id: Mapped[str] = mapped_column(ForeignKey("family_tenants.id", ondelete="CASCADE"), index=True)
    email: Mapped[str] = mapped_column(String(190), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(140), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    parental_pin_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    family: Mapped["FamilyTenant"] = relationship(back_populates="parents")


class ConsentAuditLog(Base):
    """Libro mayor de consentimientos y revocaciones (Ley 21.719 Chile)."""

    __tablename__ = "consent_audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    family_id: Mapped[str] = mapped_column(ForeignKey("family_tenants.id", ondelete="CASCADE"), index=True)
    parent_user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    consent_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    statement_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    family: Mapped["FamilyTenant"] = relationship(back_populates="consents")


class VoiceProfile(Base):
    """Voz de catálogo ético o voz parental cifrada AES-256-GCM con borrado criptográfico."""

    __tablename__ = "voice_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    family_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("family_tenants.id", ondelete="CASCADE"), nullable=True, index=True
    )
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    is_catalog: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    locale: Mapped[str] = mapped_column(String(16), nullable=False, default="es-CL")
    storage_key: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    key_salt_hex: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    plaintext_sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    consent_log_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    family: Mapped[Optional["FamilyTenant"]] = relationship(back_populates="voices")


class ChildProfile(Base):
    """Perfil sensorial de Diseño Universal (CERO diagnósticos médicos almacenados)."""

    __tablename__ = "child_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    family_id: Mapped[str] = mapped_column(ForeignKey("family_tenants.id", ondelete="CASCADE"), index=True)
    alias: Mapped[str] = mapped_column(String(80), nullable=False)
    age_years: Mapped[int] = mapped_column(Integer, nullable=False)

    # 4 Ejes de Diseño Universal (no clínicos)
    axis_lenguaje: Mapped[str] = mapped_column(String(40), nullable=False, default="literario_rico")
    axis_estructura: Mapped[str] = mapped_column(String(40), nullable=False, default="serie_multiverso")
    axis_estimulo_sonoro: Mapped[str] = mapped_column(String(40), nullable=False, default="dinamico_a_calmo")
    axis_foco_tematico: Mapped[str] = mapped_column(String(40), nullable=False, default="rotacion_multiverso")

    special_interests_json: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    valores_json: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    temas_evitar_json: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    closing_phrase: Mapped[str] = mapped_column(String(160), nullable=False, default="en casa te queremos mucho")
    preferred_voice_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    bible_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Evidencia cuantitativa de sanitización Ley 21.719 (cuenta purgas sin guardar el término clínico)
    sanitized_terms_blocked_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    family: Mapped["FamilyTenant"] = relationship(back_populates="children")
    jobs: Mapped[List["StoryJob"]] = relationship(back_populates="child", cascade="all, delete-orphan")
    chapters: Mapped[List["StoryChapter"]] = relationship(back_populates="child", cascade="all, delete-orphan")


class StoryJob(Base):
    """Cola distribuida con Lease/Heartbeat para Nodos GPU Arrendados (RunPod / Modal / Hetzner)."""

    __tablename__ = "story_jobs"
    __table_args__ = (
        Index("ix_story_jobs_status_created", "status", "created_at"),
        Index("ix_story_jobs_lease_exp", "status", "lease_expires_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    family_id: Mapped[str] = mapped_column(ForeignKey("family_tenants.id", ondelete="CASCADE"), index=True)
    child_id: Mapped[str] = mapped_column(ForeignKey("child_profiles.id", ondelete="CASCADE"), index=True)
    voice_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    chapter_number: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_seed: Mapped[str] = mapped_column(Text, nullable=False)
    bridge_place: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    bridge_character: Mapped[str] = mapped_column(String(120), nullable=False, default="")

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="QUEUED", index=True)
    stage_detail: Mapped[str] = mapped_column(String(64), nullable=False, default="waiting_gpu_worker")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    leased_by_worker_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    lease_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Telemetría auditable de soberanía y costos unitarios en GPU arrendada
    slm_model_tag: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    slm_params_billions: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    gpu_seconds_used: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_error: Mapped[str] = mapped_column(Text, nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    child: Mapped["ChildProfile"] = relationship(back_populates="jobs")
    chapter: Mapped[Optional["StoryChapter"]] = relationship(
        back_populates="job", uselist=False, cascade="all, delete-orphan"
    )


class StoryChapter(Base):
    """Capítulo verificado listo para reproducción o descarga offline en PWA / E-Ink."""

    __tablename__ = "story_chapters"
    __table_args__ = (UniqueConstraint("child_id", "chapter_number", name="uq_child_chapter_num"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("story_jobs.id", ondelete="CASCADE"), unique=True)
    child_id: Mapped[str] = mapped_column(ForeignKey("child_profiles.id", ondelete="CASCADE"), index=True)
    chapter_number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    story_text: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    szigriszt_score: Mapped[float] = mapped_column(Float, nullable=False)
    fernandez_huerta_score: Mapped[float] = mapped_column(Float, nullable=False)
    whisper_wer: Mapped[float] = mapped_column(Float, nullable=False)
    speaker_similarity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    audio_storage_key: Mapped[str] = mapped_column(String(255), nullable=False)
    parent_guide_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    qa_report_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    job: Mapped["StoryJob"] = relationship(back_populates="chapter")
    child: Mapped["ChildProfile"] = relationship(back_populates="chapters")
