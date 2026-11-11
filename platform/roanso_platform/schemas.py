# -*- coding: utf-8 -*-
"""Esquemas Pydantic v2 con Sanitización Automática Ley 21.719 y Reglas Human-Scale."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .domain_bridge import (
    MAX_PARAM_BILLIONS,
    normalizar_perfil_sensorial,
    sanitizar_onboarding_ley21719,
    verificar_soberania_human_scale,
)


class FamilyRegisterRequest(BaseModel):
    family_name: str = Field(min_length=2, max_length=120)
    full_name: str = Field(min_length=2, max_length=140)
    email: str = Field(min_length=5, max_length=190, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=128)
    parental_pin: str = Field(min_length=4, max_length=8, pattern=r"^\d{4,8}$")
    plan_tier: Literal["esencial_digital", "familiar_voces", "lector_eink"] = "esencial_digital"
    accepted_ley21719_notice: bool

    @field_validator("accepted_ley21719_notice")
    @classmethod
    def _must_accept_privacy(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Debe aceptar expresamente el aviso de privacidad de menores (Ley 21.719).")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class VerifyPinRequest(BaseModel):
    parental_pin: str = Field(pattern=r"^\d{4,8}$")


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    family_id: str
    parent_user_id: str
    plan_tier: str


class SensoryAxesDTO(BaseModel):
    lenguaje: str = "literario_rico"
    estructura: str = "serie_multiverso"
    estimulo_sonoro: str = "dinamico_a_calmo"
    foco_tematico: str = "rotacion_multiverso"


class ChildProfileCreate(BaseModel):
    """Frontera estricta Ley 21.719: purga cualquier etiqueta clínica antes de llegar a PostgreSQL."""

    alias: str = Field(min_length=2, max_length=60)
    age_years: int = Field(ge=3, le=13)
    perfil_sensorial: SensoryAxesDTO = Field(default_factory=SensoryAxesDTO)
    intereses_especiales: List[str] = Field(default_factory=list, max_length=12)
    valores_trabajar: List[str] = Field(default_factory=lambda: ["calma", "empatía mutua"], max_length=10)
    temas_evitar: List[str] = Field(default_factory=list, max_length=15)
    notas_libres_padre: str = Field(default="", max_length=600)
    closing_phrase: str = Field(default="en casa te queremos mucho", min_length=4, max_length=160)
    preferred_voice_id: Optional[str] = "catalog-chileno-sereno-m1"
    mascota_nombre: str = Field(default="Canela", min_length=2, max_length=50)
    mascota_especie: str = Field(default="perrita", min_length=2, max_length=50)

    # Campo interno calculado durante validación (nunca almacena las palabras clínicas)
    sanitized_terms_blocked_count: int = 0

    @model_validator(mode="after")
    def _enforce_ley21719_sanitization(self) -> "ChildProfileCreate":
        blocked_total = 0

        clean_alias, det = sanitizar_onboarding_ley21719(self.alias)
        blocked_total += len(det)
        if not clean_alias:
            raise ValueError("El apodo del niño no puede consistir en una etiqueta clínica.")
        self.alias = clean_alias

        def _clean_list(items: List[str]) -> List[str]:
            nonlocal blocked_total
            out: List[str] = []
            for raw in items:
                c, found = sanitizar_onboarding_ley21719(raw)
                blocked_total += len(found)
                if c:
                    out.append(c)
            return out

        self.intereses_especiales = _clean_list(self.intereses_especiales)
        self.valores_trabajar = _clean_list(self.valores_trabajar)
        self.temas_evitar = _clean_list(self.temas_evitar)

        clean_notes, det_notes = sanitizar_onboarding_ley21719(self.notas_libres_padre)
        blocked_total += len(det_notes)
        self.notas_libres_padre = clean_notes

        clean_close, det_close = sanitizar_onboarding_ley21719(self.closing_phrase)
        blocked_total += len(det_close)
        self.closing_phrase = clean_close or "en casa te queremos mucho"

        norm_axes = normalizar_perfil_sensorial({"perfil_sensorial": self.perfil_sensorial.model_dump()})
        self.perfil_sensorial = SensoryAxesDTO(**norm_axes)
        self.sanitized_terms_blocked_count = blocked_total
        return self


class ChildProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    alias: str
    age_years: int
    perfil_sensorial: SensoryAxesDTO
    intereses_especiales: List[str]
    valores_trabajar: List[str]
    temas_evitar: List[str]
    closing_phrase: str
    preferred_voice_id: Optional[str]
    sanitized_terms_blocked_count: int
    created_at: datetime


class ParentalVoiceEnrollRequest(BaseModel):
    display_name: str = Field(min_length=2, max_length=120)
    adult_full_name: str = Field(min_length=3, max_length=140)
    spoken_consent_transcript: str = Field(min_length=25, max_length=600)
    is_minor_voice_declaration_false: bool = Field(
        description="Debe ser True confirmando que la muestra pertenece exclusivamente a un adulto (>18)."
    )
    embedding_b64: str = Field(min_length=16)

    @model_validator(mode="after")
    def _validate_adult_consent(self) -> "ParentalVoiceEnrollRequest":
        if not self.is_minor_voice_declaration_false:
            raise ValueError("Prohibición ética absoluta: ROANSO jamás clona ni almacena voces de menores.")
        text_low = self.spoken_consent_transcript.lower()
        required_tokens = ["autorizo", "roanso", "voz"]
        missing = [tok for tok in required_tokens if tok not in text_low]
        if missing:
            raise ValueError(
                f"La frase hablada de consentimiento biométrico no contiene términos obligatorios: {missing}"
            )
        return self


class VoiceProfileResponse(BaseModel):
    id: str
    display_name: str
    is_catalog: bool
    locale: str
    status: str
    created_at: datetime
    revoked_at: Optional[datetime] = None


class StoryJobEnqueueRequest(BaseModel):
    child_id: str
    parent_seed: str = Field(min_length=8, max_length=800)
    bridge_place: str = Field(default="", max_length=120)
    bridge_character: str = Field(default="", max_length=120)
    voice_id: Optional[str] = None

    @field_validator("parent_seed")
    @classmethod
    def _sanitize_seed(cls, v: str) -> str:
        clean, _ = sanitizar_onboarding_ley21719(v)
        if len(clean) < 5:
            raise ValueError("La semilla cotidiana quedó vacía tras aplicar el filtro de datos clínicos.")
        return clean


class StoryJobResponse(BaseModel):
    id: str
    child_id: str
    chapter_number: int
    status: str
    stage_detail: str
    attempts: int
    leased_by_worker_id: Optional[str]
    slm_model_tag: str
    slm_params_billions: float
    gpu_seconds_used: float
    estimated_cost_usd: float
    last_error: str
    created_at: datetime
    completed_at: Optional[datetime]


# --- Contratos M2M para Nodos GPU Arrendados (RunPod / Modal / Hetzner GPU) ---


class WorkerClaimRequest(BaseModel):
    worker_id: str = Field(min_length=3, max_length=120)
    gpu_model: str = Field(default="NVIDIA-L4-24GB", max_length=80)
    inference_endpoint: str = Field(default="http://127.0.0.1:8000/v1")
    slm_params_billions: float = Field(default=7.6, gt=0.0)

    @model_validator(mode="after")
    def _check_worker_sovereignty(self) -> "WorkerClaimRequest":
        try:
            verificar_soberania_human_scale(self.inference_endpoint, param_billions=self.slm_params_billions)
        except PermissionError as exc:
            raise ValueError(str(exc)) from exc
        return self


class WorkerLeasePayload(BaseModel):
    job_id: str
    lease_token: str
    lease_expires_at: datetime
    chapter_number: int
    parent_seed: str
    bridge_place: str
    bridge_character: str
    writer_prompt: str
    bible_spec: Dict[str, Any]
    voice_mode: Literal["catalog", "parental_ephemeral"]
    voice_id: str
    ephemeral_voice_embedding_b64: Optional[str] = None


class WorkerHeartbeatRequest(BaseModel):
    worker_id: str
    lease_token: str
    stage_detail: str = Field(default="processing", max_length=64)


class WorkerUploadAudioRequest(BaseModel):
    worker_id: str
    lease_token: str
    audio_mp3_b64: str = Field(min_length=16)


class WorkerCompleteJobRequest(BaseModel):
    worker_id: str
    lease_token: str
    title: str = Field(min_length=3, max_length=180)
    story_text: str = Field(min_length=120)
    audio_storage_key: str = Field(min_length=5, max_length=255)
    whisper_wer: float = Field(ge=0.0, le=1.0)
    speaker_similarity: float = Field(ge=0.0, le=1.0, default=0.88)
    slm_model_tag: str = Field(min_length=2, max_length=120)
    slm_params_billions: float = Field(gt=0.0, le=MAX_PARAM_BILLIONS)
    inference_endpoint_used: str = Field(default="http://127.0.0.1:8000/v1")
    gpu_seconds_used: float = Field(ge=0.0)
    editorial_review: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_qa_gates(self) -> "WorkerCompleteJobRequest":
        try:
            verificar_soberania_human_scale(self.inference_endpoint_used, param_billions=self.slm_params_billions)
        except PermissionError as exc:
            raise ValueError(str(exc)) from exc
        if self.whisper_wer > 0.12:
            raise ValueError(f"QA Acústico rechazado: Whisper WER={self.whisper_wer:.3f} supera el máximo 0.12.")
        return self


class WorkerFailJobRequest(BaseModel):
    worker_id: str
    lease_token: str
    error_message: str = Field(min_length=3, max_length=1000)
    retryable: bool = True
