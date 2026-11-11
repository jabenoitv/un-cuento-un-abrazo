# -*- coding: utf-8 -*-
"""Configuración 12-Factor para Control Plane (VPS) y GPU Workers Arrendados."""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlatformSettings(BaseSettings):
    """Variables de entorno prefijadas con `ROANSO_` para despliegue en contenedores."""

    model_config = SettingsConfigDict(
        env_prefix="ROANSO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["dev", "test", "prod"] = "dev"
    database_url: str = "sqlite:///./roanso_platform_dev.db"

    # Autenticación familiar JWT
    jwt_secret_key: str = "dev-only-change-in-production-roanso-2026-secret-key-32b"
    jwt_expire_minutes: int = 60 * 24 * 7  # 7 días

    # Bóveda biométrica AES-256-GCM (32 bytes codificados en urlsafe base64)
    voice_vault_master_key_b64: str = Field(
        default_factory=lambda: base64.urlsafe_b64encode(b"dev-roanso-aes256-master-key-32b").decode("ascii")
    )

    # Autenticación M2M para nodos GPU arrendados (RunPod / Modal / Hetzner GPU)
    worker_shared_token: str = "dev-worker-shared-token-change-in-prod"
    lease_ttl_seconds: int = 180
    max_job_attempts: int = 3

    # Almacenamiento de objetos (Cloudflare R2 / AWS S3 / MinIO / disco local en dev)
    storage_backend: Literal["local", "s3"] = "local"
    local_storage_dir: Path = Path("./storage_data")
    s3_endpoint_url: str = ""
    s3_bucket: str = "roanso-cuentos-prod"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_region: str = "auto"
    presigned_url_ttl_seconds: int = 3600

    # Gobernanza Human-Scale AI (Mozilla Incubator compliance)
    enforce_human_scale_ceiling: bool = True
    max_model_params_billions: float = 10.0

    # Costo referencial de nodo GPU arrendado (ej. NVIDIA L4 / RTX 4090 ~US$0.54/h)
    rented_gpu_hourly_usd: float = 0.54

    @model_validator(mode="after")
    def _validate_prod_secrets(self) -> "PlatformSettings":
        raw_key = base64.urlsafe_b64decode(self.voice_vault_master_key_b64.encode("ascii"))
        if len(raw_key) != 32:
            raise ValueError("ROANSO_VOICE_VAULT_MASTER_KEY_B64 debe decodificar exactamente a 32 bytes (AES-256).")
        if self.environment == "prod":
            if "dev-" in self.jwt_secret_key or "dev-" in self.worker_shared_token:
                raise ValueError("En producción (ROANSO_ENVIRONMENT=prod) se exigen secretos reales, no valores dev.")
            if raw_key == b"dev-roanso-aes256-master-key-32b":
                raise ValueError("En producción debe configurar ROANSO_VOICE_VAULT_MASTER_KEY_B64 único de 32 bytes.")
        return self

    @property
    def master_key_bytes(self) -> bytes:
        return base64.urlsafe_b64decode(self.voice_vault_master_key_b64.encode("ascii"))


def get_settings_from_env() -> PlatformSettings:
    """Construye configuración leyendo variables actuales de entorno."""
    return PlatformSettings(_env_file=os.environ.get("ROANSO_ENV_FILE", ".env"))
