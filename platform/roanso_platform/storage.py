# -*- coding: utf-8 -*-
"""Adaptador de Almacenamiento de Objetos (Cloudflare R2 / AWS S3 / MinIO / Local)."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import boto3
from botocore.config import Config as BotoConfig

from .config import PlatformSettings


class ObjectStorageAdapter(Protocol):
    """Contrato uniforme para guardar embeddings cifrados (.enc) y capítulos (.mp3)."""

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...

    def get_bytes(self, key: str) -> bytes: ...

    def delete_key(self, key: str) -> None: ...

    def exists(self, key: str) -> bool: ...

    def get_read_url(self, key: str, expires_in: int = 3600) -> str: ...


def _sanitize_key(key: str) -> str:
    clean = key.strip("/").replace("\\", "/")
    if ".." in clean.split("/"):
        raise ValueError(f"Storage key inválida: {key}")
    return clean


class LocalObjectStorage:
    """Almacenamiento local aislado por clave para desarrollo, CI e integración sin red."""

    def __init__(self, root_dir: Path, public_prefix: str = "/v1/media") -> None:
        self.root_dir = root_dir.resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.public_prefix = public_prefix.rstrip("/")

    def _resolve(self, key: str) -> Path:
        clean = _sanitize_key(key)
        target = (self.root_dir / clean).resolve()
        if self.root_dir not in target.parents and target != self.root_dir:
            raise ValueError("Intento de path traversal bloqueado.")
        return target

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        clean = _sanitize_key(key)
        target = self._resolve(clean)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return clean

    def get_bytes(self, key: str) -> bytes:
        target = self._resolve(key)
        if not target.exists():
            raise FileNotFoundError(f"Objeto no encontrado: {key}")
        return target.read_bytes()

    def delete_key(self, key: str) -> None:
        target = self._resolve(key)
        if target.exists():
            target.unlink()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def get_read_url(self, key: str, expires_in: int = 3600) -> str:
        clean = _sanitize_key(key)
        return f"{self.public_prefix}/{clean}"


class S3CompatibleStorage:
    """Almacenamiento productivo S3 / Cloudflare R2 / MinIO con URLs pre-firmadas."""

    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str = "auto",
    ) -> None:
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=BotoConfig(signature_version="s3v4"),
        )

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        clean = _sanitize_key(key)
        self._client.put_object(
            Bucket=self.bucket,
            Key=clean,
            Body=data,
            ContentType=content_type,
        )
        return clean

    def get_bytes(self, key: str) -> bytes:
        clean = _sanitize_key(key)
        resp = self._client.get_object(Bucket=self.bucket, Key=clean)
        return resp["Body"].read()

    def delete_key(self, key: str) -> None:
        clean = _sanitize_key(key)
        self._client.delete_object(Bucket=self.bucket, Key=clean)

    def exists(self, key: str) -> bool:
        clean = _sanitize_key(key)
        try:
            self._client.head_object(Bucket=self.bucket, Key=clean)
            return True
        except Exception:
            return False

    def get_read_url(self, key: str, expires_in: int = 3600) -> str:
        clean = _sanitize_key(key)
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": clean},
            ExpiresIn=expires_in,
        )


def build_storage_adapter(settings: PlatformSettings) -> ObjectStorageAdapter:
    if settings.storage_backend == "s3":
        return S3CompatibleStorage(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            region=settings.s3_region,
        )
    return LocalObjectStorage(root_dir=settings.local_storage_dir)
