# -*- coding: utf-8 -*-
"""Fixtures compartidos para la suite Pytest de ROANSO Platform."""
from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from roanso_platform.api import create_app
from roanso_platform.config import PlatformSettings


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
