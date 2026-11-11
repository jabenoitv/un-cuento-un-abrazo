# -*- coding: utf-8 -*-
"""Seguridad, JWT, PIN Parental y Bóveda Biométrica AES-256-GCM (Ley 21.719)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

PBKDF2_ITERATIONS = 310_000


def hash_secret_pbkdf2(secret: str) -> str:
    """Hashea contraseñas o PIN parental con PBKDF2-HMAC-SHA256 (310k iteraciones)."""
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    digest = kdf.derive(secret.encode("utf-8"))
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_secret_pbkdf2(secret: str, encoded: str) -> bool:
    """Verifica en tiempo constante una contraseña o PIN parental."""
    try:
        scheme, iters_str, salt_hex, digest_hex = encoded.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=bytes.fromhex(salt_hex),
            iterations=int(iters_str),
        )
        candidate = kdf.derive(secret.encode("utf-8"))
        return hmac.compare_digest(candidate, bytes.fromhex(digest_hex))
    except Exception:
        return False


def create_access_token(
    subject_user_id: str,
    family_id: str,
    secret_key: str,
    expire_minutes: int = 10080,
    scope: str = "family_parent",
) -> str:
    """Emite JWT firmado (HS256) con aislamiento estricto por `family_id`."""
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": subject_user_id,
        "fam": family_id,
        "scope": scope,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, secret_key, algorithm="HS256")


def decode_access_token(token: str, secret_key: str) -> Dict[str, Any]:
    """Valida firma y expiración de JWT familiar."""
    return jwt.decode(token, secret_key, algorithms=["HS256"])


def sha256_hex(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class EncryptedVoiceArtifact:
    """Artefacto cifrado listo para persistir en R2/S3 sin exponer biometría en claro."""

    ciphertext_blob: bytes
    key_salt_hex: str
    plaintext_sha256: str


class VoiceVaultCrypto:
    """Cifrado autenticado AES-256-GCM con clave efímera derivada por HKDF.

    Diseño de Borrado Criptográfico (Ley 21.719):
    - Cada perfil de voz genera un `key_salt` aleatorio único de 16 bytes.
    - La clave AES-256 nunca se guarda; se deriva al vuelo vía `HKDF(master_key, salt, info=family_id)`.
    - Al revocar el consentimiento (`DELETE /v1/voices/{id}`), se destruye el objeto en S3/R2
      Y se sobreescribe `key_salt_hex = ""` en PostgreSQL, volviendo irrecuperable cualquier copia fría.
    """

    def __init__(self, master_key: bytes) -> None:
        if len(master_key) != 32:
            raise ValueError("VoiceVaultCrypto requiere una clave maestra exacta de 32 bytes.")
        self._master_key = master_key

    def _derive_voice_key(self, family_id: str, key_salt: bytes) -> bytes:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=key_salt,
            info=f"roanso-voice-vault-v1:{family_id}".encode("utf-8"),
        )
        return hkdf.derive(self._master_key)

    def encrypt_embedding(self, plaintext_bytes: bytes, family_id: str) -> EncryptedVoiceArtifact:
        if not plaintext_bytes:
            raise ValueError("El embedding de voz no puede estar vacío.")
        key_salt = os.urandom(16)
        nonce = os.urandom(12)
        derived_key = self._derive_voice_key(family_id=family_id, key_salt=key_salt)
        aesgcm = AESGCM(derived_key)
        aad = f"family:{family_id}".encode("utf-8")
        ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, aad)
        return EncryptedVoiceArtifact(
            ciphertext_blob=nonce + ciphertext,
            key_salt_hex=key_salt.hex(),
            plaintext_sha256=sha256_hex(plaintext_bytes),
        )

    def decrypt_embedding(self, ciphertext_blob: bytes, family_id: str, key_salt_hex: str) -> bytes:
        if not key_salt_hex:
            raise PermissionError("Esta voz fue revocada y destruida criptográficamente (Ley 21.719).")
        if len(ciphertext_blob) <= 28:
            raise ValueError("Blob cifrado inválido o truncado.")
        nonce = ciphertext_blob[:12]
        ciphertext = ciphertext_blob[12:]
        derived_key = self._derive_voice_key(family_id=family_id, key_salt=bytes.fromhex(key_salt_hex))
        aesgcm = AESGCM(derived_key)
        aad = f"family:{family_id}".encode("utf-8")
        return aesgcm.decrypt(nonce, ciphertext, aad)

    @staticmethod
    def encode_ephemeral_worker_b64(raw_bytes: bytes) -> str:
        return base64.b64encode(raw_bytes).decode("ascii")
