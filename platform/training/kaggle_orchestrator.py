# -*- coding: utf-8 -*-
"""Orquestador Local -> Kaggle GPU (`T4 x2`, 30h/semana US$0) para ROANSO SpA.

Automatiza de punta a punta el ciclo de entrenamiento sin quemar presupuesto en GPUs arrendadas:
1. Soporta tanto tokens modernos `KGAT_...` (Bearer API v1) como llaves clásicas en `~/.kaggle/kaggle.json`.
2. Aplica una cerca de seguridad estricta (`ROANSO_REQUIRED_SLUG_PREFIX = "roanso-"`): jamás toca,
   modifica ni sobrescribe otros datasets o notebooks preexistentes en la cuenta de Kaggle del usuario.
3. Empaqueta datasets **100% Privados por defecto (`isPrivate: true`)** con verificación previa `SHA-256`
   contra `licencias_corpus.csv` o `actor_voice_license.json`.
4. Permite preparar bundles locales (`prepare-slm`, `prepare-tts`), enviarlos a ejecución GPU remota en
   Kaggle (`push-slm`), consultar estado (`status`) y descargar los adaptadores entrenados (`pull-output`).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from roanso_platform.lora_kit import (  # noqa: E402
    build_sft_and_dpo_datasets,
    verify_corpus_integrity_against_ledger,
)

ROANSO_REQUIRED_SLUG_PREFIX = "roanso-"
DEFAULT_SLM_DATASET_SLUG = "roanso-cuentos-chileno-sft-v1"
DEFAULT_SLM_KERNEL_SLUG = "roanso-qlora-cuentos-chileno-v1"
DEFAULT_TTS_KERNEL_SLUG = "roanso-tts-voces-chile-v1"


@dataclass(frozen=True)
class KaggleCredentials:
    username: str
    token: str
    is_bearer_kgat: bool


def assert_safe_roanso_slug(slug: str) -> str:
    """Impide tocar proyectos ajenos en la cuenta Kaggle exigiendo prefijo exclusivo `roanso-`."""
    clean = slug.strip().lower()
    if not clean.startswith(ROANSO_REQUIRED_SLUG_PREFIX):
        raise PermissionError(
            f"Seguridad ROANSO: el identificador '{slug}' debe comenzar con '{ROANSO_REQUIRED_SLUG_PREFIX}' "
            "para proteger otros proyectos y datasets en tu cuenta de Kaggle."
        )
    if "/" in clean or ".." in clean or " " in clean:
        raise ValueError(f"Slug Kaggle inválido: '{slug}'")
    return clean


def load_kaggle_credentials(kaggle_json_path: Path | None = None) -> KaggleCredentials:
    """Carga credenciales desde variables de entorno o `~/.kaggle/kaggle.json` (con soporte nativo `KGAT_`)."""
    env_user = os.environ.get("KAGGLE_USERNAME", "").strip()
    env_token = (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY") or "").strip()

    if not (env_user and env_token):
        cfg_path = kaggle_json_path or (Path.home() / ".kaggle" / "kaggle.json")
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"No se encontró credencial Kaggle en {cfg_path} ni en KAGGLE_USERNAME/KAGGLE_API_TOKEN."
            )
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        env_user = env_user or str(raw.get("username") or "").strip()
        env_token = env_token or str(raw.get("key") or raw.get("token") or "").strip()

    if not env_user or not env_token:
        raise ValueError("Credencial Kaggle incompleta: se requieren `username` y `key`/`KAGGLE_API_TOKEN`.")

    is_kgat = env_token.startswith("KGAT_")
    os.environ["KAGGLE_USERNAME"] = env_user
    if is_kgat:
        os.environ["KAGGLE_API_TOKEN"] = env_token
    else:
        os.environ["KAGGLE_KEY"] = env_token

    return KaggleCredentials(username=env_user, token=env_token, is_bearer_kgat=is_kgat)


def get_authenticated_kaggle_api(kaggle_json_path: Path | None = None) -> tuple[Any, KaggleCredentials]:
    creds = load_kaggle_credentials(kaggle_json_path=kaggle_json_path)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api, creds


def stage_slm_kaggle_bundle(
    corpus_dir: Path,
    staging_root: Path,
    kaggle_username: str,
    dataset_slug: str = DEFAULT_SLM_DATASET_SLUG,
    kernel_slug: str = DEFAULT_SLM_KERNEL_SLUG,
) -> Dict[str, Any]:
    """Prepara localmente el Dataset Privado + Kernel GPU (`T4 x2`) listos para subir a Kaggle."""
    safe_ds_slug = assert_safe_roanso_slug(dataset_slug)
    safe_kn_slug = assert_safe_roanso_slug(kernel_slug)

    verified = verify_corpus_integrity_against_ledger(corpus_dir)

    ds_dir = staging_root / "slm_dataset"
    kn_dir = staging_root / "slm_kernel"
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    if kn_dir.exists():
        shutil.rmtree(kn_dir)
    ds_dir.mkdir(parents=True, exist_ok=True)
    kn_dir.mkdir(parents=True, exist_ok=True)

    ds_stats = build_sft_and_dpo_datasets(corpus_dir=corpus_dir, output_dir=ds_dir)
    shutil.copy2(corpus_dir / "licencias_corpus.csv", ds_dir / "licencias_corpus.csv")
    if (corpus_dir / "CREDITOS_LICENCIAS.md").exists():
        shutil.copy2(corpus_dir / "CREDITOS_LICENCIAS.md", ds_dir / "CREDITOS_LICENCIAS.md")

    dataset_meta = {
        "title": "ROANSO Cuentos Chileno SFT y DPO Auditado",
        "id": f"{kaggle_username}/{safe_ds_slug}",
        "licenses": [{"name": "CC-BY-4.0"}],
        "isPrivate": True,
    }
    (ds_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    kernel_src = Path(__file__).resolve().parent / "kaggle_qlora_kernel.py"
    shutil.copy2(kernel_src, kn_dir / "kaggle_qlora_kernel.py")

    kernel_meta = {
        "id": f"{kaggle_username}/{safe_kn_slug}",
        "title": "ROANSO QLoRA Narrador Chileno Apache 2.0",
        "code_file": "kaggle_qlora_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [f"{kaggle_username}/{safe_ds_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kn_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "dataset_id": dataset_meta["id"],
        "kernel_id": kernel_meta["id"],
        "dataset_dir": str(ds_dir),
        "kernel_dir": str(kn_dir),
        "verified_works": len(verified),
        "sft_train_examples": ds_stats["sft_train_examples"],
        "sft_val_examples": ds_stats["sft_val_examples"],
        "dpo_preference_pairs": ds_stats["dpo_preference_pairs"],
    }


def stage_tts_kaggle_bundle(
    staging_root: Path,
    kaggle_username: str,
    private_voice_dataset_slug: str,
    kernel_slug: str = DEFAULT_TTS_KERNEL_SLUG,
) -> Dict[str, Any]:
    """Prepara el Kernel Privado de Kaggle para entrenar voces comerciales de actores chilenos."""
    safe_kn_slug = assert_safe_roanso_slug(kernel_slug)
    kn_dir = staging_root / "tts_kernel"
    if kn_dir.exists():
        shutil.rmtree(kn_dir)
    kn_dir.mkdir(parents=True, exist_ok=True)

    kernel_src = Path(__file__).resolve().parent / "kaggle_tts_voice_kernel.py"
    shutil.copy2(kernel_src, kn_dir / "kaggle_tts_voice_kernel.py")

    ds_ref = (
        private_voice_dataset_slug
        if "/" in private_voice_dataset_slug
        else f"{kaggle_username}/{private_voice_dataset_slug}"
    )
    kernel_meta = {
        "id": f"{kaggle_username}/{safe_kn_slug}",
        "title": "ROANSO Voces Actores Chile Entrenamiento Privado",
        "code_file": "kaggle_tts_voice_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [ds_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kn_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "kernel_id": kernel_meta["id"],
        "kernel_dir": str(kn_dir),
        "mounted_private_voice_dataset": ds_ref,
    }


def push_slm_to_kaggle(
    corpus_dir: Path,
    staging_root: Path,
    dataset_slug: str = DEFAULT_SLM_DATASET_SLUG,
    kernel_slug: str = DEFAULT_SLM_KERNEL_SLUG,
    kaggle_json_path: Path | None = None,
) -> Dict[str, Any]:
    """Sube (o versiona) el Dataset Privado ROANSO y lanza el entrenamiento GPU en Kaggle."""
    safe_ds_slug = assert_safe_roanso_slug(dataset_slug)
    safe_kn_slug = assert_safe_roanso_slug(kernel_slug)
    api, creds = get_authenticated_kaggle_api(kaggle_json_path=kaggle_json_path)

    staged = stage_slm_kaggle_bundle(
        corpus_dir=corpus_dir,
        staging_root=staging_root,
        kaggle_username=creds.username,
        dataset_slug=safe_ds_slug,
        kernel_slug=safe_kn_slug,
    )

    existing_ds_refs = {str(d.ref).lower() for d in api.dataset_list(mine=True)}
    target_ds_ref = staged["dataset_id"].lower()

    if target_ds_ref in existing_ds_refs:
        api.dataset_create_version(
            staged["dataset_dir"],
            version_notes="Actualizacion corpus auditado SHA-256 y pares DPO chilenos",
            quiet=True,
            convert_to_csv=False,
        )
        ds_action = "versioned"
    else:
        api.dataset_create_new(
            staged["dataset_dir"],
            public=False,
            quiet=True,
            convert_to_csv=False,
        )
        ds_action = "created_private"
        time.sleep(8)

    push_res = api.kernels_push(staged["kernel_dir"])
    return {
        "dataset_id": staged["dataset_id"],
        "dataset_action": ds_action,
        "kernel_id": staged["kernel_id"],
        "kernel_url": getattr(push_res, "url", f"https://www.kaggle.com/code/{staged['kernel_id']}"),
        "sft_train_examples": staged["sft_train_examples"],
        "dpo_preference_pairs": staged["dpo_preference_pairs"],
    }


def check_kernel_status(kernel_slug: str = DEFAULT_SLM_KERNEL_SLUG, kaggle_json_path: Path | None = None) -> Dict[str, Any]:
    safe_kn_slug = assert_safe_roanso_slug(kernel_slug)
    api, creds = get_authenticated_kaggle_api(kaggle_json_path=kaggle_json_path)
    full_ref = f"{creds.username}/{safe_kn_slug}"
    st = api.kernels_status(full_ref)
    return {
        "kernel_id": full_ref,
        "status": str(getattr(st, "status", st)),
        "failure_message": getattr(st, "failureMessage", None),
    }


def pull_kernel_output(
    output_dir: Path,
    kernel_slug: str = DEFAULT_SLM_KERNEL_SLUG,
    kaggle_json_path: Path | None = None,
) -> Dict[str, Any]:
    safe_kn_slug = assert_safe_roanso_slug(kernel_slug)
    api, creds = get_authenticated_kaggle_api(kaggle_json_path=kaggle_json_path)
    full_ref = f"{creds.username}/{safe_kn_slug}"
    output_dir.mkdir(parents=True, exist_ok=True)
    files, _token = api.kernels_output(full_ref, path=str(output_dir), quiet=True)
    return {
        "kernel_id": full_ref,
        "output_dir": str(output_dir),
        "downloaded_files": [str(f) for f in (files or [])],
    }


def main() -> None:
    default_corpus = PLATFORM_ROOT.parent / "corpus_limpio"
    default_staging = PLATFORM_ROOT / ".kaggle_staging"
    default_checkpoints = PLATFORM_ROOT / "checkpoints" / "kaggle_pulled"

    parser = argparse.ArgumentParser(description="ROANSO Kaggle GPU Training Orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep_slm = sub.add_parser("prepare-slm", help="Prepara localmente Dataset + Kernel GPU QLoRA para Kaggle")
    p_prep_slm.add_argument("--corpus-dir", default=str(default_corpus))
    p_prep_slm.add_argument("--staging-dir", default=str(default_staging))
    p_prep_slm.add_argument("--username", default=None)

    p_prep_tts = sub.add_parser("prepare-tts", help="Prepara localmente Kernel GPU Privado para voces de actores")
    p_prep_tts.add_argument("--staging-dir", default=str(default_staging))
    p_prep_tts.add_argument("--voice-dataset-slug", default="roanso-voces-actores-estudio-v1")
    p_prep_tts.add_argument("--username", default=None)

    p_push = sub.add_parser("push-slm", help="Sube Dataset Privado y lanza entrenamiento GPU QLoRA en Kaggle")
    p_push.add_argument("--corpus-dir", default=str(default_corpus))
    p_push.add_argument("--staging-dir", default=str(default_staging))

    p_stat = sub.add_parser("status", help="Consulta estado en vivo del entrenamiento GPU en Kaggle")
    p_stat.add_argument("--kernel-slug", default=DEFAULT_SLM_KERNEL_SLUG)

    p_pull = sub.add_parser("pull-output", help="Descarga el adaptador LoRA entrenado desde Kaggle")
    p_pull.add_argument("--kernel-slug", default=DEFAULT_SLM_KERNEL_SLUG)
    p_pull.add_argument("--output-dir", default=str(default_checkpoints))

    args = parser.parse_args()
    if args.cmd == "prepare-slm":
        user = args.username or load_kaggle_credentials().username
        res = stage_slm_kaggle_bundle(Path(args.corpus_dir), Path(args.staging_dir), kaggle_username=user)
    elif args.cmd == "prepare-tts":
        user = args.username or load_kaggle_credentials().username
        res = stage_tts_kaggle_bundle(Path(args.staging_dir), kaggle_username=user, private_voice_dataset_slug=args.voice_dataset_slug)
    elif args.cmd == "push-slm":
        res = push_slm_to_kaggle(Path(args.corpus_dir), Path(args.staging_dir))
    elif args.cmd == "status":
        res = check_kernel_status(args.kernel_slug)
    elif args.cmd == "pull-output":
        res = pull_kernel_output(Path(args.output_dir), args.kernel_slug)
    else:
        raise ValueError(f"Comando desconocido: {args.cmd}")

    # No imprime usernames reales en stdout salvo campos de conteo/estado seguros
    safe_print = {
        k: v
        for k, v in res.items()
        if k not in ("dataset_id", "kernel_id", "kernel_url", "mounted_private_voice_dataset")
    }
    safe_print["status_ok"] = True
    print(json.dumps(safe_print, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
