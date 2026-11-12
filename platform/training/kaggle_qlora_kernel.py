# -*- coding: utf-8 -*-
"""Kernel Autónomo 100% Offline QLoRA + DPO para Kaggle Notebooks (`GPU T4 x2` / `P100`).

Opera de extremo a extremo **sin requerir conexión a Internet** dentro del contenedor GPU de Kaggle:
1. Instala `bitsandbytes`, `peft` y `trl` desde wheels `.whl` locales incluidos en el dataset privado
   `/kaggle/input/roanso-corpus-chileno-sft` solo si no vienen preinstalados en la imagen base.
2. Carga los pesos base `Apache-2.0` (`Qwen2.5-7B-Instruct` o `Qwen3-4B`) directamente desde el montaje
   NVMe interno de **Kaggle Models Hub** (`model_sources`), en ~8 segundos y sin llamadas de red.
3. Fija `device_map={"": 0}` y `fp16=True` en GPUs Turing (`sm_75`) y libera VRAM entre `SFT` y `DPO`.
4. Normaliza `adapter_config.json` al ID canónico de Hugging Face y exporta `roanso-chileno-lora-v1.zip`.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_BASE_MODEL = os.environ.get("ROANSO_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

APPROVED_APACHE_BASE_MODELS: Dict[str, float] = {
    "Qwen/Qwen2.5-7B-Instruct": 7.6,
    "Qwen/Qwen3-4B": 4.0,
    "Qwen/Qwen3-8B": 8.2,
    "BSC-LT/salamandra-7b-instruct": 7.0,
}
MAX_HUMAN_SCALE_BILLIONS: float = 10.0
ALLOWED_COMMERCIAL_SPDX = {"PDM-1.0", "CC0-1.0", "CC-BY-4.0"}


def is_kaggle_runtime() -> bool:
    return Path("/kaggle/working").exists() or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def ensure_kaggle_ml_dependencies(dataset_dir: Path) -> None:
    """Instala paquetes faltantes desde wheels locales offline montados en `/kaggle/input`."""
    if not is_kaggle_runtime():
        return
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    missing = [pkg for pkg in ("bitsandbytes", "peft", "trl") if importlib.util.find_spec(pkg) is None]
    if not missing:
        print("[KAGGLE-DEPS] bitsandbytes, peft y trl ya están preinstalados en la imagen GPU.")
        return

    wheel_dirs = [dataset_dir, dataset_dir / "wheels"]
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.exists():
        for whl in kaggle_in.rglob("*.whl"):
            if whl.parent not in wheel_dirs:
                wheel_dirs.append(whl.parent)

    local_wheels: List[str] = []
    for pkg in missing:
        matched = None
        for wd in wheel_dirs:
            if wd.exists():
                found = sorted(wd.glob(f"{pkg}-*.whl"))
                if found:
                    matched = str(found[-1])
                    break
        if matched:
            local_wheels.append(matched)

    if len(local_wheels) == len(missing):
        print(f"[KAGGLE-DEPS] Instalando wheels offline para {missing}: {local_wheels}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *local_wheels],
            check=True,
        )
        return

    raise FileNotFoundError(
        f"Faltan paquetes {missing} y no se hallaron sus wheels `.whl` offline en {dataset_dir}."
    )


def locate_mounted_dataset(explicit_dir: str | None = None) -> Path:
    """Localiza el dataset ROANSO ya sea por ruta explícita o escaneando `/kaggle/input/*`."""
    if explicit_dir:
        candidate = Path(explicit_dir).resolve()
        if (candidate / "sft_train_chileno.jsonl").exists():
            return candidate
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for found in kaggle_input.rglob("sft_train_chileno.jsonl"):
            return found.parent
    raise FileNotFoundError(
        "No se encontró `sft_train_chileno.jsonl` en `--dataset-dir` ni bajo `/kaggle/input/`."
    )


def resolve_base_model_path_or_id(base_model: str) -> tuple[str, bool]:
    """Resuelve montaje NVMe local en `/kaggle/input` (`model_sources`) para operar 100% sin red."""
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return base_model, False

    candidates: List[Path] = []
    for cfg in kaggle_input.rglob("config.json"):
        parent = cfg.parent
        if (parent / "tokenizer_config.json").exists():
            candidates.append(parent)

    token_hint = "7b-instruct" if "7B-Instruct" in base_model else ("4b" if "4B" in base_model else "")
    if token_hint:
        for c in candidates:
            if token_hint in str(c).lower():
                return str(c), True
    if len(candidates) == 1:
        return str(candidates[0]), True
    return base_model, False


def verify_mounted_ledger_governance(dataset_dir: Path) -> List[Dict[str, str]]:
    """Valida reglas legales comerciales y humanas en `licencias_corpus.csv` dentro del nodo GPU."""
    ledger_path = dataset_dir / "licencias_corpus.csv"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Falta `licencias_corpus.csv` obligatorio en {dataset_dir}")

    with ledger_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("`licencias_corpus.csv` está vacío.")

    for row in rows:
        spdx = (row.get("licencia_spdx") or "").strip()
        if spdx not in ALLOWED_COMMERCIAL_SPDX:
            raise PermissionError(
                f"Obra {row.get('id')} tiene licencia '{spdx}' no apta para entrenamiento comercial abierto."
            )
        if (row.get("sintetico") or "").strip().lower() == "si":
            raise PermissionError(f"Obra {row.get('id')} marcada como sintética; prohibida en ROANSO.")
    return rows


def compute_kaggle_gpu_profile(has_cuda: bool, major_cc: int, device_count: int) -> Dict[str, Any]:
    """Calcula perfil óptimo para Tesla T4 (`sm_75`, sin `bf16` nativo) vs Ampere+ (`sm_80+`)."""
    if not has_cuda:
        raise RuntimeError(
            "No se detectó GPU CUDA activa. En Kaggle activa 'Accelerator: GPU T4 x2' o 'GPU P100'."
        )
    use_bf16 = major_cc >= 8
    return {
        "device_map": {"": 0},
        "secondary_eval_device": "cuda:1" if device_count >= 2 else "cuda:0",
        "gpu_count": device_count,
        "compute_capability_major": major_cc,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
    }


def _filter_supported_kwargs(cls: Any, desired: Dict[str, Any]) -> Dict[str, Any]:
    sig_params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()):
        return desired
    return {k: v for k, v in desired.items() if k in sig_params}


def run_kaggle_qlora_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    if args.base_model not in APPROVED_APACHE_BASE_MODELS:
        raise PermissionError(
            f"Modelo '{args.base_model}' fuera de la lista Apache-2.0 aprobada: {sorted(APPROVED_APACHE_BASE_MODELS)}"
        )
    params_b = APPROVED_APACHE_BASE_MODELS[args.base_model]
    if params_b > MAX_HUMAN_SCALE_BILLIONS:
        raise ValueError(f"Modelo excede límite Human-Scale AI ({MAX_HUMAN_SCALE_BILLIONS}B).")

    dataset_dir = locate_mounted_dataset(args.dataset_dir)
    verified_rows = verify_mounted_ledger_governance(dataset_dir)
    train_jsonl = dataset_dir / "sft_train_chileno.jsonl"
    val_jsonl = dataset_dir / "sft_val_chileno.jsonl"
    dpo_jsonl = dataset_dir / "dpo_preferencias_chilenas.jsonl"

    train_count = sum(1 for ln in train_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip())
    val_count = sum(1 for ln in val_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip())
    dpo_count = (
        sum(1 for ln in dpo_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip())
        if dpo_jsonl.exists()
        else 0
    )

    preflight_summary = {
        "base_model": args.base_model,
        "license": "Apache-2.0",
        "params_billions": params_b,
        "verified_ledger_works": len(verified_rows),
        "sft_train_rows": train_count,
        "sft_val_rows": val_count,
        "dpo_rows": dpo_count,
        "dataset_dir": str(dataset_dir),
    }
    print(f"[KAGGLE-PREFLIGHT-OK] {json.dumps(preflight_summary, ensure_ascii=False)}")
    if args.verify_only:
        return preflight_summary

    ensure_kaggle_ml_dependencies(dataset_dir)
    resolved_model_path, is_offline_mount = resolve_base_model_path_or_id(args.base_model)
    print(f"[KAGGLE-MODEL] Fuente pesos='{resolved_model_path}' (offline_nvme={is_offline_mount})")
    t0 = time.time()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    has_cuda = torch.cuda.is_available()
    major_cc = torch.cuda.get_device_capability(0)[0] if has_cuda else 0
    gpu_profile = compute_kaggle_gpu_profile(
        has_cuda=has_cuda,
        major_cc=major_cc,
        device_count=torch.cuda.device_count() if has_cuda else 0,
    )
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[KAGGLE-GPU] Detectada '{gpu_name}' (GPUs={gpu_profile['gpu_count']}, perfil={gpu_profile})")

    compute_dtype = torch.bfloat16 if gpu_profile["bf16"] else torch.float16
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, local_files_only=is_offline_mount)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        quantization_config=bnb_cfg,
        device_map=gpu_profile["device_map"],
        local_files_only=is_offline_mount,
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    ds = load_dataset(
        "json",
        data_files={"train": str(train_jsonl), "validation": str(val_jsonl)},
    )

    output_dir = Path(args.output_dir).resolve()
    sft_stage_dir = output_dir / "sft_checkpoint" if (args.run_dpo and dpo_count > 0) else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_sig = inspect.signature(SFTConfig.__init__).parameters
    sft_raw_kwargs: Dict[str, Any] = {
        "output_dir": str(sft_stage_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "bf16": gpu_profile["bf16"],
        "fp16": gpu_profile["fp16"],
        "gradient_checkpointing": True,
        "optim": gpu_profile["optim"],
        "report_to": "none",
    }
    if "max_length" in sft_sig:
        sft_raw_kwargs["max_length"] = 1280
    elif "max_seq_length" in sft_sig:
        sft_raw_kwargs["max_seq_length"] = 1280
    if "eval_strategy" in sft_sig:
        sft_raw_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sft_sig:
        sft_raw_kwargs["evaluation_strategy"] = "epoch"

    sft_cfg = SFTConfig(**_filter_supported_kwargs(SFTConfig, sft_raw_kwargs))

    sft_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft_cfg,
        args=sft_cfg,
    )
    sft_result = sft_trainer.train()
    sft_loss = float(sft_result.training_loss)
    sft_trainer.save_model(str(sft_stage_dir))
    tokenizer.save_pretrained(str(sft_stage_dir))

    # Libera estados del optimizador SFT antes de iniciar DPO / inferencia en T4 16GB
    del sft_trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    dpo_metrics: Dict[str, Any] = {"executed": False}
    base_post = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        quantization_config=bnb_cfg,
        device_map=gpu_profile["device_map"],
        local_files_only=is_offline_mount,
    )
    base_post.config.use_cache = False
    active_model = PeftModel.from_pretrained(
        base_post,
        str(sft_stage_dir),
        is_trainable=bool(args.run_dpo and dpo_count > 0),
    )

    if args.run_dpo and dpo_count > 0:
        print(f"[KAGGLE-DPO] Refinando {dpo_count} pares chilenos de Diseño Universal...")
        dpo_ds = load_dataset("json", data_files={"train": str(dpo_jsonl)})["train"]
        dpo_raw_kwargs: Dict[str, Any] = {
            "output_dir": str(output_dir),
            "num_train_epochs": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "beta": 0.1,
            "max_length": 768,
            "max_prompt_length": 256,
            "bf16": gpu_profile["bf16"],
            "fp16": gpu_profile["fp16"],
            "gradient_checkpointing": True,
            "optim": gpu_profile["optim"],
            "logging_steps": 2,
            "save_strategy": "no",
            "report_to": "none",
        }
        dpo_cfg = DPOConfig(**_filter_supported_kwargs(DPOConfig, dpo_raw_kwargs))
        dpo_trainer = DPOTrainer(
            model=active_model,
            ref_model=None,
            args=dpo_cfg,
            train_dataset=dpo_ds,
            processing_class=tokenizer,
        )
        dpo_res = dpo_trainer.train()
        dpo_trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        dpo_metrics = {"executed": True, "train_loss": float(dpo_res.training_loss)}
        active_model = dpo_trainer.model

    # Normaliza `base_model_name_or_path` en `adapter_config.json` al ID canónico de Hugging Face
    adapter_cfg_path = output_dir / "adapter_config.json"
    if adapter_cfg_path.exists():
        acfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        acfg["base_model_name_or_path"] = args.base_model
        adapter_cfg_path.write_text(json.dumps(acfg, indent=2), encoding="utf-8")

    modelfile_src = dataset_dir / "Modelfile.roanso-chileno"
    if modelfile_src.exists():
        shutil.copy2(modelfile_src, output_dir / "Modelfile.roanso-chileno")

    # Prueba de humo de generación cálida chilena en inferencia
    smoke_prompt = [
        {
            "role": "system",
            "content": (
                "Eres un narrador chileno cálido de 'Un Cuento, Un Abrazo'. Escribes en 3ª persona, "
                "sin etiquetas clínicas ni moralejas, mostrando emociones e intenciones reales."
            ),
        },
        {
            "role": "user",
            "content": (
                "Escribe un pasaje nocturno breve (80 palabras) sobre Martín y su amiga Sofía "
                "compartiendo turnos con barquitos de papel junto al estero antes de dormir."
            ),
        },
    ]
    template_kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if "Qwen3" in args.base_model:
        template_kwargs["enable_thinking"] = False
    try:
        rendered = tokenizer.apply_chat_template(smoke_prompt, **template_kwargs)
    except TypeError:
        rendered = tokenizer.apply_chat_template(smoke_prompt, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(rendered, return_tensors="pt").to("cuda:0")
    active_model.eval()
    with torch.no_grad():
        gen_ids = active_model.generate(**inputs, max_new_tokens=180, temperature=0.55, do_sample=True)
    sample_story = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    if sft_stage_dir != output_dir and sft_stage_dir.exists():
        shutil.rmtree(sft_stage_dir, ignore_errors=True)

    elapsed_s = round(time.time() - t0, 2)
    audit_payload = {
        **preflight_summary,
        "training_platform": "Kaggle Notebooks GPU (Offline Air-Gapped)",
        "mounted_model_path": resolved_model_path,
        "gpu_name": gpu_name,
        "gpu_profile": gpu_profile,
        "sft_train_loss": sft_loss,
        "dpo_stage": dpo_metrics,
        "elapsed_seconds": elapsed_s,
        "smoke_story_preview": sample_story,
    }
    (output_dir / "provenance_audit.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    zip_base = output_dir.parent / output_dir.name
    archive_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(output_dir))
    print(f"[KAGGLE-QLORA-COMPLETE] Artefacto listo en {archive_path} ({elapsed_s}s)")
    return audit_payload


def main() -> None:
    default_out = (
        "/kaggle/working/roanso-chileno-lora-v1"
        if is_kaggle_runtime()
        else "./checkpoints/roanso-chileno-lora-v1"
    )
    parser = argparse.ArgumentParser(description="ROANSO Kaggle GPU QLoRA + DPO Trainer")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-dir", default=default_out)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--run-dpo", action="store_true", default=True)
    parser.add_argument("--no-dpo", action="store_false", dest="run_dpo")
    parser.add_argument("--verify-only", action="store_true")
    run_kaggle_qlora_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
