# -*- coding: utf-8 -*-
"""Kernel Autónomo de Entrenamiento QLoRA + DPO para Kaggle Notebooks (`GPU T4 x2` / `P100`).

Diseñado específicamente para correr sin costo dentro de la cuota gratuita de **Kaggle GPU**
(`2x NVIDIA Tesla T4 16 GB = 32 GB VRAM`, arquitectura Turing `sm_75`):
1. Instala automáticamente versiones compatibles de `bitsandbytes`, `peft`, `trl` y `accelerate` si corre en `/kaggle/working`.
2. Descubre el dataset privado montado en `/kaggle/input/roanso-cuentos-chileno-sft-v1` y audita `licencias_corpus.csv`
   (bloqueando licencias no comerciales/virales o corpus sintéticos antes de inicializar CUDA).
3. Fija `device_map={"": 0}` y `fp16=True` en GPUs Turing (`sm_75`) para evitar el choque clásico de tensores multi-GPU
   (`cuda:0` vs `cuda:1`) en `SFTTrainer` / `DPOTrainer` sobre instancias `T4 x2`.
4. Ejecuta Etapa 1 (`SFT` sobre pasajes literarios humanos) + Etapa 2 (`DPO` sobre preferencias chilenas y Diseño Universal)
   + prueba de humo nocturna y empaqueta `/kaggle/working/roanso-chileno-lora-v1.zip` (`Apache-2.0`).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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


def ensure_kaggle_ml_dependencies() -> None:
    """Instala dependencias QLoRA probadas cuando el script corre dentro de una VM GPU de Kaggle."""
    if not is_kaggle_runtime():
        return
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    pkgs = [
        "bitsandbytes>=0.44.0",
        "peft>=0.13.0",
        "trl>=0.12.0",
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs], check=True)


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

    ensure_kaggle_ml_dependencies()
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

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_cfg,
        device_map=gpu_profile["device_map"],
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

    sft_cfg = SFTConfig(
        output_dir=str(sft_stage_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.08,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        max_seq_length=1536,
        bf16=gpu_profile["bf16"],
        fp16=gpu_profile["fp16"],
        gradient_checkpointing=True,
        optim=gpu_profile["optim"],
        report_to="none",
    )

    sft_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft_cfg,
        args=sft_cfg,
    )
    sft_result = sft_trainer.train()
    sft_trainer.save_model(str(sft_stage_dir))
    tokenizer.save_pretrained(str(sft_stage_dir))

    dpo_metrics: Dict[str, Any] = {"executed": False}
    if args.run_dpo and dpo_count > 0:
        print(f"[KAGGLE-DPO] Refinando {dpo_count} pares chilenos de Diseño Universal...")
        dpo_ds = load_dataset("json", data_files={"train": str(dpo_jsonl)})["train"]
        base_for_dpo = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_cfg,
            device_map=gpu_profile["device_map"],
        )
        base_for_dpo.config.use_cache = False
        policy_model = PeftModel.from_pretrained(base_for_dpo, str(sft_stage_dir), is_trainable=True)
        dpo_cfg = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            beta=0.1,
            bf16=gpu_profile["bf16"],
            fp16=gpu_profile["fp16"],
            gradient_checkpointing=True,
            optim=gpu_profile["optim"],
            logging_steps=2,
            save_strategy="no",
            report_to="none",
        )
        dpo_trainer = DPOTrainer(
            model=policy_model,
            ref_model=None,
            args=dpo_cfg,
            train_dataset=dpo_ds,
            processing_class=tokenizer,
        )
        dpo_res = dpo_trainer.train()
        dpo_trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        dpo_metrics = {"executed": True, "train_loss": float(dpo_res.training_loss)}

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
    rendered = tokenizer.apply_chat_template(smoke_prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to("cuda:0")
    eval_model = sft_trainer.model
    eval_model.eval()
    with torch.no_grad():
        gen_ids = eval_model.generate(**inputs, max_new_tokens=180, temperature=0.55, do_sample=True)
    sample_story = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    elapsed_s = round(time.time() - t0, 2)
    audit_payload = {
        **preflight_summary,
        "training_platform": "Kaggle Notebooks GPU",
        "gpu_name": gpu_name,
        "gpu_profile": gpu_profile,
        "sft_train_loss": float(sft_result.training_loss),
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
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
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
