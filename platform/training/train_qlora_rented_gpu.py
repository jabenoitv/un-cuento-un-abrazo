# -*- coding: utf-8 -*-
"""Entrenador Reproducible QLoRA (4-bit NF4) para Servidor GPU Arrendado (`NVIDIA L4` / `RTX 4090` / `T4`).

Licencia de pesos resultantes: `Apache-2.0` (sobre `Qwen/Qwen2.5-7B-Instruct` o `Qwen/Qwen3-4B`).
Antes de tocar la GPU verifica criptográficamente `licencias_corpus.csv` (`SHA-256` + reglas §7.1/§11.3).

Ejecución en nodo GPU arrendado (RunPod / Modal / Hetzner GPU):
  python training/train_qlora_rented_gpu.py \
      --base-model Qwen/Qwen2.5-7B-Instruct \
      --corpus-dir ../corpus_limpio \
      --dataset-dir ./training_data \
      --output-dir ./checkpoints/roanso-chileno-qwen7b-lora-v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from roanso_platform.domain_bridge import MAX_PARAM_BILLIONS  # noqa: E402
from roanso_platform.lora_kit import (  # noqa: E402
    build_sft_and_dpo_datasets,
    verify_corpus_integrity_against_ledger,
)

APPROVED_APACHE_BASE_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": 7.6,
    "Qwen/Qwen3-4B": 4.0,
    "Qwen/Qwen3-8B": 8.2,
    "BSC-LT/salamandra-7b-instruct": 7.0,
}


def run_qlora_training(args: argparse.Namespace) -> None:
    if args.base_model not in APPROVED_APACHE_BASE_MODELS:
        raise PermissionError(
            f"Modelo base '{args.base_model}' no está en la lista Apache-2.0 auditada "
            f"(CORPUS_CUENTOS_LIBRES.md §10): {sorted(APPROVED_APACHE_BASE_MODELS)}"
        )
    params_b = APPROVED_APACHE_BASE_MODELS[args.base_model]
    if params_b > MAX_PARAM_BILLIONS:
        raise ValueError(f"Excede techo Human-Scale AI ({MAX_PARAM_BILLIONS}B).")

    corpus_dir = Path(args.corpus_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    verified = verify_corpus_integrity_against_ledger(corpus_dir)
    ds_stats = build_sft_and_dpo_datasets(corpus_dir=corpus_dir, output_dir=dataset_dir)
    print(f"[QLoRA-AUDIT] Ledger verificado ({len(verified)} obras SHA-256 OK): {json.dumps(ds_stats)}")

    if args.verify_only:
        return

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
    )

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
        data_files={
            "train": str(dataset_dir / "sft_train_chileno.jsonl"),
            "validation": str(dataset_dir / "sft_val_chileno.jsonl"),
        },
    )

    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.08,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        max_seq_length=1536,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft_cfg,
        args=sft_cfg,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    (output_dir / "provenance_audit.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "license": "Apache-2.0",
                "params_billions": params_b,
                "human_corpus_works": len(verified),
                "dataset_stats": ds_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[QLoRA-DONE] Adaptador Apache-2.0 guardado en {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ROANSO Rented-GPU QLoRA Fine-Tuning")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--corpus-dir", default="../corpus_limpio")
    parser.add_argument("--dataset-dir", default="./training_data")
    parser.add_argument("--output-dir", default="./checkpoints/roanso-chileno-qwen7b-lora-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--verify-only", action="store_true")
    run_qlora_training(parser.parse_args())


if __name__ == "__main__":
    main()
