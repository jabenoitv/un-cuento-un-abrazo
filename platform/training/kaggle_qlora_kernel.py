# -*- coding: utf-8 -*-
"""Kernel Autónomo 100% Offline QLoRA + DPO Nativo (PyTorch + PEFT) para Kaggle GPU (`T4 x2`).

Diseñado para máxima robustez sobre la imagen 2026 de Kaggle (`Python 3.13` + `transformers 5.x` + `CUDA 12.8/13.x`):
1. Instala únicamente `bitsandbytes>=0.50.2` (`cuda118`..`cuda132`) desde el wheel offline incluido en
   `/kaggle/input/roanso-corpus-chileno-sft` (con respaldo automático a `fp16` `device_map="auto"` si no hay bnb).
2. No depende de `trl` (evitando incompatibilidades con `transformers 5.x`): ejecuta un bucle **SFT con máscara
   exacta de tokens de asistente** + **DPO exacto (Rafailov et al.) usando `model.disable_adapter()`** sin duplicar VRAM.
3. Carga `Qwen2.5-7B-Instruct` o `Qwen3-4B` (`Apache-2.0`) desde el montaje NVMe interno de Kaggle (`model_sources`),
   corre prueba de humo de narración chilena nocturna y empaqueta `/kaggle/working/roanso-chileno-lora-v1.zip`.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
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


def ensure_kaggle_ml_dependencies(dataset_dir: Path) -> bool:
    """Instala `bitsandbytes` moderno (`>=0.50`) desde wheel local si está disponible; retorna si bnb quedó activo."""
    if not is_kaggle_runtime():
        return importlib.util.find_spec("bitsandbytes") is not None

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    search_dirs = [dataset_dir, dataset_dir / "wheels"]
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.exists():
        for whl in kaggle_in.rglob("bitsandbytes-*.whl"):
            if whl.parent not in search_dirs:
                search_dirs.append(whl.parent)

    for wd in search_dirs:
        if wd.exists():
            bnb_wheels = sorted(wd.glob("bitsandbytes-*.whl"))
            if bnb_wheels:
                target_whl = str(bnb_wheels[-1])
                print(f"[KAGGLE-DEPS] Instalando wheel offline CUDA moderno: {target_whl}")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--force-reinstall", target_whl],
                    check=True,
                )
                return True

    return importlib.util.find_spec("bitsandbytes") is not None


def locate_mounted_dataset(explicit_dir: str | None = None) -> Path:
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


def _apply_chat(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool, base_model: str) -> str:
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if "Qwen3" in base_model:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def _encode_sft_record(tokenizer: Any, messages: List[Dict[str, str]], max_len: int, base_model: str) -> Dict[str, List[int]]:
    prompt_msgs = messages[:-1]
    full_text = _apply_chat(tokenizer, messages, add_generation_prompt=False, base_model=base_model)
    prompt_text = _apply_chat(tokenizer, prompt_msgs, add_generation_prompt=True, base_model=base_model)

    full_ids = tokenizer(full_text, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]

    prompt_len = min(len(prompt_ids), max(0, len(full_ids) - 1))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return {"input_ids": full_ids, "labels": labels}


def _sequence_logprob(model: Any, input_ids_1d: List[int], prompt_len: int, device: Any) -> Any:
    import torch
    import torch.nn.functional as F

    x = torch.tensor([input_ids_1d], dtype=torch.long, device=device)
    logits = model(input_ids=x).logits[:, :-1, :]
    targets = x[:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_lp = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    start_idx = max(0, prompt_len - 1)
    return token_lp[:, start_idx:].sum(dim=-1).squeeze(0)


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

    train_rows = [json.loads(ln) for ln in train_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    val_rows = [json.loads(ln) for ln in val_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dpo_rows = (
        [json.loads(ln) for ln in dpo_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if dpo_jsonl.exists()
        else []
    )

    preflight_summary = {
        "base_model": args.base_model,
        "license": "Apache-2.0",
        "params_billions": params_b,
        "verified_ledger_works": len(verified_rows),
        "sft_train_rows": len(train_rows),
        "sft_val_rows": len(val_rows),
        "dpo_rows": len(dpo_rows),
        "dataset_dir": str(dataset_dir),
    }
    print(f"[KAGGLE-PREFLIGHT-OK] {json.dumps(preflight_summary, ensure_ascii=False)}")
    if args.verify_only:
        return preflight_summary

    has_bnb = ensure_kaggle_ml_dependencies(dataset_dir)
    resolved_model_path, is_offline_mount = resolve_base_model_path_or_id(args.base_model)
    print(f"[KAGGLE-MODEL] Fuente pesos='{resolved_model_path}' (offline_nvme={is_offline_mount}, bnb={has_bnb})")
    t0 = time.time()

    import random
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    has_cuda = torch.cuda.is_available()
    major_cc = torch.cuda.get_device_capability(0)[0] if has_cuda else 0
    gpu_profile = compute_kaggle_gpu_profile(
        has_cuda=has_cuda,
        major_cc=major_cc,
        device_count=torch.cuda.device_count() if has_cuda else 0,
    )
    gpu_name = torch.cuda.get_device_name(0)
    compute_dtype = torch.bfloat16 if gpu_profile["bf16"] else torch.float16
    print(f"[KAGGLE-GPU] Detectada '{gpu_name}' (GPUs={gpu_profile['gpu_count']}, perfil={gpu_profile})")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, local_files_only=is_offline_mount)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_mode = "fp16_sharded"
    model = None
    if has_bnb:
        try:
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_path,
                quantization_config=bnb_cfg,
                device_map=gpu_profile["device_map"],
                local_files_only=is_offline_mount,
            )
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            quant_mode = "qlora_nf4_4bit"
        except Exception as bnb_err:
            print(f"[KAGGLE-WARN] 4-bit NF4 no disponible ({bnb_err}); activando respaldo nativo fp16.")
            model = None

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_path,
            torch_dtype=compute_dtype,
            device_map="auto",
            local_files_only=is_offline_mount,
        )
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    encoded_train = [_encode_sft_record(tokenizer, r["messages"], 1024, args.base_model) for r in train_rows]
    encoded_val = [_encode_sft_record(tokenizer, r["messages"], 1024, args.base_model) for r in val_rows]

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=0.01)

    total_micro_steps = args.epochs * len(encoded_train)
    total_opt_steps = max(1, math.ceil(total_micro_steps / args.grad_accum))
    warmup_steps = max(2, int(0.08 * total_opt_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=gpu_profile["fp16"])
    first_device = next(model.parameters()).device
    amp_dtype = torch.bfloat16 if gpu_profile["bf16"] else torch.float16

    rng = random.Random(42)
    running_losses: List[float] = []
    opt_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        indices = list(range(len(encoded_train)))
        rng.shuffle(indices)
        epoch_losses: List[float] = []
        for idx_pos, sample_idx in enumerate(indices, start=1):
            rec = encoded_train[sample_idx]
            inp = torch.tensor([rec["input_ids"]], dtype=torch.long, device=first_device)
            lbl = torch.tensor([rec["labels"]], dtype=torch.long, device=first_device)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out = model(input_ids=inp, labels=lbl)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()
            raw_loss = float(out.loss.detach().cpu())
            running_losses.append(raw_loss)
            epoch_losses.append(raw_loss)

            if idx_pos % args.grad_accum == 0 or idx_pos == len(indices):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                opt_step += 1

        # Evaluación en validación al cierre de cada época
        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for vrec in encoded_val:
                vinp = torch.tensor([vrec["input_ids"]], dtype=torch.long, device=first_device)
                vlbl = torch.tensor([vrec["labels"]], dtype=torch.long, device=first_device)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    vloss = model(input_ids=vinp, labels=vlbl).loss
                val_losses.append(float(vloss.detach().cpu()))
        mean_train = round(sum(epoch_losses) / max(1, len(epoch_losses)), 4)
        mean_val = round(sum(val_losses) / max(1, len(val_losses)), 4)
        print(f"[KAGGLE-SFT] Época {epoch + 1}/{args.epochs} · train_loss={mean_train} · val_loss={mean_val}")
        model.train()

    sft_loss = round(sum(running_losses) / max(1, len(running_losses)), 4)
    val_final_loss = mean_val

    # Etapa 2: DPO exacto usando `model.disable_adapter()` como referencia congelada sin duplicar VRAM
    dpo_metrics: Dict[str, Any] = {"executed": False}
    if args.run_dpo and dpo_rows:
        print(f"[KAGGLE-DPO] Refinando {len(dpo_rows)} pares chilenos de Diseño Universal...")
        dpo_opt = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.0)
        dpo_losses: List[float] = []
        sys_dpo = (
            "Eres un narrador chileno cálido de 'Un Cuento, Un Abrazo'. Escribes en 3ª persona, "
            "sin etiquetas clínicas ni moralejas, mostrando emociones e intenciones reales."
        )
        for _dpo_epoch in range(3):
            for pair in dpo_rows:
                p_msgs = [{"role": "system", "content": sys_dpo}, {"role": "user", "content": pair["prompt"]}]
                c_msgs = p_msgs + [{"role": "assistant", "content": pair["chosen"]}]
                r_msgs = p_msgs + [{"role": "assistant", "content": pair["rejected"]}]

                p_ids = tokenizer(
                    _apply_chat(tokenizer, p_msgs, True, args.base_model),
                    truncation=True,
                    max_length=384,
                    add_special_tokens=False,
                )["input_ids"]
                c_ids = tokenizer(
                    _apply_chat(tokenizer, c_msgs, False, args.base_model),
                    truncation=True,
                    max_length=768,
                    add_special_tokens=False,
                )["input_ids"]
                r_ids = tokenizer(
                    _apply_chat(tokenizer, r_msgs, False, args.base_model),
                    truncation=True,
                    max_length=768,
                    add_special_tokens=False,
                )["input_ids"]
                plen = len(p_ids)

                with torch.no_grad(), model.disable_adapter():
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        ref_c = _sequence_logprob(model, c_ids, plen, first_device)
                        ref_r = _sequence_logprob(model, r_ids, plen, first_device)

                dpo_opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    pi_c = _sequence_logprob(model, c_ids, plen, first_device)
                    pi_r = _sequence_logprob(model, r_ids, plen, first_device)
                    logits_diff = (pi_c - ref_c) - (pi_r - ref_r)
                    dpo_loss = -F.logsigmoid(0.1 * logits_diff)

                scaler.scale(dpo_loss).backward()
                scaler.unscale_(dpo_opt)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(dpo_opt)
                scaler.update()
                dpo_losses.append(float(dpo_loss.detach().cpu()))

        dpo_metrics = {
            "executed": True,
            "dpo_epochs": 3,
            "mean_dpo_loss": round(sum(dpo_losses) / max(1, len(dpo_losses)), 4),
        }
        print(f"[KAGGLE-DPO-OK] {dpo_metrics}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    adapter_cfg_path = output_dir / "adapter_config.json"
    if adapter_cfg_path.exists():
        acfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        acfg["base_model_name_or_path"] = args.base_model
        adapter_cfg_path.write_text(json.dumps(acfg, indent=2), encoding="utf-8")

    modelfile_src = dataset_dir / "Modelfile.roanso-chileno"
    if modelfile_src.exists():
        shutil.copy2(modelfile_src, output_dir / "Modelfile.roanso-chileno")

    # Prueba de humo de generación cálida chilena en inferencia real
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
    rendered = _apply_chat(tokenizer, smoke_prompt, add_generation_prompt=True, base_model=args.base_model)
    inputs = tokenizer(rendered, return_tensors="pt").to(first_device)
    model.eval()
    model.config.use_cache = True
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
        gen_ids = model.generate(**inputs, max_new_tokens=180, temperature=0.55, do_sample=True)
    sample_story = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    gc.collect()
    torch.cuda.empty_cache()

    elapsed_s = round(time.time() - t0, 2)
    audit_payload = {
        **preflight_summary,
        "training_platform": "Kaggle Notebooks GPU (Offline Air-Gapped Native PyTorch+PEFT)",
        "quantization_mode": quant_mode,
        "mounted_model_path": resolved_model_path,
        "gpu_name": gpu_name,
        "gpu_profile": gpu_profile,
        "sft_train_loss": sft_loss,
        "sft_val_loss": val_final_loss,
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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--run-dpo", action="store_true", default=True)
    parser.add_argument("--no-dpo", action="store_false", dest="run_dpo")
    parser.add_argument("--verify-only", action="store_true")
    run_kaggle_qlora_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
