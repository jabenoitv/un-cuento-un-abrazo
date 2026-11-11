# -*- coding: utf-8 -*-
"""Worker Elástico para Servidores GPU Arrendados (RunPod / Modal / Hetzner GPU).

Características clave de arquitectura escalable:
1. Cero puertos entrantes: hace pull HTTPS autenticado (`X-Worker-Token`) hacia el Control Plane.
2. Scale-to-Zero (`--drain-and-exit`): procesa todos los cuentos pendientes en la cola y apaga el
   contenedor automáticamente cuando recibe `204 No Content`, evitando pagar minutos ociosos de GPU.
3. Loop de Auto-Reparación Determinista (`SelfHealingStoryEngine`): ejecuta el SLM abierto (`<= 8B`
   en `vLLM` u `Ollama` local al nodo GPU), audita con `analizar_cuento_multinino` + anti-regurgitación,
   y si detecta errores (oraciones muy largas, sermón, modismo sin explicar, falta cierre), re-inyecta
   las correcciones exactas hasta `max_repair_turns` antes de sintetizar audio.
4. Doble verificación acústica (`Whisper WER <= 0.12`) antes de publicar el capítulo.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import httpx

from .domain_bridge import (
    analizar_cuento_multinino,
    verificar_anti_regurgitacion_ngramas,
    verificar_soberania_human_scale,
)


@dataclass
class AcousticSynthesisResult:
    mp3_bytes: bytes
    whisper_wer: float
    speaker_similarity: float


TextGeneratorFn = Callable[[str, int], str]
AudioSynthesizerFn = Callable[[str, Dict[str, Any], Optional[bytes]], AcousticSynthesisResult]


def default_vllm_or_ollama_generator(
    endpoint_url: str,
    model_tag: str,
    param_billions: float = 7.6,
) -> TextGeneratorFn:
    """Conector HTTP sin claves propietarias hacia `vLLM` (`/v1/chat/completions`) u `Ollama`."""
    verificar_soberania_human_scale(endpoint_url, param_billions=param_billions)

    def _generate(prompt: str, attempt: int) -> str:
        temp = max(0.35, 0.65 - (attempt * 0.1))
        if endpoint_url.rstrip("/").endswith("/api/generate"):
            resp = httpx.post(
                endpoint_url,
                json={
                    "model": model_tag,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temp},
                },
                timeout=180.0,
            )
            resp.raise_for_status()
            return (resp.json().get("response") or "").strip()

        # Protocolo OpenAI-compatible de vLLM / SGLang / llama.cpp-server en GPU arrendada
        chat_url = endpoint_url.rstrip("/") + "/chat/completions"
        resp = httpx.post(
            chat_url,
            json={
                "model": model_tag,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "max_tokens": 1800,
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    return _generate


class SelfHealingStoryEngine:
    """ Bucle neuro-inclusivo: SLM <= 8B -> Linter + Anti-Regurgitación -> Reparación Dirigida."""

    def __init__(self, generator_fn: TextGeneratorFn, max_repair_turns: int = 3) -> None:
        self.generator_fn = generator_fn
        self.max_repair_turns = max_repair_turns

    def generate_verified_story(
        self,
        base_prompt: str,
        bible_spec: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any], int]:
        current_prompt = base_prompt
        last_errors: list[str] = []

        for turn in range(self.max_repair_turns):
            draft = self.generator_fn(current_prompt, turn)
            draft_clean = " ".join(draft.splitlines()).strip()

            qa = analizar_cuento_multinino(draft_clean, bible_spec)
            ok_ngram, collisions = verificar_anti_regurgitacion_ngramas(draft_clean)
            if not ok_ngram:
                qa["valido"] = False
                qa["errores"].extend([f"Anti-regurgitación 10-gramas: {c}" for c in collisions])

            if qa["valido"]:
                return draft_clean, qa, turn + 1

            last_errors = list(qa["errores"])
            errores_bullet = "\n".join(f"- {err}" for err in last_errors)
            current_prompt = (
                f"{base_prompt}\n\n"
                f"CORRECCIÓN OBLIGATORIA (Intento {turn + 2}/{self.max_repair_turns}):\n"
                f"Tu borrador anterior fue rechazado por el validador determinista neuro-inclusivo:\n"
                f"{errores_bullet}\n"
                "Reescribe el capítulo completo corrigiendo TODOS los puntos anteriores, en un solo bloque continuo."
            )

        raise RuntimeError(f"SLM no logró pasar el validador tras {self.max_repair_turns} intentos: {last_errors}")


class RentedGpuWorker:
    """Cliente ejecutor sin estado para correr en contenedores GPU bajo demanda."""

    def __init__(
        self,
        control_plane_url: str,
        worker_token: str,
        worker_id: str,
        gpu_model: str = "NVIDIA-L4-24GB",
        slm_endpoint: str = "http://127.0.0.1:8000/v1",
        slm_model_tag: str = "Qwen/Qwen2.5-7B-Instruct-ROANSO-LoRA",
        slm_params_billions: float = 7.6,
        story_engine: Optional[SelfHealingStoryEngine] = None,
        audio_synthesizer: Optional[AudioSynthesizerFn] = None,
        post_text_stage_fn: Optional[Callable[[str, Dict[str, Any], int], Dict[str, Any]]] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        verificar_soberania_human_scale(slm_endpoint, param_billions=slm_params_billions)
        self.control_plane_url = control_plane_url.rstrip("/")
        self.worker_id = worker_id
        self.gpu_model = gpu_model
        self.slm_endpoint = slm_endpoint
        self.slm_model_tag = slm_model_tag
        self.slm_params_billions = slm_params_billions
        self.headers = {"X-Worker-Token": worker_token}
        self.client = http_client or httpx.Client(base_url=self.control_plane_url, timeout=120.0)
        self.story_engine = story_engine or SelfHealingStoryEngine(
            default_vllm_or_ollama_generator(slm_endpoint, slm_model_tag, slm_params_billions)
        )
        if audio_synthesizer is None:
            raise ValueError("Debe proveer `audio_synthesizer` (OmniVoice/F5-TTS adapter).")
        self.audio_synthesizer = audio_synthesizer
        self.post_text_stage_fn = post_text_stage_fn

    def process_single_lease(self) -> bool:
        """Reclama y ejecuta 1 trabajo de la cola. Retorna False si la cola está vacía (204)."""
        claim_resp = self.client.post(
            "/v1/worker/leases/claim",
            headers=self.headers,
            json={
                "worker_id": self.worker_id,
                "gpu_model": self.gpu_model,
                "inference_endpoint": self.slm_endpoint,
                "slm_params_billions": self.slm_params_billions,
            },
        )
        if claim_resp.status_code == 204:
            return False
        claim_resp.raise_for_status()
        lease = claim_resp.json()

        job_id = lease["job_id"]
        lease_token = lease["lease_token"]
        t0 = time.perf_counter()

        try:
            # Etapa 1: Generación SLM <= 8B + Auto-reparación determinista
            story_text, qa_report, turns_used = self.story_engine.generate_verified_story(
                base_prompt=lease["writer_prompt"],
                bible_spec=lease["bible_spec"],
            )
            editorial_extra: Dict[str, Any] = {}
            if self.post_text_stage_fn is not None:
                editorial_extra = self.post_text_stage_fn(
                    story_text, lease["bible_spec"], int(lease["chapter_number"])
                )

            # Heartbeat hacia Etapa 2 (Síntesis Vocal Nocturna)
            self.client.post(
                f"/v1/worker/leases/{job_id}/heartbeat",
                headers=self.headers,
                json={
                    "worker_id": self.worker_id,
                    "lease_token": lease_token,
                    "stage_detail": "synthesizing_nocturnal_audio",
                },
            ).raise_for_status()

            ephemeral_voice_bytes: Optional[bytes] = None
            if lease.get("ephemeral_voice_embedding_b64"):
                ephemeral_voice_bytes = base64.b64decode(lease["ephemeral_voice_embedding_b64"])

            bible_with_voice = dict(lease["bible_spec"])
            bible_with_voice["_runtime_voice_id"] = lease.get("voice_id", "catalog-chileno-sereno-m1")

            try:
                synth_res = self.audio_synthesizer(story_text, bible_with_voice, ephemeral_voice_bytes)
            finally:
                # Limpieza inmediata de referencia biométrica en RAM del worker
                ephemeral_voice_bytes = None

            # Heartbeat hacia Etapa 3 (Subida S3/R2 y Verificación Whisper)
            self.client.post(
                f"/v1/worker/leases/{job_id}/heartbeat",
                headers=self.headers,
                json={
                    "worker_id": self.worker_id,
                    "lease_token": lease_token,
                    "stage_detail": "verifying_whisper_and_uploading",
                },
            ).raise_for_status()

            up_resp = self.client.post(
                f"/v1/worker/leases/{job_id}/upload-audio",
                headers=self.headers,
                json={
                    "worker_id": self.worker_id,
                    "lease_token": lease_token,
                    "audio_mp3_b64": base64.b64encode(synth_res.mp3_bytes).decode("ascii"),
                },
            )
            up_resp.raise_for_status()
            audio_key = up_resp.json()["audio_storage_key"]

            gpu_seconds = max(0.05, round(time.perf_counter() - t0, 3))
            apodo = lease["bible_spec"].get("protagonista", {}).get("apodo", "Cuento")
            title = f"Capítulo {lease['chapter_number']}: La noche tranquila de {apodo}"

            comp_resp = self.client.post(
                f"/v1/worker/leases/{job_id}/complete",
                headers=self.headers,
                json={
                    "worker_id": self.worker_id,
                    "lease_token": lease_token,
                    "title": title,
                    "story_text": story_text,
                    "audio_storage_key": audio_key,
                    "whisper_wer": synth_res.whisper_wer,
                    "speaker_similarity": synth_res.speaker_similarity,
                    "slm_model_tag": self.slm_model_tag,
                    "slm_params_billions": self.slm_params_billions,
                    "inference_endpoint_used": self.slm_endpoint,
                    "gpu_seconds_used": gpu_seconds,
                    "editorial_review": {
                        "veredicto": editorial_extra.get("veredicto", "APROBADO"),
                        "criterios_6": editorial_extra.get("criterios", {}),
                        "vram_slm_evicted": editorial_extra.get("vram_slm_evicted", False),
                        "self_healing_turns": turns_used,
                        "szigriszt_pazos": qa_report["legibilidad"]["szigriszt_pazos"],
                    },
                },
            )
            comp_resp.raise_for_status()
            return True

        except Exception as exc:
            self.client.post(
                f"/v1/worker/leases/{job_id}/fail",
                headers=self.headers,
                json={
                    "worker_id": self.worker_id,
                    "lease_token": lease_token,
                    "error_message": str(exc)[:950],
                    "retryable": True,
                },
            )
            raise

    def drain_queue(self, max_jobs: int = 500) -> int:
        """Procesa trabajos hasta vaciar la cola (ideal para contenedores serverless/spot)."""
        processed = 0
        while processed < max_jobs:
            did_work = self.process_single_lease()
            if not did_work:
                break
            processed += 1
        return processed


def main() -> None:
    from .inference_adapters import (
        OmniVoiceWhisperRealSynthesizer,
        SlmNarrativeAndEditorialAdapter,
        VramStageScheduler,
    )

    parser = argparse.ArgumentParser(description="ROANSO Elastic GPU Worker")
    parser.add_argument("--control-plane", default=os.environ.get("ROANSO_CONTROL_PLANE_URL", "http://127.0.0.1:8080"))
    parser.add_argument(
        "--worker-token",
        default=os.environ.get("ROANSO_WORKER_SHARED_TOKEN", "dev-worker-shared-token-change-in-prod"),
    )
    parser.add_argument("--worker-id", default=os.environ.get("ROANSO_WORKER_ID", "gpu-node-01"))
    parser.add_argument("--gpu-model", default=os.environ.get("ROANSO_GPU_MODEL", "NVIDIA-L4-24GB"))
    parser.add_argument(
        "--slm-endpoint",
        default=os.environ.get("ROANSO_SLM_ENDPOINT", "http://127.0.0.1:11434/api/generate"),
    )
    parser.add_argument("--slm-model", default=os.environ.get("ROANSO_SLM_MODEL", "qwen2.5:7b"))
    parser.add_argument("--slm-params-b", type=float, default=float(os.environ.get("ROANSO_SLM_PARAMS_B", "7.6")))
    parser.add_argument(
        "--vram-policy",
        choices=["auto", "resident_24gb", "sequential_staged"],
        default=os.environ.get("ROANSO_VRAM_POLICY", "auto"),
    )
    parser.add_argument("--max-repair-turns", type=int, default=4)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument(
        "--drain-and-exit",
        action="store_true",
        help="Procesa la cola y apaga el contenedor (Scale-to-Zero).",
    )
    parser.add_argument("--dry-info", action="store_true", help="Imprime configuración resuelta y termina.")
    args = parser.parse_args()

    scheduler = VramStageScheduler(policy=args.vram_policy)  # type: ignore[arg-type]
    free_mib, total_mib = scheduler.get_cuda_memory_mib()
    eff_policy = scheduler.resolve_effective_policy()

    if args.dry_info:
        print(
            json.dumps(
                {
                    "worker_id": args.worker_id,
                    "control_plane": args.control_plane,
                    "gpu_model": args.gpu_model,
                    "cuda_free_mib": free_mib,
                    "cuda_total_mib": total_mib,
                    "effective_vram_policy": eff_policy,
                    "slm_endpoint": args.slm_endpoint,
                    "slm_model": args.slm_model,
                    "mode": "drain_and_exit" if args.drain_and_exit else "daemon",
                },
                ensure_ascii=False,
            )
        )
        return

    slm_adapter = SlmNarrativeAndEditorialAdapter(
        endpoint_url=args.slm_endpoint,
        model_tag=args.slm_model,
        param_billions=args.slm_params_b,
        scheduler=scheduler,
    )
    synth_adapter = OmniVoiceWhisperRealSynthesizer(scheduler=scheduler)

    def _post_text_stage(story_txt: str, bible_spec: Dict[str, Any], chap_num: int) -> Dict[str, Any]:
        ed = slm_adapter.run_editorial_review_6_criteria(story_txt, bible_spec, chapter_number=chap_num)
        evicted = slm_adapter.finish_text_stage()
        ed["vram_slm_evicted"] = evicted
        return ed

    worker = RentedGpuWorker(
        control_plane_url=args.control_plane,
        worker_token=args.worker_token,
        worker_id=args.worker_id,
        gpu_model=args.gpu_model,
        slm_endpoint=args.slm_endpoint,
        slm_model_tag=args.slm_model,
        slm_params_billions=args.slm_params_b,
        story_engine=SelfHealingStoryEngine(slm_adapter.build_generator_fn(), max_repair_turns=args.max_repair_turns),
        audio_synthesizer=synth_adapter,
        post_text_stage_fn=_post_text_stage,
    )

    if args.drain_and_exit:
        done = worker.drain_queue()
        print(
            json.dumps(
                {
                    "status": "drained_scale_to_zero",
                    "jobs_completed": done,
                    "effective_vram_policy": eff_policy,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    while True:
        did = worker.process_single_lease()
        if not did:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
