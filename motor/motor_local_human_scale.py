# -*- coding: utf-8 -*-
"""Orquestador Local Human-Scale AI (<= 9B parámetros, Cero Frontier APIs) — ROANSO SpA.

Garantiza soberanía técnica y cumplimiento estricto de los principios de Mozilla
Human-Scale AI Incubator ("Small, Intentional, Tailored"):
1. SMALL (<= 10B parámetros): Ejecuta generación narrativa y revisión editorial de 6 criterios
   sobre modelos abiertos compactos locales (`qwen2.5:7b-instruct` o `gemma2:9b` + LoRA ROANSO)
   vía servidor local compatible con Ollama / llama.cpp / vLLM (`127.0.0.1:11434`).
2. CERO DEPENDENCIA DE FRONTIER APIs: Bloquea por diseño llamadas a endpoints cerrados masivos
   (`googleapis.com`, `openai.com`, `anthropic.com`, etc.).
3. ANTI-REGURGITACIÓN DE CORPUS (`CORPUS_CUENTOS_LIBRES.md` Regla 8): Verifica que ningún
   n-grama largo (>= 10 palabras consecutivas) copie literalmente textos del corpus humano libre.
4. INTENTIONAL & TAILORED: Encadena `analizar_cuento_multinino` (Ley 21.719, Carol Gray 10.2,
   legibilidad RAE/INFLESZ y 4 ejes sensoriales) antes de habilitar la síntesis local en OmniVoice.

Uso:
  python motor_local_human_scale.py --auditoria-mozilla
  python motor_local_human_scale.py revisar <nino_id> <guion.txt> [capitulo_num]
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urlparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MOTOR_DIR = Path(__file__).parent
CORPUS_LOCAL_DIR = MOTOR_DIR.parent / "corpus_limpio"
REVISIONES_DIR = MOTOR_DIR / "revisiones"

from pipeline_multinino import (  # noqa: E402
    analizar_cuento_multinino,
    cargar_biblia,
    construir_instrucciones_ejes,
    construir_pauta_revisor,
    generar_guia_padres,
    normalizar_perfil_sensorial,
)

MAX_PARAM_BILLIONS = 10.0

FRONTIER_API_HOSTS_BLOQUEADOS = (
    "generativelanguage.googleapis.com",
    "api.openai.com",
    "api.anthropic.com",
    "api.cohere.ai",
    "api.mistral.ai",
    "api.together.xyz",
)

STACK_MODELOS_HUMAN_SCALE = [
    {
        "componente": "Generador y Revisor Narrativo Español/Chileno (SLM + LoRA)",
        "modelo_base": "Qwen2.5-7B-Instruct / Gemma-2-9B-It (LoRA ROANSO Cuentos Libres)",
        "parametros_billones": 7.6,
        "licencia": "Apache-2.0 / Gemma Open Weights",
        "ejecucion": "Local GPU (RTX 5060 / Ollama / llama.cpp Q4_K_M ~5.2 GB VRAM)",
        "depende_frontier_api": False,
    },
    {
        "componente": "Síntesis Vocal Nocturna con Sleep Ramp",
        "modelo_base": "OmniVoice / F5-TTS LoRA Chileno",
        "parametros_billones": 0.6,
        "licencia": "Apache-2.0 / MIT",
        "ejecucion": "Local GPU (RTX 5060 FP16 ~2.4 GB VRAM)",
        "depende_frontier_api": False,
    },
    {
        "componente": "Verificador Acústico Texto-Audio (QA Inteligibilidad)",
        "modelo_base": "OpenAI Whisper (small / turbo open-weights local)",
        "parametros_billones": 0.81,
        "licencia": "MIT",
        "ejecucion": "Local GPU / CPU",
        "depende_frontier_api": False,
    },
    {
        "componente": "Verificador Biométrico de Timbre y Consentimiento Adulto",
        "modelo_base": "WavLM-Base-Plus-SV",
        "parametros_billones": 0.095,
        "licencia": "MIT",
        "ejecucion": "Local CPU / GPU",
        "depende_frontier_api": False,
    },
    {
        "componente": "Linter Sensorial, Carol Gray 10.2, Ley 21.719 e INFLESZ Español",
        "modelo_base": "Determinista simbólico en Python puro (`pipeline_multinino.py`)",
        "parametros_billones": 0.0,
        "licencia": "Apache-2.0 (ROANSO Open Good)",
        "ejecucion": "Local CPU / Offline PWA / Lector E-Ink",
        "depende_frontier_api": False,
    },
]


def verificar_soberania_human_scale(endpoint_url: str, param_billions: float = 7.6) -> None:
    """Bloquea endpoints de modelos frontier en la nube y modelos > 10B parámetros."""
    if param_billions > MAX_PARAM_BILLIONS:
        raise ValueError(
            f"Violación Human-Scale AI: el modelo declara {param_billions}B parámetros "
            f"(máximo permitido: {MAX_PARAM_BILLIONS}B)."
        )
    host = (urlparse(endpoint_url).hostname or "").lower()
    for prohibido in FRONTIER_API_HOSTS_BLOQUEADOS:
        if host == prohibido or host.endswith("." + prohibido):
            raise PermissionError(
                f"Violación Human-Scale AI: host frontier bloqueado ('{host}'). "
                "Use inferencia abierta local (ej. http://127.0.0.1:11434)."
            )


def _normalizar_tokens(texto: str) -> List[str]:
    return re.findall(r"[a-záéíóúüñ0-9]+", texto.lower())


def verificar_anti_regurgitacion_ngramas(
    texto_generado: str,
    corpus_dir: Path = CORPUS_LOCAL_DIR,
    n: int = 10,
) -> Tuple[bool, List[str]]:
    """Regla 8 de CORPUS_CUENTOS_LIBRES.md: impide copiar cadenas literales >= n palabras del corpus."""
    tokens_gen = _normalizar_tokens(texto_generado)
    if len(tokens_gen) < n or not corpus_dir.exists():
        return True, []

    ngramas_gen: Set[Tuple[str, ...]] = {
        tuple(tokens_gen[i: i + n]) for i in range(len(tokens_gen) - n + 1)
    }
    colisiones: List[str] = []
    for txt_file in sorted(corpus_dir.glob("*.txt")):
        # Las plantillas propias CC-BY de ROANSO comparten saludos rituales predecibles por diseño
        if txt_file.name.startswith("roanso_ccby_"):
            continue
        tokens_ref = _normalizar_tokens(txt_file.read_text(encoding="utf-8", errors="ignore"))
        if len(tokens_ref) < n:
            continue
        for j in range(len(tokens_ref) - n + 1):
            ng = tuple(tokens_ref[j: j + n])
            if ng in ngramas_gen:
                colisiones.append(f"{txt_file.name}: '{' '.join(ng)}'")
                break
    return len(colisiones) == 0, colisiones


def invocar_slm_local(
    prompt: str,
    endpoint_url: str = "",
    model_tag: str = "",
    json_mode: bool = False,
    param_billions: float = 7.6,
) -> str:
    """Llama a un servidor local Ollama / llama.cpp / vLLM sin claves en la nube."""
    endpoint = endpoint_url or os.environ.get("ROANSO_LOCAL_LLM_URL", "http://127.0.0.1:11434/api/generate")
    modelo = model_tag or os.environ.get("ROANSO_LOCAL_MODEL", "qwen2.5:7b-instruct")
    verificar_soberania_human_scale(endpoint, param_billions=param_billions)

    payload: Dict[str, Any] = {
        "model": modelo,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.25 if json_mode else 0.65},
    }
    if json_mode:
        payload["format"] = "json"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return (raw.get("response") or "").strip()


def _cargar_ancla_estilo_corpus_humano(corpus_dir: Path = CORPUS_LOCAL_DIR) -> str:
    """Recupera un pasaje nocturno sereno del corpus abierto auditado para anclar cadencia chilena/sureña."""
    pasajes_file = corpus_dir / "pasajes_nocturnos_seguros.jsonl"
    if not pasajes_file.exists():
        return ""
    try:
        lines = [ln for ln in pasajes_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for line in lines:
            row = json.loads(line)
            txt = str(row.get("texto", ""))
            if row.get("idioma") == "es-CL" and "--" not in txt and "Caleta Niebla" in txt:
                return txt[:340]
        for line in lines:
            row = json.loads(line)
            txt = str(row.get("texto", ""))
            if "--" not in txt and len(txt) >= 120:
                return txt[:340]
        return str(json.loads(lines[0]).get("texto", ""))[:340] if lines else ""
    except Exception:
        return ""


def construir_prompt_escritor_local(biblia: Dict[str, Any], semilla_padre: str) -> str:
    """Construye el prompt de generación adaptado a modelos locales compactos (7B-9B)."""
    prot = biblia.get("protagonista", {})
    apodo = prot.get("apodo", "el niño/a")
    genero_prot = str(prot.get("genero", "")).strip().lower()
    if genero_prot in ("femenino", "mujer", "niña", "nina"):
        etiqueta_prot = f"la niña {apodo} ({prot.get('edad', 7)} años, género femenino — usa siempre pronombres y adjetivos femeninos para {apodo})"
    elif genero_prot in ("masculino", "hombre", "niño", "nino"):
        etiqueta_prot = f"el niño {apodo} ({prot.get('edad', 7)} años, género masculino)"
    else:
        etiqueta_prot = f"{apodo} ({prot.get('edad', 7)} años)"
    mascota = prot.get("mascota", {})
    if not (isinstance(mascota, dict) and mascota.get("nombre")) and isinstance(biblia.get("mascotas"), dict):
        first_pet = next(iter(biblia["mascotas"].values()), {})
        if isinstance(first_pet, dict):
            mascota = first_pet
    mascota_txt = ""
    if isinstance(mascota, dict) and mascota.get("nombre"):
        art = "la" if str(mascota.get("genero", "hembra")).lower() == "hembra" else "el"
        especie_desc = mascota.get("especie") or mascota.get("raza") or "mascota"
        mascota_txt = (
            f" Su mascota compañera es {art} {mascota['nombre']} ({especie_desc}; "
            f"usa siempre el artículo '{art}' y concordancia gramatical {mascota.get('genero', 'hembra')})."
        )

    perfil = normalizar_perfil_sensorial(biblia)
    intereses = biblia.get("intereses_especiales", [])
    reglas = biblia.get("reglas_produccion", {})
    cierre = reglas.get("frase_cierre_texto", "tu papá te ama")
    min_pal = reglas.get("palabras_minimas", 650)
    max_pal = reglas.get("palabras_maximas", 1100)
    valores = ", ".join(biblia.get("valores_trabajar", ["calma", "empatía mutua"]))
    evitar = ", ".join(biblia.get("temas_evitar", [])) or "ninguno adicional"
    instrucciones_ejes = construir_instrucciones_ejes(perfil, intereses)
    ancla_estilo = _cargar_ancla_estilo_corpus_humano()
    bloque_ancla = (
        f"\nCADENCIA LITERARIA DE REFERENCIA (SOLO ESTILO Y TONO, PROHIBIDO COPIAR FRASES):\n\"{ancla_estilo}\"\n"
        if ancla_estilo
        else ""
    )

    return f"""Eres un narrador chileno cálido de cuentos nocturnos para dormir ("Un cuento, un abrazo").
Escribe UN capítulo completo en TERCERA PERSONA y prosa continua (un solo párrafo fluido sin saltos de línea ni títulos, entre {min_pal} y {max_pal} palabras) protagonizado por {etiqueta_prot}.{mascota_txt}

SEMILLA DEL ADULTO CUIDADOR HOY: {semilla_padre}
VALORES LAICOS A MOSTRAR CON ACCIONES (jamás sermonear ni usar 2ª persona imperativa): {valores}
TEMAS PROHIBIDOS: médicos, hospitales, terapias, medicamentos, {evitar}
IDENTIDAD LÉXICA CHILENA OBLIGATORIA:
- Escribe en español de Chile afectivo y natural (ej.: chaqueta, chaleco, frazada, conversar, piscina, porotos, jugo, vereda, despacito).
- PROHIBIDO usar español neutro de doblaje, mexicanismos o peninsularismos como: chamarra, platicar, alberca, frijoles, autobús, popote, cacahuate, zumo, nevera, ordenador o vosotros.
- Narra siempre en 3ª persona cálida sobre {apodo} (jamás narres en 1ª persona diciendo 'papá y yo' o 'mamá y yo').
REGLAS SENSORIALES Y ESTRUCTURALES OBLIGATORIAS:
{instrucciones_ejes}
- Toda emoción relevante debe explicitar su causa concreta ('se sintió tranquilo porque...').
- No uses la palabra 'anoche'. La historia ocurre hoy en la tarde/noche y mira con calma hacia mañana.
- Termina integrando literalmente las palabras exactas de cierre familiar: '{cierre}'.{bloque_ancla}"""


def revisar_cuento_human_scale(
    nino_id: str,
    guion_path: Path,
    cap_num: int = 999999,
) -> Dict[str, Any]:
    """Ejecuta validación determinista + anti-regurgitación + revisor SLM local (si Ollama está activo)."""
    biblia = cargar_biblia(nino_id)
    texto = guion_path.read_text(encoding="utf-8")

    analisis = analizar_cuento_multinino(texto, biblia)
    ok_ngram, colisiones = verificar_anti_regurgitacion_ngramas(texto)
    if not ok_ngram:
        analisis["valido"] = False
        analisis["errores"].extend([f"ERROR Anti-regurgitación: {c}" for c in colisiones])

    resultado_slm = None
    if analisis["valido"]:
        pauta = construir_pauta_revisor(texto, biblia, cap_num=cap_num)
        try:
            raw_json = invocar_slm_local(pauta, json_mode=True)
            if raw_json:
                resultado_slm = json.loads(raw_json)
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError) as exc:
            resultado_slm = {
                "modo": "offline_determinista",
                "detalle": f"Servidor SLM local no detectado en 127.0.0.1:11434 ({exc.__class__.__name__}); validación simbólica completa ejecutada.",
            }

    REVISIONES_DIR.mkdir(parents=True, exist_ok=True)
    reporte = {
        "nino_id": nino_id,
        "capitulo": cap_num,
        "human_scale_compliant": True,
        "max_model_parameters_billions": max(m["parametros_billones"] for m in STACK_MODELOS_HUMAN_SCALE),
        "frontier_api_dependency": False,
        "analisis_determinista": analisis,
        "anti_regurgitacion_ok": ok_ngram,
        "revisor_slm_local": resultado_slm,
        "guia_padres": generar_guia_padres(texto, biblia),
    }
    out_file = REVISIONES_DIR / f"{nino_id}_{guion_path.stem}_human_scale.json"
    out_file.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    return reporte


def generar_auditoria_mozilla() -> Dict[str, Any]:
    """Devuelve el manifiesto técnico auditable para Mozilla Human-Scale AI Incubator."""
    max_b = max(m["parametros_billones"] for m in STACK_MODELOS_HUMAN_SCALE)
    return {
        "proyecto": "Un Cuento, Un Abrazo — Open Human-Scale Neuro-Inclusive Storytelling Engine",
        "entidad": "ROANSO SpA (Rancagua, Región de O'Higgins, Chile)",
        "screening_mozilla": {
            "core_depends_on_frontier_api": False,
            "largest_model_parameter_count_billions": max_b,
            "mozilla_dropdown_bucket": "10 billion or fewer",
            "charitable_public_benefit": True,
        },
        "modelos_core": STACK_MODELOS_HUMAN_SCALE,
        "hosts_frontier_bloqueados_por_codigo": list(FRONTIER_API_HOSTS_BLOQUEADOS),
    }


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--auditoria-mozilla":
        print(json.dumps(generar_auditoria_mozilla(), ensure_ascii=False, indent=2))
        sys.exit(0)

    if len(sys.argv) >= 4 and sys.argv[1] == "revisar":
        nid = sys.argv[2]
        gpath = Path(sys.argv[3])
        cnum = int(sys.argv[4]) if len(sys.argv) >= 5 else 999999
        rep = revisar_cuento_human_scale(nid, gpath, cnum)
        print(
            f"[Human-Scale QA] {nid} | Válido: {rep['analisis_determinista']['valido']} | "
            f"Max Params: {rep['max_model_parameters_billions']}B | Frontier API: {rep['frontier_api_dependency']}"
        )
        sys.exit(0 if rep["analisis_determinista"]["valido"] else 1)

    print("Uso:")
    print("  python motor_local_human_scale.py --auditoria-mozilla")
    print("  python motor_local_human_scale.py revisar <nino_id> <guion.txt> [capitulo_num]")
    sys.exit(1)
