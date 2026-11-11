# -*- coding: utf-8 -*-
"""Puente limpio hacia el motor determinista neuro-inclusivo (`motor/`)."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict

_CANDIDATE_MOTOR_DIRS = [
    Path(__file__).resolve().parents[2] / "motor",  # <repo_root>/motor
    Path(__file__).resolve().parents[1] / "motor",  # /app/motor en Docker
]

for _d in _CANDIDATE_MOTOR_DIRS:
    if _d.exists() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
        break

from motor_local_human_scale import (  # type: ignore  # noqa: E402
    MAX_PARAM_BILLIONS,
    STACK_MODELOS_HUMAN_SCALE,
    construir_prompt_escritor_local,
    verificar_anti_regurgitacion_ngramas,
    verificar_soberania_human_scale,
)
from pipeline_multinino import (  # type: ignore  # noqa: E402
    DESVIO_PRIMERA_PERSONA,
    ETIQUETAS_EJES,
    LEXICO_NO_CHILENO_DOBLAJE,
    PERFIL_SENSORIAL_DEFAULTS,
    analizar_cuento_multinino,
    calcular_legibilidad_es,
    construir_pauta_revisor,
    generar_guia_padres,
    normalizar_perfil_sensorial,
    sanitizar_onboarding_ley21719,
)


def build_bible_dict_from_child(
    alias: str,
    age_years: int,
    axes: Dict[str, str],
    special_interests: list[str],
    valores: list[str],
    temas_evitar: list[str],
    closing_phrase: str,
    extra_bible: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Construye el diccionario canónico de Biblia compatible 1:1 con `pipeline_multinino.py`."""
    extra = extra_bible or {}
    min_words = int(extra.get("palabras_minimas") or (240 if axes.get("estructura") == "enganche_rampa_calma" else 550))
    max_words = int(extra.get("palabras_maximas") or (550 if axes.get("estructura") == "enganche_rampa_calma" else 1050))
    mascota_obj = extra.get("mascota", {"nombre": "Canela", "especie": "perrita", "genero": "hembra"})
    return {
        "protagonista": {
            "apodo": alias,
            "edad": age_years,
            "mascota": mascota_obj,
        },
        "mascotas": {"principal": mascota_obj},
        "perfil_sensorial": normalizar_perfil_sensorial({"perfil_sensorial": axes}),
        "intereses_especiales": special_interests,
        "valores_trabajar": valores or ["calma", "empatía mutua"],
        "temas_evitar": temas_evitar,
        "puentes_multiverso": extra.get(
            "puentes_multiverso",
            {
                "lugares": ["Estación Quillay Alto", "Caleta Niebla"],
                "personajes": ["Don Anselmo", "Señora Rosa"],
            },
        ),
        "reglas_produccion": {
            "palabras_minimas": min_words,
            "palabras_maximas": max_words,
            "frase_cierre_texto": closing_phrase,
            "frase_cierre_regex": r"\s+".join(re.escape(w) for w in closing_phrase.strip().split()),
        },
    }


__all__ = [
    "DESVIO_PRIMERA_PERSONA",
    "ETIQUETAS_EJES",
    "LEXICO_NO_CHILENO_DOBLAJE",
    "MAX_PARAM_BILLIONS",
    "PERFIL_SENSORIAL_DEFAULTS",
    "STACK_MODELOS_HUMAN_SCALE",
    "analizar_cuento_multinino",
    "build_bible_dict_from_child",
    "calcular_legibilidad_es",
    "construir_pauta_revisor",
    "construir_prompt_escritor_local",
    "generar_guia_padres",
    "normalizar_perfil_sensorial",
    "sanitizar_onboarding_ley21719",
    "verificar_anti_regurgitacion_ngramas",
    "verificar_soberania_human_scale",
]
