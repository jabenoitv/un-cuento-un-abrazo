# -*- coding: utf-8 -*-
"""Motor Multi-Niño y Sensorial v2 de Cuentos IA (ROANSO SpA).

Integra:
1. Los 4 Ejes Sensoriales y Narrativos (Diseño Universal: TEA / TDAH / Neurotípico)
   SIN almacenar etiquetas médicas ni diagnósticos clínicos (cumplimiento Ley 21.719).
2. Reglas derivadas de `estudio-tea/FUENTES_ESTUDIO_TEA.md`:
   - Literalidad y detección de modismos no explicados (Regla 2)
   - Control de largo oracional e índices de legibilidad en español:
     Fernández-Huerta, Szigriszt-Pazos (INFLESZ) y Gutiérrez de Polini (Reglas 3 y 21)
   - Estructura ritual anticipada sin cliffhangers angustiantes vs. enganche rápido + rampa de calma TDAH (Reglas 5 y 6)
   - Criterio Carol Gray 10.2: describir más que ordenar; bloqueo de 2ª persona imperativa (Regla 8)
   - Bloqueo de sermones / moralejas explícitas de IA y camuflaje neurotípico (Reglas 14, 15 y 16)
   - Guía breve de lectura/escucha compartida para padres: 3 preguntas + escalera de apoyo (Regla 17)
"""
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MOTOR_DIR = Path(__file__).parent
BIBLIAS_DIR = MOTOR_DIR / "biblias"
REVISIONES_DIR = MOTOR_DIR / "revisiones"

PROHIBIDOS_SALUD_GLOBAL = [
    "terapia", "terapeuta", "psicólogo", "psicologa", "psiquiatra",
    "doctor", "hospital", "clínica", "clinica", "medicamento", "remedio", "diagnóstico", "diagnostico"
]

# Términos clínicos/sensibles que NUNCA deben persistirse en la Biblia (Ley 21.719 §5.3)
TERMINOS_SENSIBLES_LEY21719 = [
    r"\btea\b", r"\bautis(?:mo|ta|tas)\b", r"\basperger\b", r"\btdah\b",
    r"\bd[eé]ficit\s+atencional\b", r"\bhiperactividad\b", r"\bmetilfenidato\b",
    r"\baripiprazol\b", r"\brisperidona\b", r"\bpsiquiatra\b", r"\bneur[oó]log[oa]\b",
    r"\bfonoaudi[oó]log[oa]\b", r"\bterapia\s+ocupacional\b", r"\bcie-?10\b", r"\bdsm-?5\b",
    r"\bdiscapacidad\b", r"\btrastorno\b", r"\bs[ií]ndrome\b"
]

CLICHES_MORALIZANTES = [
    (r"aprendi[oó]\s+una\s+valiosa\s+lecci[oó]n", "Cliché moralizante ('aprendió una valiosa lección')"),
    (r"la\s+moraleja\s+de\s+esta\s+historia", "Sermón explícito ('la moraleja de esta historia')"),
    (r"comprendi[oó]\s+que\s+lo\s+m[aá]s\s+importante\s+en\s+la\s+vida", "Sermón abstracto de IA"),
    (r"en\s+un\s+rinc[oó]n\s+m[aá]gico\s+del\s+universo", "Cliché sintético de apertura IA"),
    (r"desde\s+aquel\s+d[ií]a\s+prometi[oó]\s+portarse\s+bien", "Cierre conductista / moralizante"),
]

# Carol Gray 10.2: prohibido ordenar en 2ª persona ("tú debes", "tienes que portarte")
DIRECTIVAS_SEGUNDA_PERSONA = [
    r"\bt[uú]\s+debes\b",
    r"\btienes\s+que\s+(?:portarte|obedecer|callarte|mirar\s+a\s+los\s+ojos|quedarte\s+quiet[oa])\b",
    r"\bnunca\s+debes\s+(?:llorar|enojarte|mover\s+las\s+manos)\b",
    r"\bdebes\s+aprender\s+a\s+ser\s+normal\b",
]

# Modismos figurados opacos que confunden en modo literal si no se explican al tiro
MODISMOS_FIGURADOS = [
    (r"llov[ií]a\s+a\s+c[aá]ntaros", "llovía a cántaros"),
    (r"cost[oó]\s+un\s+ojo\s+de\s+la\s+cara", "costó un ojo de la cara"),
    (r"se\s+le\s+hel[oó]\s+la\s+sangre", "se le heló la sangre"),
    (r"se\s+mor[ií]a\s+de\s+(?:miedo|risa|hambre)", "se moría de... (hipérbole literalizable)"),
    (r"ech[oó]\s+chispas", "echó chispas"),
    (r"ten[ií]a\s+la\s+cabeza\s+en\s+las\s+nubes", "tenía la cabeza en las nubes"),
    (r"hablar\s+por\s+los\s+codos", "hablar por los codos"),
]

MARCADORES_EXPLICACION_LITERAL = re.compile(
    r"(?:eso\s+quiere\s+decir|quiere\s+decir\s+que|es\s+decir|o\s+sea\s+que|una\s+forma\s+de\s+decir)",
    re.IGNORECASE,
)

CLIFFHANGERS_ANGUSTIANTES = [
    r"\bcontinuar[aá]\s*\.\.\.",
    r"¿lograr[aá]n?\s+salvarse\?",
    r"nadie\s+sab[ií]a\s+qu[eé]\s+peligro",
    r"algo\s+terrible\s+estaba\s+por\s+ocurrir",
]

PERFIL_SENSORIAL_DEFAULTS = {
    "lenguaje": "literario_rico",          # literal_claro | agil_visual | literario_rico
    "estructura": "serie_multiverso",      # ritual_anticipado | enganche_rampa_calma | serie_multiverso
    "estimulo_sonoro": "dinamico_a_calmo", # monovoz_estable | dinamico_a_calmo | multivoz_personajes
    "foco_tematico": "rotacion_multiverso" # interes_profundo | misiones_hiperfoco | rotacion_multiverso
}

ETIQUETAS_EJES = {
    "lenguaje": {
        "literal_claro": "Literal y explícito (frases cortas, sin ironía)",
        "agil_visual": "Ágil y concreto (acción directa, sin rodeos)",
        "literario_rico": "Literario y cálido (metáforas suaves)"
    },
    "estructura": {
        "ritual_anticipado": "Ritual predecible con anticipación (sin suspenso abierto)",
        "enganche_rampa_calma": "Enganche en 15s + Rampa gradual de sueño",
        "serie_multiverso": "Serie episódica con guiños del Multiverso"
    },
    "estimulo_sonoro": {
        "monovoz_estable": "Monovoz pareja y serena (0.88x, re-escucha fácil)",
        "dinamico_a_calmo": "Expresivo al inicio → Susurro al final (Sleep Ramp)",
        "multivoz_personajes": "Narrador + Voces aprobadas de personajes"
    },
    "foco_tematico": {
        "interes_profundo": "Anclado en su interés especial como motor",
        "misiones_hiperfoco": "Misiones dinámicas conectadas a su pasión",
        "rotacion_multiverso": "Rotación amplia de 12 géneros"
    }
}


def normalizar_perfil_sensorial(biblia: Dict[str, Any]) -> Dict[str, str]:
    raw = biblia.get("perfil_sensorial") or {}
    out = {}
    for k, default_val in PERFIL_SENSORIAL_DEFAULTS.items():
        val = str(raw.get(k) or default_val).strip()
        out[k] = val if val in ETIQUETAS_EJES[k] else default_val
    return out


def sanitizar_onboarding_ley21719(texto: str) -> Tuple[str, List[str]]:
    """Elimina menciones a diagnósticos o datos clínicos de menores antes de persistir (Ley 21.719)."""
    if not texto:
        return "", []
    detectados: List[str] = []
    limpio = texto
    for pat in TERMINOS_SENSIBLES_LEY21719:
        for m in re.finditer(pat, limpio, flags=re.IGNORECASE):
            detectados.append(m.group(0))
        limpio = re.sub(pat, "", limpio, flags=re.IGNORECASE)
    limpio = re.sub(r"\s*,\s*,+", ", ", limpio)
    limpio = re.sub(r"\s{2,}", " ", limpio).strip(" ,;-")
    return limpio, sorted(set(d.lower() for d in detectados))


def contar_silabas_palabra_es(palabra: str) -> int:
    """Silabeador fonológico español basado en reglas de diptongo/hiato RAE."""
    w = re.sub(r"[^a-záéíóúüñ]", "", palabra.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    # Reducir dígrafos mudos que contienen 'u' sin sonido vocálico
    w = re.sub(r"([gq])u([eiéí])", r"\1\2", w)
    fuertes = set("aeoáéóíú")  # í/ú tónicas rompen diptongo (hiato acentual)
    debiles = set("iuü")
    vocales = fuertes | debiles

    grupos = 0
    i = 0
    n = len(w)
    while i < n:
        if w[i] not in vocales:
            # 'y' final suena como vocal 'i' (ej. 'hoy', 'rey')
            if w[i] == "y" and i == n - 1 and (i == 0 or w[i - 1] not in vocales):
                grupos += 1
            i += 1
            continue
        grupos += 1
        v1 = w[i]
        i += 1
        # Consumir diptongo o triptongo válido
        while i < n and w[i] in vocales:
            v2 = w[i]
            # Dos vocales fuertes (incluyendo í/ú tónicas) forman hiato -> nueva sílaba
            if v1 in fuertes and v2 in fuertes:
                break
            v1 = v2
            i += 1
    return max(1, grupos)


def calcular_legibilidad_es(texto: str) -> Dict[str, Any]:
    """Calcula Fernández-Huerta, Szigriszt-Pazos (INFLESZ) y Gutiérrez de Polini en español."""
    limpio = texto.strip()
    if not limpio:
        return {
            "palabras": 0,
            "oraciones": 0,
            "palabras_por_oracion": 0.0,
            "p90_palabras_oracion": 0,
            "silabas_por_palabra": 0.0,
            "fernandez_huerta": 0.0,
            "szigriszt_pazos": 0.0,
            "gutierrez_polini": 0.0,
            "nivel_lectura": "Sin texto",
        }

    oraciones_raw = [s.strip() for s in re.split(r"[.!?;:…]+", limpio) if re.search(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]", s)]
    num_oraciones = max(1, len(oraciones_raw))
    largos_oracion = [len(re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]+", s)) for s in oraciones_raw]
    largos_sorted = sorted(largos_oracion) if largos_oracion else [0]
    p90_idx = min(len(largos_sorted) - 1, int(0.9 * len(largos_sorted)))

    palabras = re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+", limpio)
    num_palabras = max(1, len(palabras))
    total_silabas = sum(contar_silabas_palabra_es(p) for p in palabras)
    total_letras = sum(len(p) for p in palabras)

    sil_por_100 = (total_silabas / num_palabras) * 100.0
    ora_por_100 = (num_oraciones / num_palabras) * 100.0
    pal_por_ora = num_palabras / num_oraciones
    sil_por_pal = total_silabas / num_palabras
    let_por_pal = total_letras / num_palabras

    fh = round(206.84 - (0.60 * sil_por_100) - (1.02 * pal_por_ora), 1)
    # Nota: la fórmula clásica de Fernández-Huerta usa P (sílabas/100 pal) y F (palabras/oración;
    # en la literatura original 1.02*F donde F=palabras/frase según adaptación moderna de INFLESZ/Muñoz).
    sz = round(206.835 - (62.3 * sil_por_pal) - pal_por_ora, 1)
    gp = round(95.2 - (9.7 * let_por_pal) - (0.35 * pal_por_ora), 1)

    if sz >= 75:
        nivel = "Muy fácil (4-7 años / Lectura Fácil)"
    elif sz >= 62:
        nivel = "Fácil-Fluido (7-10 años)"
    elif sz >= 50:
        nivel = "Estándar (10-12 años)"
    else:
        nivel = "Complejo (simplificar oraciones)"

    return {
        "palabras": len(palabras),
        "oraciones": num_oraciones,
        "palabras_por_oracion": round(pal_por_ora, 1),
        "p90_palabras_oracion": largos_sorted[p90_idx],
        "silabas_por_palabra": round(sil_por_pal, 2),
        "fernandez_huerta": fh,
        "szigriszt_pazos": sz,
        "gutierrez_polini": gp,
        "nivel_lectura": nivel,
    }


def cargar_biblia(nino_id: str) -> Dict[str, Any]:
    path = BIBLIAS_DIR / f"{nino_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe la biblia para nino_id='{nino_id}' en {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def analizar_cuento_multinino(texto: str, biblia: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta auditoría completa determinista + sensorial + legibilidad."""
    errores: List[str] = []
    advertencias: List[str] = []
    reglas = biblia.get("reglas_produccion", {})
    perfil = normalizar_perfil_sensorial(biblia)
    legibilidad = calcular_legibilidad_es(texto)

    # 1. Rango de palabras según la edad/configuración del niño
    min_pal = int(reglas.get("palabras_minimas", 900))
    max_pal = int(reglas.get("palabras_maximas", 1200))
    total_palabras = len(texto.split())
    apodo = biblia.get("protagonista", {}).get("apodo", "el protagonista")
    if total_palabras < min_pal or total_palabras > max_pal:
        errores.append(
            f"ERROR: Largo fuera de rango ({total_palabras} palabras). "
            f"Para {apodo} debe ir entre {min_pal} y {max_pal}."
        )

    # 2. Prosa continua para síntesis TTS sin cortes
    if reglas.get("prosa_continua", True) and "\n" in texto.strip():
        errores.append("ERROR: El guion trae saltos de línea; debe ser prosa continua para OmniVoice.")

    # 3. Frase de cierre personalizada de la familia
    frase_cierre_regex = reglas.get("frase_cierre_regex", r"tu\s+pap[aá]\s+te\s+ama")
    frase_cierre_ejemplo = reglas.get("frase_cierre_texto", "tu papá te ama")
    if frase_cierre_regex and not re.search(frase_cierre_regex, texto, re.IGNORECASE):
        errores.append(f"ERROR: Falta la frase de cierre familiar ('{frase_cierre_ejemplo}').")

    # 4. Prohibición global de temas médicos/terapias + miedos específicos del niño
    evitar = list(PROHIBIDOS_SALUD_GLOBAL) + list(biblia.get("temas_evitar", []))
    for palabra_prohibida in evitar:
        if not palabra_prohibida:
            continue
        pat = rf"\b{re.escape(palabra_prohibida)}\b"
        if re.search(pat, texto, re.IGNORECASE):
            errores.append(f"ERROR: Se detectó término prohibido/sensible para este niño: '{palabra_prohibida}'.")

    # 5. Línea temporal hoy noche -> mañana
    if re.search(r"\banoche\b", texto, re.IGNORECASE):
        errores.append("ERROR: Uso de 'anoche'. La línea temporal es hoy en la noche -> mañana al despertar.")

    # 6. Concordancia de género y estado de mascotas parametrizada desde la Biblia
    mascotas = biblia.get("mascotas", {})
    for key, m in mascotas.items():
        nombres_mascota = [m.get("nombre", key)]
        if m.get("apodo"):
            nombres_mascota.extend(
                [x.strip() for x in m["apodo"].replace("la ", "").replace("el ", "").replace("La ", "").replace("El ", "").split("/") if x.strip()]
            )
        nombres_pat = "|".join(re.escape(n) for n in nombres_mascota if n)
        genero = (m.get("genero") or "").lower()

        if genero == "hembra":
            if re.search(rf"\bel\s+({nombres_pat})\b", texto, re.IGNORECASE):
                errores.append(f"ERROR: '{m.get('nombre')}' es hembra; se detectó artículo masculino ('el ...').")
            masc_adj = re.findall(
                rf"({nombres_pat})\s+(estaba\s+echado|dormido|tranquilo|un\s+gran\s+guardi[aá]n)",
                texto,
                re.IGNORECASE,
            )
            if masc_adj:
                errores.append(f"ERROR: Concordancia masculina aplicada a '{m.get('nombre')}' (hembra): {masc_adj}")
        elif genero == "macho":
            if re.search(rf"\bla\s+({nombres_pat})\b", texto, re.IGNORECASE):
                errores.append(f"ERROR: '{m.get('nombre')}' es macho; se detectó artículo femenino ('la ...').")
            fem_adj = re.findall(
                rf"({nombres_pat})\s+(estaba\s+echada|dormida|tranquila)",
                texto,
                re.IGNORECASE,
            )
            if fem_adj:
                errores.append(f"ERROR: Concordancia femenina aplicada a '{m.get('nombre')}' (macho): {fem_adj}")

    # 7. Anti-sermón IA y Carol Gray 10.2 (Cero 2ª persona imperativa / cero moraleja escolar)
    for pat, etiqueta in CLICHES_MORALIZANTES:
        if re.search(pat, texto, re.IGNORECASE):
            errores.append(f"ERROR: {etiqueta}. Los valores se muestran con acciones, nunca con sermón.")
    for pat in DIRECTIVAS_SEGUNDA_PERSONA:
        if re.search(pat, texto, re.IGNORECASE):
            errores.append("ERROR: Directiva en 2ª persona detectada (viola criterio Carol Gray 10.2: describir, no ordenar).")

    # 8. Reglas adaptativas según los 4 Ejes Sensoriales
    if perfil["lenguaje"] == "literal_claro":
        for pat, etiqueta in MODISMOS_FIGURADOS:
            for m in re.finditer(pat, texto, re.IGNORECASE):
                ventana = texto[m.start(): min(len(texto), m.end() + 140)]
                if not MARCADORES_EXPLICACION_LITERAL.search(ventana):
                    errores.append(
                        f"ERROR: Modismo figurado sin explicación inmediata en modo Literal ('{etiqueta}')."
                    )
        if legibilidad["palabras_por_oracion"] > 22:
            advertencias.append(
                f"Oraciones algo largas para modo Literal ({legibilidad['palabras_por_oracion']} pal/oración; ideal <= 18-20)."
            )
    elif perfil["lenguaje"] == "agil_visual":
        if legibilidad["palabras_por_oracion"] > 25:
            advertencias.append(
                f"Oraciones extensas para perfil Ágil ({legibilidad['palabras_por_oracion']} pal/oración; ideal <= 22)."
            )

    if perfil["estructura"] == "ritual_anticipado":
        tramo_final = texto[-350:] if len(texto) > 350 else texto
        for pat in CLIFFHANGERS_ANGUSTIANTES:
            if re.search(pat, tramo_final, re.IGNORECASE):
                errores.append("ERROR: Cierre con suspenso angustiante en perfil Ritual Anticipado.")

    if perfil["estructura"] == "enganche_rampa_calma":
        primeras_dos = " ".join(re.split(r"[.!?]+", texto)[:2])
        intereses = [str(x).lower() for x in biblia.get("intereses_especiales", []) if x]
        if intereses and not any(it in primeras_dos.lower() for it in intereses + [apodo.lower()]):
            advertencias.append("Para perfil Enganche Rápido conviene nombrar a protagonista o su interés en las primeras 2 frases.")

    valido = len(errores) == 0
    return {
        "valido": valido,
        "errores": errores,
        "advertencias": advertencias,
        "perfil_sensorial": perfil,
        "legibilidad": legibilidad,
    }


def validar_cuento_multinino(texto: str, biblia: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Interfaz compatible hacia atrás: devuelve (valido, errores)."""
    res = analizar_cuento_multinino(texto, biblia)
    return res["valido"], res["errores"]


def construir_instrucciones_ejes(perfil: Dict[str, str], intereses: List[str]) -> str:
    """Traduce los 4 ejes sensoriales en reglas operativas concretas para el LLM escritor/revisor."""
    reglas_txt = []
    if perfil["lenguaje"] == "literal_claro":
        reglas_txt.append(
            "- LENGUAJE LITERAL Y EXPLÍCITO: Usa oraciones cortas y claras (ideal 12-18 palabras), voz activa y una idea por frase. "
            "Cero ironía o sarcasmo. Si usas un dicho chileno suave, explícalo en la misma frase. Mantén siempre el mismo nombre para cada objeto y personaje."
        )
    elif perfil["lenguaje"] == "agil_visual":
        reglas_txt.append(
            "- LENGUAJE ÁGIL Y VISUAL: Verbos concretos y sensoriales, sin descripciones estáticas largas al comienzo."
        )
    else:
        reglas_txt.append(
            "- LENGUAJE LITERARIO CÁLIDO: Español chileno cariñoso y natural, con imágenes poéticas suaves e incorporando 1-2 palabras ricas explicadas por contexto."
        )

    if perfil["estructura"] == "ritual_anticipado":
        reglas_txt.append(
            "- ESTRUCTURA RITUAL ANTICIPADA: Abre con un saludo nocturno predecible y anuncia con calma qué ocurrirá hoy en el cuento. "
            "Si un plan cambia dentro de la historia, avísalo antes de que pase. Cierra el episodio por completo sin cliffhangers angustiantes."
        )
    elif perfil["estructura"] == "enganche_rampa_calma":
        reglas_txt.append(
            "- ENGANCHE EN 15 SEGUNDOS + RAMPA DE SUEÑO: Parte de inmediato en la primera frase con acción curiosa ligada a su interés especial, "
            "y ve desacelerando progresivamente el ritmo, la luz y el movimiento hasta terminar en calma profunda."
        )
    else:
        reglas_txt.append(
            "- ESTRUCTURA SERIE MULTIVERSO: Trama A autoconclusiva esta noche + un guiño sutil de Trama B o personaje puente."
        )

    if intereses:
        reglas_txt.append(
            f"- INTERÉS PROFUNDO COMO MOTOR: Integra de forma orgánica ({', '.join(intereses)}) como motor del ingenio de la trama, nunca como premio condicionado."
        )
    return "\n".join(reglas_txt)


def construir_pauta_revisor(texto: str, biblia: Dict[str, Any], cap_num: int = 999999) -> str:
    """Construye el prompt de revisión editorial adaptado al niño, sus 4 ejes y su Biblia."""
    prot = biblia["protagonista"]
    perfil = normalizar_perfil_sensorial(biblia)
    intereses = biblia.get("intereses_especiales", [])
    hist = [c for c in biblia.get("historial_capitulos", []) if c.get("capitulo", 0) < cap_num][-5:]
    valores = ", ".join(biblia.get("valores_trabajar", ["empatía", "humildad", "resolver conflictos"]))
    evitar = ", ".join(biblia.get("temas_evitar", [])) or "ninguno adicional"
    instrucciones_ejes = construir_instrucciones_ejes(perfil, intereses)

    return f"""Eres editor jefe de cuentos infantiles nocturnos de ROANSO ("Un cuento, un abrazo"). Revisa el cuento personalizado para {prot['apodo']} ({prot['edad']} años).
Valores laicos a modelar por observación vicaria (SIN moraleja explícita, SIN sermón y SIN 2ª persona imperativa): {valores}.
Si hay malentendidos entre personajes, verifica empatía de doble vía (ambos ajustan y se escuchan; jamás exigir 'camuflarse' o reprimir quién es).
Toda emoción importante debe mostrar su causa concreta ('sintió X porque pasó Y').
Temas o miedos prohibidos para {prot['apodo']}: salud, médicos, terapias, {evitar}.

PERFIL SENSORIAL Y NARRATIVO CONFIGURADO POR LA FAMILIA:
{instrucciones_ejes}

Criterios a evaluar (cada uno: ok true/false y nota breve en español):
1 edad_seguridad: emocional y físicamente seguro para {prot['edad']} años, sin amenazas ni temas médicos.
2 ajuste_sensorial: cumple fielmente las reglas de lenguaje y estructura del perfil configurado arriba.
3 valores_sin_sermon: enseña o celebra fortalezas mediante acciones concretas, causas emocionales explícitas y empatía mutua, sin clichés de IA.
4 chileno_natural: español de Chile afectivo y limpio, sin voseo argentino ni giros de traducción.
5 rampa_dormir: el último tercio baja la energía gradualmente y deja a {prot['apodo']} tranquilo/a para dormir.
6 coherencia_canon: inicio, nudo suave y desenlace propios; respeta géneros de mascotas, línea hoy noche -> mañana y variedad frente a capítulos previos.

Responde SOLO un JSON válido:
{{"veredicto":"APROBADO"|"REESCRIBIR","criterios":{{"edad_seguridad":{{"ok":true,"nota":""}},"ajuste_sensorial":{{"ok":true,"nota":""}},"valores_sin_sermon":{{"ok":true,"nota":""}},"chileno_natural":{{"ok":true,"nota":""}},"rampa_dormir":{{"ok":true,"nota":""}},"coherencia_canon":{{"ok":true,"nota":""}}}},"cambios":["instrucción concreta si hay que corregir"]}}

ÚLTIMOS CAPÍTULOS DE {prot['apodo'].upper()}: {json.dumps(hist, ensure_ascii=False)[:3200]}
CUENTO PROPUESTO:
{texto}"""


def generar_guia_padres(texto: str, biblia: Dict[str, Any], titulo: str = "") -> Dict[str, Any]:
    """Genera la micro-tarjeta de lectura/escucha compartida (Regla 17 de FUENTES_ESTUDIO_TEA.md)."""
    prot = biblia.get("protagonista", {})
    apodo = prot.get("apodo", "tu hijo/a")
    valores = biblia.get("valores_trabajar") or ["la calma y el cariño en equipo"]
    valor_foco = valores[0]
    mascotas = list((biblia.get("mascotas") or {}).values())
    compa = mascotas[0].get("nombre") if mascotas else "los amigos del cuento"

    return {
        "fundamento": "Conversación compartida opcional de 2 minutos al desayuno (práctica con mayor respaldo en comprensión narrativa e inferencia socioemocional).",
        "valor_del_capitulo": valor_foco,
        "preguntas": [
            {
                "tipo": "1. Recuerdo seguro (concreta)",
                "pregunta": f"¿Qué parte del cuento '{titulo or 'de anoche'}' con {compa} te gustó más imaginar?"
            },
            {
                "tipo": "2. Emoción con causa (perspectiva)",
                "pregunta": "¿Por qué crees que los personajes se sintieron más tranquilos cuando se escucharon con calma?"
            },
            {
                "tipo": "3. Puente cariñoso al día de hoy",
                "pregunta": f"Si hoy nos pasa algo parecido a lo del cuento, ¿qué idea chiquitita podríamos probar juntos, {apodo}?"
            }
        ],
        "escalera_apoyo": "Sin presión: si no tiene ganas de responder, comenta tú primero tu parte favorita en una frase corta y déjalo abierto."
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python pipeline_multinino.py <nino_id> <archivo_guion.txt>")
        sys.exit(1)
    nino_id = sys.argv[1]
    guion_path = Path(sys.argv[2])
    biblia = cargar_biblia(nino_id)
    texto = guion_path.read_text(encoding="utf-8")
    reporte = analizar_cuento_multinino(texto, biblia)
    leg = reporte["legibilidad"]
    print(
        f"[{nino_id}] Palabras: {leg['palabras']} | Oraciones: {leg['oraciones']} "
        f"({leg['palabras_por_oracion']} pal/or) | Szigriszt: {leg['szigriszt_pazos']} ({leg['nivel_lectura']}) | "
        f"Válido: {reporte['valido']}"
    )
    for e in reporte["errores"]:
        print(" ", e)
    for w in reporte["advertencias"]:
        print("  [AVISO]", w)
    sys.exit(0 if reporte["valido"] else 1)
