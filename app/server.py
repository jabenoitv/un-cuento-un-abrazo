# -*- coding: utf-8 -*-
"""Portable Interactive Showcase Server — Un Cuento, Un Abrazo (ROANSO SpA).

Zero external cloud dependencies; runs out-of-the-box on Linux, macOS, and Windows
using Python 3.10+ standard library (`python app/server.py`).

Demonstrates:
1. 4 Universal Design Sensory & Narrative Axes (TEA / ADHD / Neurotypical inclusive without medical labels).
2. Live Law 21.719 intake sanitizer (`sanitizar_onboarding_ley21719`) stripping clinical terms before disk write.
3. Real-time Spanish phonological readability (`Szigriszt-Pazos / INFLESZ`, `Fernández-Huerta`, `Gutiérrez de Polini`)
   + Carol Gray 10.2 non-directive check + anti-sermon validation (`pipeline_multinino.py`).
4. 3-Question Parent-Child Shared Conversation Card (`generar_guia_padres`).
5. 6" Screen-Safe E-Ink Bedside Reader Simulator & Human-Scale AI Architecture Manifest (`/api/human_scale_audit`).
"""
import json
import os
import re
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
STATIC_DIR = APP_DIR / "static"
MOTOR_DIR = BASE_DIR / "motor"
BIBLIAS_DIR = MOTOR_DIR / "biblias"
CAPITULOS_DIR = MOTOR_DIR / "capitulos"
COLA_PRODUCCION_FILE = MOTOR_DIR / "cola_produccion.jsonl"

sys.path.insert(0, str(MOTOR_DIR))
from motor_local_human_scale import generar_auditoria_mozilla  # noqa: E402
from pipeline_multinino import (  # noqa: E402
    ETIQUETAS_EJES,
    analizar_cuento_multinino,
    cargar_biblia,
    generar_guia_padres,
    normalizar_perfil_sensorial,
    sanitizar_onboarding_ley21719,
)


def normalize_profile_summary(pid: str, biblia: dict) -> dict:
    prot = biblia.get("protagonista", {})
    nombre = prot.get("nombre") or pid.capitalize()
    apodo = prot.get("apodo") or nombre
    edad = prot.get("edad") or 8
    rol = prot.get("rol") or "Explorador/a de historias"
    titulo_serie = biblia.get("titulo") or f"El Universo de {apodo}"
    valores = biblia.get("valores_trabajar") or ["Empatía", "Calma"]
    evitar = biblia.get("temas_evitar") or []
    reglas = biblia.get("reglas_produccion") or {}

    mascotas_list = []
    for k, m in (biblia.get("mascotas") or {}).items():
        mascotas_list.append({
            "nombre": m.get("nombre", k),
            "apodo": m.get("apodo", m.get("nombre", k)),
            "raza": m.get("raza") or "",
            "genero": m.get("genero", ""),
            "articulo": m.get("articulo", "la" if m.get("genero") == "hembra" else "el"),
            "estado": m.get("estado", "viva"),
        })

    hist_by_num = {
        int(c.get("capitulo") or (idx + 1)): c
        for idx, c in enumerate(biblia.get("historial_capitulos") or [])
        if isinstance(c, dict)
    }

    chapters = []
    for path in sorted(CAPITULOS_DIR.glob(f"{pid}_cap*.txt")):
        m = re.search(r"_cap(\d+)\.txt$", path.name)
        num = int(m.group(1)) if m else 1
        text_content = path.read_text(encoding="utf-8", errors="ignore")
        meta = hist_by_num.get(num, {})
        rep = analizar_cuento_multinino(text_content, biblia)
        words = len(text_content.split())
        chapters.append({
            "numero": num,
            "titulo": meta.get("titulo", f"Capítulo {num}: {titulo_serie}"),
            "resumen": meta.get("resumen", f"Capítulo adaptado para {apodo}."),
            "hilo_abierto": meta.get("hilo_abierto", ""),
            "mundo": meta.get("mundo", titulo_serie),
            "genero": meta.get("genero", "Cuento nocturno"),
            "guinos": meta.get("guinos", []),
            "words": words,
            "duracion_min": round(words / 145, 1),
            "valido": rep["valido"],
            "errores": rep["errores"],
        })

    perfil_sensorial = normalizar_perfil_sensorial(biblia)
    perfil_etiquetas = {
        eje: ETIQUETAS_EJES.get(eje, {}).get(val, val)
        for eje, val in perfil_sensorial.items()
    }

    return {
        "id": pid,
        "nombre": nombre,
        "apodo": apodo,
        "edad": edad,
        "titulo_serie": titulo_serie,
        "rol_protagonista": rol,
        "valores_trabajar": valores,
        "temas_evitar": evitar,
        "intereses_especiales": biblia.get("intereses_especiales") or [],
        "perfil_sensorial": perfil_sensorial,
        "perfil_sensorial_etiquetas": perfil_etiquetas,
        "mascotas": mascotas_list,
        "reglas_produccion": reglas,
        "multiverso_v2": biblia.get("multiverso_v2", {}),
        "chapters": chapters,
    }


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class ShowcaseHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            html_bytes = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)
            return

        if path == "/api/human_scale_audit":
            self._send_json(generar_auditoria_mozilla())
            return

        if path == "/api/export_data":
            pid = qs.get("profile", ["demo_lucas"])[0]
            biblia = cargar_biblia(pid)
            self._send_json({
                "ley_21719_portabilidad": True,
                "export_date": datetime.now().isoformat(timespec="seconds"),
                "biblia": biblia,
                "resumen_capitulos": normalize_profile_summary(pid, biblia)["chapters"],
            })
            return

        if path == "/api/profiles":
            profiles = []
            for p in sorted(BIBLIAS_DIR.glob("*.json")):
                profiles.append(normalize_profile_summary(p.stem, cargar_biblia(p.stem)))
            order = {"demo_lucas": 0, "demo_mateo": 1, "demo_fresia": 2}
            profiles.sort(key=lambda x: (order.get(x["id"], 99), x["nombre"]))
            self._send_json({
                "profiles": profiles,
                "ejes_sensoriales_catalogo": ETIQUETAS_EJES,
                "human_scale_audit": generar_auditoria_mozilla(),
            })
            return

        if path == "/api/chapter":
            pid = qs.get("profile", ["demo_lucas"])[0]
            cap_num = int(qs.get("cap", ["1"])[0])
            biblia = cargar_biblia(pid)
            summary = normalize_profile_summary(pid, biblia)
            cap_meta = next((c for c in summary["chapters"] if c["numero"] == cap_num), {})
            cap_path = CAPITULOS_DIR / f"{pid}_cap{cap_num}.txt"
            texto = cap_path.read_text(encoding="utf-8", errors="ignore") if cap_path.exists() else ""
            analisis = analizar_cuento_multinino(texto, biblia)
            title = cap_meta.get("titulo") or f"Capítulo {cap_num}"
            showcase_mp3 = BASE_DIR / "platform" / "showcase_assets" / f"{pid}_cap{cap_num}.mp3"
            self._send_json({
                "profile": pid,
                "capitulo": cap_num,
                "titulo": title,
                "resumen": cap_meta.get("resumen", ""),
                "hilo_abierto": cap_meta.get("hilo_abierto", ""),
                "mundo": cap_meta.get("mundo", summary["titulo_serie"]),
                "texto": texto,
                "audio_url": f"/api/audio?profile={pid}&cap={cap_num}" if showcase_mp3.exists() else None,
                "validacion": {
                    "valido": analisis["valido"],
                    "palabras": len(texto.split()),
                    "errores": analisis["errores"],
                    "advertencias": analisis["advertencias"],
                },
                "legibilidad": analisis["legibilidad"],
                "perfil_sensorial": analisis["perfil_sensorial"],
                "guia_padres": generar_guia_padres(texto, biblia, title),
            })
            return

        if path == "/api/audio":
            pid = qs.get("profile", ["demo_lucas"])[0]
            cap_num = int(qs.get("cap", ["1"])[0])
            if not re.fullmatch(r"demo_[a-z0-9_]+", pid):
                self.send_error(400, "Invalid profile id")
                return
            mp3_file = BASE_DIR / "platform" / "showcase_assets" / f"{pid}_cap{cap_num}.mp3"
            if not mp3_file.exists():
                self.send_error(404, "Showcase audio not found")
                return
            raw = mp3_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(raw)
            return

        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}") if length > 0 else {}

        if parsed.path == "/api/profiles":
            nombre = (payload.get("nombre") or "Valeria").strip()
            apodo = (payload.get("apodo") or nombre.split()[0]).strip()
            edad = int(payload.get("edad") or 7)
            terminos_omitidos = set()

            def _clean(raw_txt: str) -> str:
                limpio, det = sanitizar_onboarding_ley21719(raw_txt or "")
                terminos_omitidos.update(det)
                return limpio

            titulo = _clean(payload.get("universo") or f"El Valle de {apodo}") or f"El Valle de {apodo}"
            rol = _clean(payload.get("rol_protagonista") or f"Explorador/a de {titulo}") or f"Explorador/a de {titulo}"
            mascota_nombre = (payload.get("mascota_nombre") or "Canela").strip()
            mascota_genero = (payload.get("mascota_genero") or "hembra").strip().lower()
            valores = [v.strip() for v in _clean(payload.get("valores") or "Calma, Empatía mutua").split(",") if v.strip()] or ["Calma"]
            evitar = [p.strip() for p in _clean(payload.get("prohibidos") or "ruidos bruscos").split(",") if p.strip()]
            intereses = [i.strip() for i in _clean(payload.get("intereses_especiales") or "faroles tibios").split(",") if i.strip()]
            cierre = (payload.get("frase_cierre") or "tu familia te abraza siempre").strip()
            perfil_sensorial = normalizar_perfil_sensorial({"perfil_sensorial": payload.get("perfil_sensorial") or {}})

            slug = "demo_" + (re.sub(r"[^a-z0-9]+", "_", apodo.lower()).strip("_") or "perfil")
            m_key = re.sub(r"[^a-z0-9]+", "_", mascota_nombre.lower()).strip("_") or "mascota"
            art = "la" if mascota_genero == "hembra" else "el"
            adj_dorm = "echada y tranquila" if mascota_genero == "hembra" else "echado y tranquilo"
            adj_prot = "tranquila" if apodo.lower().endswith("a") else "tranquilo"
            interes_foco = intereses[0] if intereses else "los faroles tibios"

            biblia_data = {
                "nino_id": slug,
                "titulo": titulo,
                "protagonista": {
                    "nombre": nombre,
                    "apodo": apodo,
                    "rol": rol,
                    "edad": edad,
                    "genero": "femenino" if apodo.lower().endswith("a") else "masculino",
                },
                "perfil_sensorial": perfil_sensorial,
                "intereses_especiales": intereses,
                "valores_trabajar": valores,
                "temas_evitar": evitar,
                "mascotas": {
                    m_key: {
                        "nombre": mascota_nombre,
                        "apodo": mascota_nombre,
                        "genero": mascota_genero,
                        "articulo": art,
                        "estado": "viva" if mascota_genero == "hembra" else "vivo",
                        "raza": "Compañero/a leal",
                    }
                },
                "reglas_produccion": {
                    "palabras_minimas": 110,
                    "palabras_maximas": 1100,
                    "prosa_continua": True,
                    "frase_cierre_regex": re.escape(cierre),
                    "frase_cierre_texto": cierre,
                },
            }
            (BIBLIAS_DIR / f"{slug}.json").write_text(json.dumps(biblia_data, ensure_ascii=False, indent=2), encoding="utf-8")

            cap1_text = (
                f"Buenas noches, {apodo}. Hoy vamos a escuchar un cuento tranquilo en {titulo}. "
                f"Primero revisaremos {interes_foco} junto a {art} {mascota_nombre}. "
                f"Después dejaremos todo ordenado y volveremos a descansar sin apuro. "
                f"Esta tarde {apodo} encendió su lámpara tibia de madera. "
                f"A su lado caminaba {art} {mascota_nombre} con pasos suaves y parejos. "
                f"Como {rol}, {apodo} miró con atención cada detalle de {interes_foco} y acomodó una pieza pequeña en su riel exacto. "
                f"{apodo} se sintió muy {adj_prot} porque todo funcionaba en silencio y armonía. "
                f"En la habitación tibia, {art} {mascota_nombre} quedó {adj_dorm} sobre su manta suave. "
                f"Mañana al despertar seguirá todo en su lugar seguro, recordando siempre que {cierre}."
            )
            (CAPITULOS_DIR / f"{slug}_cap1.txt").write_text(cap1_text, encoding="utf-8")
            self._send_json({"ok": True, "profile_id": slug, "advertencias_ley21719": sorted(terminos_omitidos)})
            return

        self.send_error(404, "Not found")


if __name__ == "__main__":
    port = int(os.environ.get("ROANSO_PORT", "8095"))
    server = ThreadingHTTPServer(("127.0.0.1", port), ShowcaseHandler)
    print(f"[ROANSO Human-Scale Showcase] Live at http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
