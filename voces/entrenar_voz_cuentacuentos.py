# -*- coding: utf-8 -*-
"""Nocturnal Voice Embedding Trainer, Acoustic Warmth Filter & Adult Consent Verifier — ROANSO SpA.

Open-source (`Apache-2.0`) acoustic pipeline for Human-Scale nocturnal storytelling:
1. Organizes consenting adult recordings across 3 physiological bedtime registers:
   - `inicio_curioso`    : warm, gentle opening curiosity
   - `desarrollo_sereno` : steady, calm, articulated body narration
   - `cierre_sueno`      : intimate, low-energy sleep transition register
2. Applies an anti-sensory-fatigue DSP chain (`75 Hz` highpass, `190 Hz` warm body boost,
   `6.2 kHz` sibilance attenuation, `10.5 kHz` gentle rolloff, `-19 LUFS` normalization).
3. Evaluates candidate `.pt` embeddings objectively on Chilean/Spanish bedtime benchmark sentences:
   - Speaker timbre similarity (`microsoft/wavlm-base-plus-sv`, `95M` params, held-out centroid)
   - Intelligibility (`faster-whisper`, requiring `WER <= 0.12`)
   - Spectral warmth ratio (`130-650 Hz` vs. `2.5-5.5 kHz` harshness)
   - Sensory RMS stability (penalizing sudden loudness jumps that startle sensitive listeners)
4. Enforces spoken adult consent verification (`--verificar-consentimiento`) under Chile's Law 21.719.
   Child voices are strictly prohibited.

Usage:
  python voces/entrenar_voz_cuentacuentos.py --narrador narrador_sur --registro desarrollo_sereno
  python voces/entrenar_voz_cuentacuentos.py --verificar-consentimiento sample_adult.wav "Ana Morales"
"""
import argparse
import csv
import gc
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VOCES_DIR = Path(__file__).resolve().parent
CRUDOS_NOCTURNOS_DIR = VOCES_DIR / "crudos_nocturnos"
PERFILES_NARRADOR_DIR = VOCES_DIR / "perfiles_narrador"
TRABAJO_DIR = VOCES_DIR / "_trabajo_evaluacion"
HISTORIAL_CSV = VOCES_DIR / "historial_voces_cuentos.csv"

REGISTROS_VALIDOS = ("inicio_curioso", "desarrollo_sereno", "cierre_sueno")
EXTS_AUDIO = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".opus"}
WER_MAX = 0.12
MARGEN_MEJORA = 0.025

FRASES_CUENTO_CHILE = {
    "inicio_curioso": [
        "Esta noche en la estación de Quillay Alto se encendieron despacito los tres faroles de luz ámbar junto al puente colgante.",
        "Mira qué curioso ese pequeño reloj de bronce que don Anselmo dejó sobre la mesa de madera tibia.",
    ],
    "desarrollo_sereno": [
        "La perrita dio dos vueltas lentas sobre la alfombra de lana y apoyó el hocico tranquila mientras afuera caía una lluvia suave.",
        "Los dos amigos respiraron hondo, se escucharon con cariño y encontraron al tiro una forma justa de compartir el turno.",
    ],
    "cierre_sueno": [
        "Todo quedó en orden y en silencio bajo las estrellas lentas del sur hasta mañana al despertar.",
        "Ahora suelta los hombros sobre la almohada blandita, cierra los ojos sin prisa y descansa, porque tu familia te abraza siempre.",
    ],
}


def norm_palabras(texto: str) -> List[str]:
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-zñ0-9]+", t)


def calcular_wer(ref: str, hip: str) -> float:
    r, h = norm_palabras(ref), norm_palabras(hip)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def duracion_audio(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out or 0.0)


def preprocesar_wav_nocturno(src: Path, dst: Path, ini: float = 0.2, dur: float = 18.0) -> None:
    """Converts audio to 24 kHz mono with nocturnal warmth EQ and de-essing."""
    af_chain = (
        "highpass=f=75,"
        "equalizer=f=190:t=q:w=1.1:g=1.8,"
        "equalizer=f=6200:t=q:w=1.5:g=-2.5,"
        "lowpass=f=10500,"
        "silenceremove=stop_periods=-1:stop_duration=0.9:stop_threshold=-44dB,"
        "loudnorm=I=-19:TP=-2.0:LRA=7"
    )
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(ini), "-i", str(src)]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-af", af_chain, "-ac", "1", "-ar", "24000", str(dst)]
    subprocess.run(cmd, check=True)


def metricas_acusticas_nocturnas(wav_path: Path) -> Dict[str, float]:
    """Measures approximate SNR, spectral warmth (130-650 Hz vs 2.5-5.5 kHz), and RMS stability."""
    y, sr = sf.read(str(wav_path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)

    win = int(0.05 * sr)
    frames_rms = np.array(
        [np.sqrt(np.mean(y[i: i + win] ** 2)) + 1e-9 for i in range(0, max(1, len(y) - win), win)]
    )
    snr_db = (
        float(20.0 * np.log10(np.percentile(frames_rms, 90) / np.percentile(frames_rms, 10)))
        if len(frames_rms) > 10
        else 0.0
    )

    activos = frames_rms[frames_rms > np.percentile(frames_rms, 35)]
    saltos_db = float(np.std(20.0 * np.log10(activos))) if len(activos) > 5 else 5.0

    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    cuerpo = float(np.mean(spec[(freqs >= 130) & (freqs <= 650)]) + 1e-6)
    aspereza = float(np.mean(spec[(freqs >= 2500) & (freqs <= 5500)]) + 1e-6)
    indice_calidez = round(cuerpo / aspereza, 3)

    return {
        "snr_db": round(snr_db, 2),
        "estabilidad_db": round(saltos_db, 2),
        "calidez": indice_calidez,
    }


def verificar_consentimiento_hablado(audio_path: Path, nombre_adulto: str) -> Tuple[bool, Dict[str, object]]:
    """Verifies via local faster-whisper that an adult spoke explicit biometric consent (Law 21.719)."""
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segs, info = model.transcribe(str(audio_path), language="es", vad_filter=True)
    transcripcion = " ".join(s.text for s in segs).strip()
    palabras_trans = set(norm_palabras(transcripcion))
    tokens_nombre = [t for t in norm_palabras(nombre_adulto) if len(t) >= 3]

    tiene_nombre = all(t in palabras_trans for t in tokens_nombre[:2]) if tokens_nombre else False
    tiene_autorizo = any(w in palabras_trans for w in ("autorizo", "consiento", "acepto"))
    tiene_voz = "voz" in palabras_trans
    tiene_cuentos = any(w in palabras_trans for w in ("cuento", "cuentos", "roanso", "familia"))

    ok = bool(tiene_nombre and tiene_autorizo and tiene_voz and tiene_cuentos and info.duration >= 5.0)
    return ok, {
        "consentimiento_valido": ok,
        "duracion_s": round(info.duration, 1),
        "transcripcion": transcripcion,
        "checks": {
            "nombre_presente": tiene_nombre,
            "verbo_autorizacion": tiene_autorizo,
            "menciona_voz": tiene_voz,
            "finalidad_cuentos": tiene_cuentos,
        },
    }


def entrenar_registro(narrador: str, registro: str) -> int:
    if registro not in REGISTROS_VALIDOS:
        raise SystemExit(f"Invalid register '{registro}'. Valid: {REGISTROS_VALIDOS}")

    src_dir = CRUDOS_NOCTURNOS_DIR / narrador / registro
    out_dir = PERFILES_NARRADOR_DIR / narrador
    src_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    TRABAJO_DIR.mkdir(parents=True, exist_ok=True)

    audios = sorted(p for p in src_dir.glob("*") if p.is_file() and p.suffix.lower() in EXTS_AUDIO)
    if len(audios) < 2:
        print(
            f"[ROANSO-VOICE] Input folder ready at: {src_dir}\n"
            f"Place at least 2-3 clean adult takes (10-25s each) for register '{registro}'."
        )
        return 1

    fuentes = []
    for p in audios:
        if duracion_audio(p) < 6.0:
            continue
        w = TRABAJO_DIR / f"norm_{narrador}_{registro}_{p.stem}.wav"
        preprocesar_wav_nocturno(p, w, ini=0.25, dur=18.0)
        d = duracion_audio(w)
        if d >= 5.5:
            ac = metricas_acusticas_nocturnas(w)
            fuentes.append({"src": p, "wav": w, "dur": d, **ac})

    if len(fuentes) < 2:
        raise SystemExit(f"Insufficient usable takes (>=6s) in {src_dir}")

    fuentes.sort(key=lambda x: -(x["snr_db"] + 2.0 * x["calidez"] - x["estabilidad_db"]))
    pt_vigente = out_dir / f"{registro}.pt"
    candidatas = []
    if pt_vigente.exists():
        candidatas.append({"nombre": "vigente", "pt": pt_vigente, "excluir": set()})

    for f in fuentes[:4]:
        candidatas.append({"nombre": f["src"].stem, "ref": f["wav"], "excluir": {f["src"].stem}})

    import librosa
    import torch
    from faster_whisper import WhisperModel
    from omnivoice import OmniVoice, VoiceClonePrompt
    from transformers import AutoFeatureExtractor, WavLMForXVector

    frases = FRASES_CUENTO_CHILE[registro]
    speed_eval = {"inicio_curioso": 0.96, "desarrollo_sereno": 0.92, "cierre_sueno": 0.87}[registro]

    tts = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    for idx, c in enumerate(candidatas):
        c["idx"] = idx
        pt_tmp = TRABAJO_DIR / f"cand_{idx}.pt"
        if "pt" in c:
            shutil.copy2(c["pt"], pt_tmp)
        else:
            tts.create_voice_clone_prompt(ref_audio=str(c["ref"])).save(str(pt_tmp))
        prompt = VoiceClonePrompt.load(str(pt_tmp))
        for j, fr in enumerate(frases):
            wav_gen = tts.generate(text=fr, language="es", voice_clone_prompt=prompt, speed=speed_eval)[0]
            sf.write(str(TRABAJO_DIR / f"gen_{idx}_{j}.wav"), np.asarray(wav_gen, dtype=np.float32), 24000)
        c["ok"] = True
    del tts
    gc.collect()
    torch.cuda.empty_cache()

    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to("cuda:0").eval()

    def extraer_emb(wav_file: Path) -> torch.Tensor:
        y, sr = sf.read(str(wav_file))
        y16 = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=16000)[: 16000 * 20]
        inp = fe(y16, sampling_rate=16000, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            e = sv(**inp).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1).cpu()

    reales_emb = {f["src"].stem: extraer_emb(f["wav"]) for f in fuentes}
    asr = WhisperModel("small", device="cpu", compute_type="int8")

    evaluadas = []
    for c in candidatas:
        if not c.get("ok"):
            continue
        restantes = [e for k, e in reales_emb.items() if k not in c["excluir"]] or list(reales_emb.values())
        centroide = torch.nn.functional.normalize(torch.stack(restantes).mean(0), dim=-1)
        sims, wers, calideces, estabs = [], [], [], []
        for j, fr in enumerate(frases):
            gwav = TRABAJO_DIR / f"gen_{c['idx']}_{j}.wav"
            segs, _ = asr.transcribe(str(gwav), language="es", vad_filter=False)
            hip = " ".join(s.text for s in segs)
            wers.append(calcular_wer(fr, hip))
            ac = metricas_acusticas_nocturnas(gwav)
            calideces.append(ac["calidez"])
            estabs.append(ac["estabilidad_db"])
            eg = extraer_emb(gwav)
            sims.append(float(eg @ centroide))
        c["wer"] = round(max(wers), 3)
        c["calidez"] = round(float(np.mean(calideces)), 3)
        c["estabilidad_db"] = round(float(np.mean(estabs)), 2)
        c["sim"] = round(float(np.mean(sims)), 4)
        c["puntaje_total"] = round(c["sim"] + 0.04 * min(c["calidez"], 4.0) - 0.01 * c["estabilidad_db"], 4)
        evaluadas.append(c)

    del sv, fe
    gc.collect()
    torch.cuda.empty_cache()

    aptas = [c for c in evaluadas if c["wer"] <= WER_MAX]
    if not aptas:
        return 1

    mejor = max(aptas, key=lambda x: x["puntaje_total"])
    vigente = next((c for c in evaluadas if c["nombre"] == "vigente"), None)
    instalar = vigente is None or (mejor is not vigente and mejor["puntaje_total"] > vigente["puntaje_total"] + MARGEN_MEJORA)

    if instalar:
        if pt_vigente.exists():
            ts = time.strftime("%Y%m%d-%H%M%S")
            shutil.copy2(pt_vigente, out_dir / f"{registro}_bak_{ts}.pt")
        shutil.copy2(TRABAJO_DIR / f"cand_{mejor['idx']}.pt", pt_vigente)
        shutil.copy2(TRABAJO_DIR / f"gen_{mejor['idx']}_0.wav", out_dir / f"{registro}_muestra.wav")

    with HISTORIAL_CSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["fecha", "narrador", "registro", "ganadora", "sim", "wer", "calidez", "estabilidad_db", "instalada"])
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M"),
            narrador,
            registro,
            mejor["nombre"],
            mejor["sim"],
            mejor["wer"],
            mejor["calidez"],
            mejor["estabilidad_db"],
            "si" if instalar else "no",
        ])
    return 0


def main():
    parser = argparse.ArgumentParser(description="Nocturnal Storytelling Voice Trainer — ROANSO SpA")
    parser.add_argument("--narrador", default="narrador_sur", help="Narrator profile slug (e.g. narrador_sur, abuela_calma)")
    parser.add_argument("--registro", default="desarrollo_sereno", choices=REGISTROS_VALIDOS)
    parser.add_argument("--verificar-consentimiento", nargs=2, metavar=("AUDIO", "ADULT_NAME"))
    args = parser.parse_args()

    if args.verificar_consentimiento:
        apath, nombre = Path(args.verificar_consentimiento[0]), args.verificar_consentimiento[1]
        ok, info = verificar_consentimiento_hablado(apath, nombre)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        sys.exit(0 if ok else 1)

    sys.exit(entrenar_registro(args.narrador, args.registro))


if __name__ == "__main__":
    main()
