# -*- coding: utf-8 -*-
"""Automated Verification Suite — Human-Scale AI Storytelling Engine (ROANSO SpA)."""
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "motor"))

from motor_local_human_scale import (  # noqa: E402
    generar_auditoria_mozilla,
    verificar_anti_regurgitacion_ngramas,
    verificar_soberania_human_scale,
)
from pipeline_multinino import (  # noqa: E402
    analizar_cuento_multinino,
    calcular_legibilidad_es,
    cargar_biblia,
    generar_guia_padres,
    sanitizar_onboarding_ley21719,
)


class HumanScaleStorytellingTests(unittest.TestCase):
    def test_01_frontier_apis_blocked(self):
        for host in (
            "https://generativelanguage.googleapis.com/v1beta/models",
            "https://api.openai.com/v1/chat/completions",
            "https://api.anthropic.com/v1/messages",
        ):
            with self.assertRaises(PermissionError):
                verificar_soberania_human_scale(host, param_billions=7.6)

    def test_02_parameter_ceiling_enforced(self):
        with self.assertRaises(ValueError):
            verificar_soberania_human_scale("http://127.0.0.1:11434/api/generate", param_billions=14.0)
        # Local <= 10B succeeds
        verificar_soberania_human_scale("http://127.0.0.1:11434/api/generate", param_billions=7.6)

    def test_03_mozilla_audit_manifest(self):
        audit = generar_auditoria_mozilla()
        self.assertFalse(audit["screening_mozilla"]["core_depends_on_frontier_api"])
        self.assertLessEqual(audit["screening_mozilla"]["largest_model_parameter_count_billions"], 10.0)
        self.assertEqual(audit["screening_mozilla"]["mozilla_dropdown_bucket"], "10 billion or fewer")

    def test_04_ley21719_clinical_sanitizer(self):
        limpio, detectados = sanitizar_onboarding_ley21719(
            "Niño con TEA y TDAH, toma metilfenidato, evitar ruidos fuertes y truenos"
        )
        self.assertIn("tea", detectados)
        self.assertIn("tdah", detectados)
        self.assertIn("metilfenidato", detectados)
        self.assertNotIn("metilfenidato", limpio.lower())
        self.assertIn("ruidos fuertes", limpio.lower())

    def test_05_carol_gray_blocks_second_person_directives(self):
        biblia = cargar_biblia("demo_lucas")
        bad = " ".join(["Lucas miró el tren azul."] * 55) + " Tú debes obedecer siempre. Tu papá y tu mamá te quieren mucho."
        res = analizar_cuento_multinino(bad, biblia)
        self.assertFalse(res["valido"])
        self.assertTrue(any("Carol Gray 10.2" in e for e in res["errores"]))

    def test_06_blocks_ai_moralizing_sermons(self):
        biblia = cargar_biblia("demo_mateo")
        bad = " ".join(["Mateo miró la locomotora suave."] * 35) + " Y así aprendió una valiosa lección. Tu mamá y tu papá te aman."
        res = analizar_cuento_multinino(bad, biblia)
        self.assertFalse(res["valido"])
        self.assertTrue(any("moralizante" in e for e in res["errores"]))

    def test_07_literal_mode_catches_unexplained_idioms(self):
        biblia = cargar_biblia("demo_lucas")
        unexplained = " ".join(["Lucas ordenó la estación tibia."] * 60) + " Afuera llovía a cántaros sobre el techo. Tu papá y tu mamá te quieren mucho."
        res_bad = analizar_cuento_multinino(unexplained, biblia)
        self.assertFalse(res_bad["valido"])

        explained = (
            " ".join(["Lucas ordenó la estación tibia."] * 60)
            + " Afuera llovía a cántaros, eso quiere decir que caía mucha agua seguida sobre el techo. Tu papá y tu mamá te quieren mucho."
        )
        res_ok = analizar_cuento_multinino(explained, biblia)
        self.assertFalse(any("Modismo figurado" in e for e in res_ok["errores"]))

    def test_08_ritual_mode_blocks_anxious_cliffhangers(self):
        biblia = cargar_biblia("demo_lucas")
        cliff = " ".join(["Lucas encendió la lámpara tibia."] * 60) + " Algo terrible estaba por ocurrir. Tu papá y tu mamá te quieren mucho."
        res = analizar_cuento_multinino(cliff, biblia)
        self.assertFalse(res["valido"])
        self.assertTrue(any("suspenso angustiante" in e for e in res["errores"]))

    def test_09_anti_regurgitation_10gram_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            (tdir / "ref.txt").write_text(
                "Había una vez en las montañas nevadas de la cordillera un pequeño zorro culpeo curioso.",
                encoding="utf-8",
            )
            ok_copy, cols = verificar_anti_regurgitacion_ngramas(
                "Hoy vimos que en las montañas nevadas de la cordillera un pequeño zorro culpeo curioso caminaba.",
                corpus_dir=tdir,
                n=10,
            )
            self.assertFalse(ok_copy)
            self.assertEqual(len(cols), 1)

    def test_10_all_three_archetype_demos_pass_validation(self):
        for pid in ("demo_lucas", "demo_mateo", "demo_fresia"):
            biblia = cargar_biblia(pid)
            cap = (REPO_ROOT / "motor" / "capitulos" / f"{pid}_cap1.txt").read_text(encoding="utf-8")
            rep = analizar_cuento_multinino(cap, biblia)
            self.assertTrue(rep["valido"], f"{pid} failed: {rep['errores']}")
            self.assertGreaterEqual(rep["legibilidad"]["szigriszt_pazos"], 55.0)
            guia = generar_guia_padres(cap, biblia, "Cap 1")
            self.assertEqual(len(guia["preguntas"]), 3)


if __name__ == "__main__":
    unittest.main()
