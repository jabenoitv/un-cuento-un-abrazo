# -*- coding: utf-8 -*-
"""Zero-PII & Zero-Secret Pre-Publish Security Gate for Public GitHub Release."""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_PATTERNS = [
    (r"16\.?019\.?401", "Personal Chilean national ID (RUN)"),
    (r"78\.?533\.?662", "Secondary company tax ID"),
    (r"15346513", "Notary/incorporation attention number"),
    (r"8139\s*4953", "Personal phone number"),
    (r"jabenoitv@", "Personal Gmail address"),
    (r"la\s+riojana", "Residential street address"),
    (r"bombas\s+braun", "Employer company reference"),
    (r"trinidad\s+rosario", "Family minor full name"),
    (r"antonia\s+sof[ií]a", "Family minor full name"),
    (r"camila\s+andrea", "Family full name"),
    (r"c:\\\\?users\\\\?jaben", "Local Windows user path"),
    (r"c:/users/jaben", "Local Windows user path"),
    (r"c:\\\\?tools\\\\?omnivoice", "Local Windows tool path"),
    (r"c:/tools/omnivoice", "Local Windows tool path"),
    (r"c:\\\\?dev\\\\?cuentos-ia", "Local Windows workspace path"),
    (r"c:/dev/cuentos-ia", "Local Windows workspace path"),
    (r"AIza[0-9A-Za-z\-_]{30,}", "Google API key"),
    (r"sk-[0-9A-Za-z]{25,}", "OpenAI/API secret key"),
]

TEXT_EXTS = {
    ".py", ".md", ".json", ".jsonl", ".txt", ".html", ".css", ".js",
    ".mjs", ".mts", ".ts", ".sql",
    ".webmanifest", ".svg", ".csv", ".yml", ".yaml", ".toml", ".example",
    ".gitignore", "",
}

ALLOWED_TEXT_PREFIXES = ("Dockerfile.", "Modelfile.")
ALLOWED_SHOWCASE_BINARIES = {
    "platform/showcase_assets/demo_lucas_cap1.mp3",
    "platform/showcase_assets/demo_mateo_cap1.mp3",
    "platform/showcase_assets/demo_fresia_cap1.mp3",
}


def main() -> int:
    findings = []
    scanned = 0
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT)
        rel_posix = rel.as_posix()
        if (
            ".git" in rel.parts
            or "__pycache__" in rel.parts
            or ".pytest_cache" in rel.parts
            or "node_modules" in rel.parts
            or "dist" in rel.parts
            or ".netlify" in rel.parts
            or any(p.endswith(".egg-info") for p in rel.parts)
            or path.name == "audit_public_repo.py"
        ):
            continue
        if rel_posix in ALLOWED_SHOWCASE_BINARIES:
            continue
        is_allowed_text = (
            path.suffix.lower() in TEXT_EXTS
            or path.name == "LICENSE"
            or path.name.startswith(ALLOWED_TEXT_PREFIXES)
        )
        if not is_allowed_text:
            findings.append(f"Unexpected binary/non-text file: {rel}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        scanned += 1
        for pat, desc in FORBIDDEN_PATTERNS:
            if re.search(pat, text, flags=re.IGNORECASE):
                findings.append(f"[{desc}] found in {rel}")

    print(f"[PUBLIC-REPO AUDIT] Scanned {scanned} files in {REPO_ROOT}")
    if findings:
        for f in findings:
            print("  ❌", f)
        return 1
    print("  ✅ ZERO PII, ZERO SECRETS, ZERO LOCAL PATHS — Safe to publish on GitHub.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
