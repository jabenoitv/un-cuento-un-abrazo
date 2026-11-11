# Open Hardware Reference Specification — 6"–7" Screen-Safe E-Ink Bedside Story Reader

* **License:** `CERN-OHL-P-2.0` (CERN Open Hardware Licence Version 2 — Permissive)
* **Maintainer:** ROANSO SpA (Rancagua, O'Higgins Region, Chile) — *Un Cuento, Un Abrazo* Open Public Goods Initiative
* **Purpose:** Provide an open, repairable, distraction-free bedside audio and e-ink reading specification so families, public-school special-education programs (**PIE**), and local hardware co-ops can deploy AI-verified bedtime stories **without smartphones, glowing LCD/OLED screens, cameras, microphones, or browsers in a child's bedroom**.

---

## 1. Design Philosophy (Human-Scale Hardware)
1. **Zero Surveillance & Zero Dopamine Traps:** Strictly **no camera**, **no microphone**, **no web browser**, **no app store**, and **no games**.
2. **Circadian & Sensory Safety:** Reflective E-Ink Carta display (zero blue-light flicker) paired with an optional adjustable warm-amber frontlight (`2200K–2700K`) and physical tactile buttons suitable for neurodivergent learners.
3. **Right to Repair:** User-replaceable compact flat LiPo pouch battery via rear screw/latch door (avoiding glued disposable electronics) + standard Classic Bluetooth A2DP fallback so the reader remains usable for years even if the internal speaker wears out.

---

## 2. Technical Requirements Matrix

| Subsystem | Open Reference Requirement |
|---|---|
| **Display** | 6" to 7" E-Ink Carta B/W (`1024×758`, `300 PPI` preferred; optional Kaleido 3 color variant). Warm-amber frontlight (`0–100%` hardware/PWM-free DC dimming). |
| **Internal Audio** | Built-in `1.5 W – 2.0 W` warm-voiced speaker tuned for human speech (`120 Hz – 8 kHz`, smooth high-frequency roll-off without harsh sibilance). |
| **Wireless Audio** | **Classic Bluetooth A2DP** source profile for external headphones or bedside speakers (BLE-only SoCs without A2DP audio streaming are excluded). |
| **Sync Connectivity** | `2.4 GHz` Wi-Fi (`802.11 b/g/n`) used strictly for periodic background sync of parent-approved encrypted story packages (`JSON + MP3/Opus`); hardware/menu airplane toggle for 100% offline bedtime operation. |
| **Local Storage** | `>= 4 GB – 8 GB` eMMC / SPI NAND flash (stores `100+` verified offline audio chapters + illustrations locally). |
| **Repairable Battery** | **User-replaceable flat LiPo pouch pack** via rear service cover with polarized JST/spring contacts (`~1,500–2,500 mAh`), `USB-C` charging (`5V/1A`), targeting `>= 7 nights` per charge (`~30 min/night` audio + page turns). |
| **Physical Controls** | Tactile physical buttons: `Play/Pause`, `Previous`, `Next`, `Volume +`, `Volume -`, `Ritual Repeat`, and `Power` (**zero microphone, zero camera**). |
| **Enclosure** | Anodized aluminum alloy or high-impact rugged `ABS/PC + TPU` bumper; flat bezel/back friendly for children's tactile stickers. |
| **Safety & Compliance** | `CE (RED)`, `FCC`, `RoHS`, `UN38.3 / MSDS` battery transport certification, and `EN 71 / ASTM F963` child-safe materials compliance. |

---

## 3. Open Firmware Architecture (`ROANSO E-Ink OS` Target)
- Minimal locked Linux (`Buildroot`) or RTOS image booting directly into the child's offline bookshelf in `< 4 seconds`.
- Local verification check ensuring every synced `.mp3`/`.opus` bundle carries a signed manifest from the parent-approved `pipeline_multinino` + `Whisper` QA harness.
- Native support for **Ritual Re-Listening Loop** and **0.85x / 0.90x / 1.0x** acoustic playback pacing.
