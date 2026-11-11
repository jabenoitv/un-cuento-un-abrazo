/* Cliente PWA Familiar & Vitrina Auditada — Un Cuento, Un Abrazo (ROANSO SpA) */
const RoansoApp = (() => {
  let showcaseArchetypes = [];
  let selectedArchetype = null;
  let einkVisible = false;
  let jwtToken = sessionStorage.getItem("roanso_jwt") || "";
  let currentJobId = "";

  function toast(msg) {
    const el = document.getElementById("globalToast");
    el.textContent = msg;
    el.style.display = "block";
    setTimeout(() => { el.style.display = "none"; }, 5500);
  }

  async function registerServiceWorker() {
    if ("serviceWorker" in navigator) {
      try {
        await navigator.serviceWorker.register("/sw.js");
        navigator.serviceWorker.addEventListener("message", (ev) => {
          if (ev.data && ev.data.type === "AUDIO_CACHED_OK") {
            toast("Capítulo guardado en memoria Offline: listo para escuchar en Modo Avión sin Wi-Fi.");
            document.getElementById("bedsideOfflineBadge").textContent = "Disponible Sin Internet (Modo Avión)";
          }
        });
      } catch (_err) {
        // Ignorar en contextos sin HTTPS local
      }
    }
  }

  function switchMode(mode) {
    document.querySelectorAll(".pane").forEach((p) => p.classList.remove("active"));
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.getElementById("mainHeader").style.display = "flex";

    if (mode === "showcase") {
      document.getElementById("paneShowcase").classList.add("active");
      document.getElementById("tabShowcase").classList.add("active");
    } else if (mode === "parent") {
      document.getElementById("paneParent").classList.add("active");
      document.getElementById("tabParent").classList.add("active");
      refreshParentPortal();
    }
  }

  function toggleAmber() {
    document.body.classList.toggle("amber-sleep");
  }

  function toggleEinkView() {
    einkVisible = !einkVisible;
    document.getElementById("storyStandardView").style.display = einkVisible ? "none" : "block";
    document.getElementById("storyEinkView").style.display = einkVisible ? "block" : "none";
  }

  async function loadShowcase() {
    const [archRes, voiceRes] = await Promise.all([
      fetch("/v1/showcase/archetypes"),
      fetch("/v1/showcase/voices")
    ]);
    const archData = await archRes.json();
    const voiceData = await voiceRes.json();

    showcaseArchetypes = archData.archetypes || [];
    const listEl = document.getElementById("archetypeList");
    listEl.innerHTML = "";

    showcaseArchetypes.forEach((item, idx) => {
      const card = document.createElement("div");
      card.className = "archetype-card" + (idx === 0 ? " selected" : "");
      card.dataset.pid = item.profile_id;
      card.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <strong>${item.child_alias} (${item.age_years} años)</strong>
          <span class="badge ok">WER ${(item.acoustic_qa.whisper_wer * 100).toFixed(2)}%</span>
        </div>
        <div style="font-size:0.78rem;color:var(--text-muted);margin-top:4px;">
          ${item.sensory_axes.lenguaje} · ${item.sensory_axes.estimulo_sonoro}
        </div>
      `;
      card.onclick = () => selectArchetype(item.profile_id);
      listEl.appendChild(card);
    });

    const vList = document.getElementById("ethicalVoiceList");
    vList.innerHTML = (voiceData.voices || []).map((v) =>
      `<div style="margin-bottom:8px;">
        <strong style="color:var(--text);">${v.display_name}</strong><br/>
        <span>Voice Design Apache-2.0 · Centroide ${v.spectral_centroid_hz} Hz · Cero clonación de terceros</span>
      </div>`
    ).join("");

    if (showcaseArchetypes.length > 0) {
      selectArchetype(showcaseArchetypes[0].profile_id);
    }
  }

  function selectArchetype(profileId) {
    const item = showcaseArchetypes.find((a) => a.profile_id === profileId);
    if (!item) return;
    selectedArchetype = item;

    document.querySelectorAll(".archetype-card").forEach((el) => {
      el.classList.toggle("selected", el.dataset.pid === profileId);
    });

    document.getElementById("showcaseTitle").textContent = item.series_title;
    document.getElementById("showcaseMeta").textContent =
      `${item.child_alias} (${item.age_years} años) · ${item.word_count} palabras · Legibilidad Szigriszt-Pazos (INFLESZ): ${item.readability.szigriszt_pazos} (${item.readability.nivel_lectura})`;

    const aq = item.acoustic_qa;
    document.getElementById("showcaseBadges").innerHTML = `
      <span class="badge accent">${item.sensory_axes.lenguaje}</span>
      <span class="badge accent">${item.sensory_axes.estructura}</span>
      <span class="badge accent">${item.sensory_axes.estimulo_sonoro}</span>
      <span class="badge ok">Precisión Whisper: ${aq.whisper_accuracy_pct}%</span>
      <span class="badge ok">Ley 21.719: Cero Diagnósticos</span>
    `;

    document.getElementById("audioVoiceLabel").textContent =
      `Voz: ${item.voice_display_name} · ${(aq.duration_seconds / 60).toFixed(1)} min (${aq.words_per_minute} WPM)`;
    document.getElementById("showcaseAudio").src = item.audio_stream_url;
    document.getElementById("acousticTelemetryBar").textContent =
      `Telemetría DSP: Atenuación Sleep Ramp = ${aq.sleep_ramp_attenuation_db} dB · Centroide cálido = ${aq.spectral_centroid_hz} Hz · SHA-256 MP3 = ${item.mp3_sha256.slice(0, 12)}...`;

    document.getElementById("storyStandardView").textContent = item.story_text;
    document.getElementById("einkTextBody").textContent = item.story_text;

    const guide = (item.parent_guide && item.parent_guide.preguntas) || [];
    document.getElementById("parentGuideList").innerHTML = guide.map((q) => `
      <div class="q-card">
        <strong>${q.tipo || "Conversación"}</strong>
        <div>${q.pregunta}</div>
        <div style="font-size:0.77rem;color:var(--text-muted);margin-top:4px;">Apoyo sugerido: ${q.pista_apoyo || ""}</div>
      </div>
    `).join("");
  }

  function cacheCurrentShowcaseOffline() {
    if (!selectedArchetype) return;
    if (navigator.serviceWorker && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: "CACHE_AUDIO_OFFLINE",
        audioUrl: selectedArchetype.audio_stream_url
      });
    } else {
      fetch(selectedArchetype.audio_stream_url).then(() => {
        toast("Audio precargado en caché local del navegador.");
      });
    }
  }

  function enterChildBedsideMode() {
    document.body.classList.add("amber-sleep");
    document.getElementById("mainHeader").style.display = "none";
    document.querySelectorAll(".pane").forEach((p) => p.classList.remove("active"));
    document.getElementById("paneBedside").classList.add("active");

    if (selectedArchetype) {
      document.getElementById("bedsideStoryTitle").textContent = selectedArchetype.series_title;
      document.getElementById("bedsideSubtitle").textContent =
        `${selectedArchetype.child_alias} · Modo ${selectedArchetype.sensory_axes.estimulo_sonoro}`;
      const audioEl = document.getElementById("bedsideAudio");
      if (!audioEl.src.endsWith(selectedArchetype.audio_stream_url)) {
        audioEl.src = selectedArchetype.audio_stream_url;
      }
    }
  }

  function toggleBedsidePlayback() {
    const audioEl = document.getElementById("bedsideAudio");
    const btn = document.getElementById("btnGiantPlay");
    if (audioEl.paused) {
      audioEl.play();
      btn.textContent = "Pausar";
    } else {
      audioEl.pause();
      btn.textContent = "Escuchar";
    }
    audioEl.onended = () => { btn.textContent = "Buenas noches"; };
  }

  async function exitChildBedsideMode() {
    const pin = document.getElementById("bedsidePinInput").value.trim();
    document.getElementById("bedsidePinInput").value = "";
    const audioEl = document.getElementById("bedsideAudio");
    audioEl.pause();

    if (jwtToken) {
      const res = await fetch("/v1/auth/verify-pin", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${jwtToken}` },
        body: JSON.stringify({ parental_pin: pin })
      });
      if (!res.ok) {
        alert("PIN parental incorrecto.");
        return;
      }
      switchMode("parent");
      return;
    }

    if (pin === "0000") {
      switchMode("showcase");
    } else {
      alert("En modo demostración ingresa el PIN 0000 para volver a la vitrina.");
    }
  }

  async function apiAuth(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error de autenticación");
    jwtToken = data.access_token;
    sessionStorage.setItem("roanso_jwt", jwtToken);
    return data;
  }

  async function registerFamily() {
    try {
      await apiAuth("/v1/auth/register", {
        family_name: document.getElementById("regFamilyName").value,
        full_name: document.getElementById("regFullName").value,
        email: document.getElementById("regEmail").value,
        password: document.getElementById("regPassword").value,
        parental_pin: document.getElementById("regPin").value,
        plan_tier: "familiar_voces",
        accepted_ley21719_notice: true
      });
      toast("Cuenta familiar creada con protección Ley 21.719.");
      refreshParentPortal();
    } catch (err) {
      alert(err.message);
    }
  }

  async function loginFamily() {
    try {
      await apiAuth("/v1/auth/login", {
        email: document.getElementById("loginEmail").value,
        password: document.getElementById("loginPassword").value
      });
      toast("Sesión familiar iniciada.");
      refreshParentPortal();
    } catch (err) {
      alert(err.message);
    }
  }

  function logout() {
    jwtToken = "";
    sessionStorage.removeItem("roanso_jwt");
    refreshParentPortal();
  }

  async function refreshParentPortal() {
    const gate = document.getElementById("authGateSection");
    const dash = document.getElementById("familyDashboardSection");
    if (!jwtToken) {
      gate.style.display = "grid";
      dash.style.display = "none";
      return;
    }
    gate.style.display = "none";
    dash.style.display = "block";

    const headers = { Authorization: `Bearer ${jwtToken}` };
    const [chRes, vRes] = await Promise.all([
      fetch("/v1/children", { headers }),
      fetch("/v1/voices", { headers })
    ]);
    if (chRes.status === 401) {
      logout();
      return;
    }
    const children = await chRes.json();
    const voices = await vRes.json();

    document.getElementById("childrenList").innerHTML = children.map((c) => `
      <div class="q-card">
        <strong>${c.alias} (${c.age_years} años) · Bloqueos clínicos Ley 21.719: ${c.sanitized_terms_blocked_count}</strong>
        <div style="font-size:0.78rem;">${c.perfil_sensorial.lenguaje} · ${c.perfil_sensorial.estructura} · ${c.perfil_sensorial.estimulo_sonoro}</div>
      </div>
    `).join("");

    const chSel = document.getElementById("jobChildSelect");
    chSel.innerHTML = children.map((c) => `<option value="${c.id}">${c.alias} (${c.age_years}a)</option>`).join("");
    chSel.onchange = () => loadChildChapters(chSel.value);
    if (children.length > 0) loadChildChapters(children[0].id);

    const activeVoices = voices.filter((v) => v.status === "ACTIVE");
    document.getElementById("jobVoiceSelect").innerHTML = activeVoices.map((v) =>
      `<option value="${v.id}">${v.display_name}</option>`
    ).join("");

    document.getElementById("familyVoicesBox").innerHTML = voices.map((v) => `
      <div class="q-card" style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <strong>${v.display_name}</strong>
          <span style="font-size:0.76rem;color:var(--text-muted);">${v.is_catalog ? "Catálogo Ético" : v.status}</span>
        </div>
        ${(!v.is_catalog && v.status === "ACTIVE")
          ? `<button class="btn btn-danger" onclick="RoansoApp.revokeVoice('${v.id}')">Destruir Voz</button>`
          : ""}
      </div>
    `).join("");
  }

  async function createChild() {
    const interests = document.getElementById("childInterests").value
      .split(",").map((s) => s.trim()).filter(Boolean);
    const res = await fetch("/v1/children", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${jwtToken}` },
      body: JSON.stringify({
        alias: document.getElementById("childAlias").value,
        age_years: parseInt(document.getElementById("childAge").value, 10) || 7,
        perfil_sensorial: {
          lenguaje: document.getElementById("axisLenguaje").value,
          estructura: document.getElementById("axisEstructura").value,
          estimulo_sonoro: document.getElementById("axisSonoro").value,
          foco_tematico: "interes_profundo"
        },
        intereses_especiales: interests.length ? interests : ["estrellas del sur"],
        valores_trabajar: ["calma", "empatía mutua"],
        temas_evitar: [],
        closing_phrase: document.getElementById("childClosing").value,
        notas_libres_padre: document.getElementById("childNotes").value
      })
    });
    const data = await res.json();
    if (!res.ok) {
      alert(JSON.stringify(data.detail));
      return;
    }
    toast(`Perfil de ${data.alias} guardado (términos clínicos omitidos por Ley 21.719: ${data.sanitized_terms_blocked_count}).`);
    refreshParentPortal();
  }

  async function enqueueJob() {
    const childId = document.getElementById("jobChildSelect").value;
    const voiceId = document.getElementById("jobVoiceSelect").value;
    const seed = document.getElementById("jobSeed").value.trim();
    if (!childId || !seed) {
      alert("Selecciona un perfil e ingresa la semilla cotidiana de hoy.");
      return;
    }
    const res = await fetch("/v1/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${jwtToken}` },
      body: JSON.stringify({ child_id: childId, voice_id: voiceId, parent_seed: seed })
    });
    const job = await res.json();
    if (!res.ok) {
      alert(job.detail || "No fue posible encolar el cuento.");
      return;
    }
    currentJobId = job.id;
    document.getElementById("activeJobBox").textContent =
      `Cuento #${job.chapter_number} en cola elástica GPU (${job.status}: ${job.stage_detail}).`;
    toast("Cuento encolado para procesamiento asíncrono en nodo GPU.");
  }

  async function loadChildChapters(childId) {
    if (!childId || !jwtToken) return;
    const res = await fetch(`/v1/children/${childId}/chapters`, {
      headers: { Authorization: `Bearer ${jwtToken}` }
    });
    if (!res.ok) return;
    const items = await res.json();
    const box = document.getElementById("familyChaptersBox");
    box.innerHTML = items.map((ch) => {
      const streamUrl = ch.audio_url.startsWith("/v1/media/")
        ? `${ch.audio_url}?token=${encodeURIComponent(jwtToken)}`
        : ch.audio_url;
      return `
        <div class="q-card">
          <strong>Cap. ${ch.chapter_number}: ${ch.title} (Szigriszt ${ch.readability.szigriszt_pazos} · WER ${(ch.acoustic_qa.whisper_wer * 100).toFixed(2)}%)</strong>
          <audio controls preload="none" src="${streamUrl}" style="width:100%;margin-top:6px;"></audio>
        </div>
      `;
    }).join("");
  }

  async function enrollAdultVoice() {
    const displayName = (document.getElementById("voiceDisplayName").value || "").trim() || "Voz Adulta Familiar";
    const consentText = (document.getElementById("voiceConsentText").value || "").trim();
    const demoVectorBytes = new Uint8Array(64);
    if (window.crypto && window.crypto.getRandomValues) {
      window.crypto.getRandomValues(demoVectorBytes);
    }
    let bin = "";
    demoVectorBytes.forEach((b) => { bin += String.fromCharCode(b); });
    const embeddingB64 = btoa(bin);

    const res = await fetch("/v1/voices/parental", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${jwtToken}` },
      body: JSON.stringify({
        display_name: displayName,
        adult_full_name: "Adulto Responsable Verificado",
        spoken_consent_transcript: consentText,
        is_minor_voice_declaration_false: true,
        embedding_b64: embeddingB64
      })
    });
    const data = await res.json();
    if (!res.ok) {
      alert(JSON.stringify(data.detail || data));
      return;
    }
    toast(`Timbre "${data.display_name}" cifrado con AES-256-GCM + HKDF en bóveda privada.`);
    refreshParentPortal();
  }

  async function revokeVoice(voiceId) {
    await fetch(`/v1/voices/${voiceId}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${jwtToken}` }
    });
    toast("Voz parental revocada y destruida criptográficamente (HKDF Shred).");
    refreshParentPortal();
  }

  async function exportFamilyData() {
    const res = await fetch("/v1/families/me/export", {
      headers: { Authorization: `Bearer ${jwtToken}` }
    });
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "roanso_portabilidad_ley21719.json";
    a.click();
  }

  async function eraseFamilyAccount() {
    if (!confirm("¿Confirmas eliminar permanentemente toda la cuenta familiar, voces y audios (Derecho de Supresión Ley 21.719)?")) {
      return;
    }
    await fetch("/v1/families/me", {
      method: "DELETE",
      headers: { Authorization: `Bearer ${jwtToken}` }
    });
    logout();
    toast("Cuenta familiar, voces y capítulos eliminados íntegramente.");
  }

  document.addEventListener("DOMContentLoaded", () => {
    registerServiceWorker();
    loadShowcase();
  });

  return {
    switchMode,
    toggleAmber,
    toggleEinkView,
    cacheCurrentShowcaseOffline,
    enterChildBedsideMode,
    toggleBedsidePlayback,
    exitChildBedsideMode,
    registerFamily,
    loginFamily,
    logout,
    createChild,
    enqueueJob,
    enrollAdultVoice,
    revokeVoice,
    exportFamilyData,
    eraseFamilyAccount
  };
})();
