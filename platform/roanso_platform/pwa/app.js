/* Aplicación Familiar — Un Cuento, Un Abrazo */
const RoansoApp = (() => {
  let showcaseArchetypes = [];
  let catalogVoices = [];
  let selectedArchetype = null;
  let currentStoryMode = "leer_imaginar";
  let currentSceneIdx = 0;
  let jwtToken = sessionStorage.getItem("roanso_jwt") || "";

  const FRIENDLY_WORDS = {
    literal_claro: "Claro y directo",
    agil_visual: "Ágil y entretenido",
    literario_rico: "Imaginativo y cálido"
  };

  const FRIENDLY_ORDER = {
    ritual_anticipado: "Rutina tranquila",
    enganche_rampa_calma: "Inicio ágil y cierre suave",
    serie_multiverso: "Colección por capítulos"
  };

  const FRIENDLY_PACE = {
    monovoz_estable: "Lectura pausada y suave",
    dinamico_a_calmo: "Expresiva y luego muy suave"
  };

  const STORY_THEMES = {
    demo_lucas: "Amistad, límites amables y entender intenciones",
    demo_mateo: "Esperar turnos, empatía y ponerse de acuerdo",
    demo_fresia: "Pedir ayuda con confianza y trabajo en equipo"
  };

  const NARRATOR_FRIENDLY_NAMES = {
    "catalog-chilena-calida-f1": {
      name: "Sofía · Valle Central",
      accentLabel: "Español Chile - Cálido",
      desc: "Voz femenina cercana y serena, ideal para bajar el ritmo antes de dormir."
    },
    "catalog-chileno-sereno-m1": {
      name: "Martín · Sur de Chile",
      accentLabel: "Español Chile - Cálido",
      desc: "Voz masculina grave y pausada, con entonación chilena natural."
    }
  };

  /* Ilustraciones estilo tinta a color y acuarela suave (3 escenas pausadas por cuento) */
  const STORY_SCENES = {
    demo_lucas: [
      {
        caption: "Escena 1 · Los faroles cálidos de la estación de Quillay Alto junto a la perrita Canela.",
        svg: `<svg viewBox="0 0 640 420" xmlns="http://www.w3.org/2000/svg">
          <rect width="640" height="420" fill="#f9f3e6"/>
          <circle cx="515" cy="88" r="46" fill="#f6c97a" opacity="0.55"/>
          <path d="M0 300 Q180 265 360 295 T640 280 L640 420 L0 420 Z" fill="#d9e4d7"/>
          <rect x="85" y="165" width="250" height="145" rx="6" fill="#ead4b4" stroke="#3c3228" stroke-width="2.4"/>
          <polygon points="70,170 210,105 350,170" fill="#c97b56" stroke="#3c3228" stroke-width="2.4"/>
          <rect x="130" y="210" width="65" height="52" rx="4" fill="#f7c873" stroke="#3c3228" stroke-width="2"/>
          <rect x="230" y="205" width="52" height="105" rx="4" fill="#8d6346" stroke="#3c3228" stroke-width="2"/>
          <line x1="420" y1="145" x2="420" y2="310" stroke="#3c3228" stroke-width="4"/>
          <circle cx="420" cy="140" r="18" fill="#f4a261" stroke="#3c3228" stroke-width="2.2"/>
          <ellipse cx="475" cy="312" rx="28" ry="15" fill="#d49a6a" stroke="#3c3228" stroke-width="2"/>
          <circle cx="498" cy="298" r="12" fill="#d49a6a" stroke="#3c3228" stroke-width="2"/>
          <line x1="0" y1="358" x2="640" y2="358" stroke="#4a4036" stroke-width="3"/>
          <line x1="0" y1="374" x2="640" y2="374" stroke="#4a4036" stroke-width="3"/>
        </svg>`
      },
      {
        caption: "Escena 2 · Lucas y Tomás acuerdan hacer una seña suave con la linterna para turnarse.",
        svg: `<svg viewBox="0 0 640 420" xmlns="http://www.w3.org/2000/svg">
          <rect width="640" height="420" fill="#f8f1e3"/>
          <rect x="70" y="90" width="500" height="250" rx="14" fill="#efe2ca" stroke="#3c3228" stroke-width="2.2"/>
          <rect x="260" y="185" width="120" height="115" rx="8" fill="#b7c6b8" stroke="#3c3228" stroke-width="2.2"/>
          <line x1="300" y1="185" x2="285" y2="128" stroke="#9e6b43" stroke-width="5" stroke-linecap="round"/>
          <circle cx="285" cy="124" r="9" fill="#d98a4e" stroke="#3c3228" stroke-width="2"/>
          <circle cx="185" cy="195" r="24" fill="#f2cbb0" stroke="#3c3228" stroke-width="2.2"/>
          <path d="M160 185 Q185 162 212 185 Z" fill="#4f759b" stroke="#3c3228" stroke-width="2"/>
          <rect x="163" y="220" width="44" height="72" rx="10" fill="#5d83a8" stroke="#3c3228" stroke-width="2.2"/>
          <circle cx="445" cy="198" r="23" fill="#f2cbb0" stroke="#3c3228" stroke-width="2.2"/>
          <rect x="423" y="222" width="44" height="70" rx="10" fill="#7da37e" stroke="#3c3228" stroke-width="2.2"/>
          <polygon points="220,242 295,215 295,265" fill="#f7cf78" opacity="0.65"/>
          <circle cx="342" cy="142" r="14" fill="#6ab187" stroke="#3c3228" stroke-width="2"/>
        </svg>`
      },
      {
        caption: "Escena 3 · El tren azul descansa bajo techo mientras Canela duerme en su alfombra tibia.",
        svg: `<svg viewBox="0 0 640 420" xmlns="http://www.w3.org/2000/svg">
          <rect width="640" height="420" fill="#f5ede0"/>
          <path d="M90 315 L90 140 L320 88 L550 140 L550 315 Z" fill="#e5d5bc" stroke="#3c3228" stroke-width="2.4"/>
          <rect x="165" y="180" width="310" height="118" rx="18" fill="#52799e" stroke="#3c3228" stroke-width="2.5"/>
          <rect x="200" y="204" width="58" height="42" rx="6" fill="#f6d28b" stroke="#3c3228" stroke-width="2"/>
          <rect x="285" y="204" width="58" height="42" rx="6" fill="#f6d28b" stroke="#3c3228" stroke-width="2"/>
          <rect x="370" y="204" width="58" height="42" rx="6" fill="#f6d28b" stroke="#3c3228" stroke-width="2"/>
          <circle cx="220" cy="306" r="18" fill="#4a433c" stroke="#3c3228" stroke-width="2"/>
          <circle cx="420" cy="306" r="18" fill="#4a433c" stroke="#3c3228" stroke-width="2"/>
          <ellipse cx="320" cy="366" rx="75" ry="20" fill="#dfb178" stroke="#3c3228" stroke-width="1.8"/>
          <ellipse cx="320" cy="360" rx="34" ry="15" fill="#c98b5b" stroke="#3c3228" stroke-width="2"/>
        </svg>`
      }
    ],
    demo_mateo: [
      {
        caption: "Escena 1 · Mateo gira el reloj de arena dorada junto a la gata Luna en el Faro de los Inventos.",
        svg: `<svg viewBox="0 0 640 420" xmlns="http://www.w3.org/2000/svg">
          <rect width="640" height="420" fill="#f8f2e5"/>
          <circle cx="320" cy="185" r="135" fill="#efe2c8" stroke="#3c3228" stroke-width="2.2"/>
          <rect x="160" y="280" width="320" height="24" rx="5" fill="#b07d56" stroke="#3c3228" stroke-width="2.2"/>
          <polygon points="265,175 315,175 290,222" fill="#f3be68" stroke="#3c3228" stroke-width="2"/>
          <polygon points="265,268 315,268 290,222" fill="#f3be68" stroke="#3c3228" stroke-width="2"/>
          <circle cx="385" cy="242" r="24" fill="#d9a16c" stroke="#3c3228" stroke-width="2"/>
          <circle cx="430" cy="252" r="16" fill="#87a896" stroke="#3c3228" stroke-width="2"/>
          <ellipse cx="215" cy="262" rx="22" ry="16" fill="#a89078" stroke="#3c3228" stroke-width="2"/>
          <polygon points="200,246 207,232 215,246" fill="#a89078" stroke="#3c3228" stroke-width="1.8"/>
          <polygon points="216,246 223,232 230,246" fill="#a89078" stroke="#3c3228" stroke-width="1.8"/>
        </svg>`
      },
      {
        caption: "Escena 2 · Por el telescopio de bronce brillan los barquitos de papel sobre el estero calmo.",
        svg: `<svg viewBox="0 0 640 420" xmlns="http://www.w3.org/2000/svg">
          <rect width="640" height="420" fill="#f5efe4"/>
          <circle cx="320" cy="210" r="155" fill="#2c3e55" stroke="#8f6843" stroke-width="8"/>
          <path d="M168 245 Q260 225 350 248 T472 242 L455 345 L185 345 Z" fill="#46698a"/>
          <polygon points="245,242 275,205 305,242" fill="#fbf5e8" stroke="#3c3228" stroke-width="1.8"/>
          <polygon points="235,242 315,242 298,258 252,258" fill="#f5c978" stroke="#3c3228" stroke-width="1.8"/>
          <polygon points="355,255 378,225 401,255" fill="#fbf5e8" stroke="#3c3228" stroke-width="1.8"/>
          <polygon points="347,255 409,255 396,268 360,268" fill="#f5c978" stroke="#3c3228" stroke-width="1.8"/>
          <circle cx="245" cy="138" r="5" fill="#f9dc94"/>
          <circle cx="328" cy="115" r="6" fill="#f9dc94"/>
          <circle cx="395" cy="148" r="4.5" fill="#f9dc94"/>
        </svg>`
      },
      {
        caption: "Escena 3 · Engranajes guardados en su caja de cedro y Luna ronroneando en el sillón.",
        svg: `<svg viewBox="0 0 640 420" xmlns="http://www.w3.org/2000/svg">
          <rect width="640" height="420" fill="#f6efe2"/>
          <rect x="135" y="185" width="235" height="125" rx="28" fill="#c77d58" stroke="#3c3228" stroke-width="2.4"/>
          <rect x="160" y="238" width="185" height="68" rx="16" fill="#d9936e" stroke="#3c3228" stroke-width="2"/>
          <ellipse cx="252" cy="252" rx="34" ry="16" fill="#a89078" stroke="#3c3228" stroke-width="2"/>
          <rect x="420" y="235" width="110" height="72" rx="8" fill="#9c6f49" stroke="#3c3228" stroke-width="2.2"/>
          <line x1="420" y1="258" x2="530" y2="258" stroke="#3c3228" stroke-width="2"/>
          <circle cx="475" cy="175" r="24" fill="#f6cb79" opacity="0.5"/>
        </svg>`
      }
    ],
    demo_fresia: [
      {
        caption: "Escena 1 · Fresia y el gato Almirante miran las luces tibias de los palafitos en Caleta Niebla.",
        svg: `<svg viewBox="0 0 640 420" xmlns="http://www.w3.org/2000/svg">
          <rect width="640" height="420" fill="#f6f1e6"/>
          <rect x="0" y="285" width="640" height="135" fill="#9cb7c4"/>
          <line x1="125" y1="245" x2="125" y2="325" stroke="#4a3b2e" stroke-width="5"/>
          <line x1="215" y1="245" x2="215" y2="325" stroke="#4a3b2e" stroke-width="5"/>
          <rect x="105" y="155" width="130" height="95" rx="5" fill="#7faac4" stroke="#3c3228" stroke-width="2.2"/>
          <polygon points="95,158 170,110 245,158" fill="#b86b52" stroke="#3c3228" stroke-width="2.2"/>
          <rect x="150" y="185" width="40" height="34" rx="4" fill="#f7cc78" stroke="#3c3228" stroke-width="2"/>
          <line x1="295" y1="250" x2="295" y2="328" stroke="#4a3b2e" stroke-width="5"/>
          <line x1="385" y1="250" x2="385" y2="328" stroke="#4a3b2e" stroke-width="5"/>
          <rect x="275" y="165" width="130" height="90" rx="5" fill="#e9c77b" stroke="#3c3228" stroke-width="2.2"/>
          <polygon points="265,168 340,122 415,168" fill="#7c8f79" stroke="#3c3228" stroke-width="2.2"/>
          <rect x="322" y="192" width="36" height="32" rx="4" fill="#f7cc78" stroke="#3c3228" stroke-width="2"/>
          <ellipse cx="500" cy="272" rx="24" ry="13" fill="#8d8278" stroke="#3c3228" stroke-width="2"/>
        </svg>`
      },
      {
        caption: "Escena 2 · Los vecinos de la caleta reúnen con cariño las estrofas en el cuaderno azul.",
        svg: `<svg viewBox="0 0 640 420" xmlns="http://www.w3.org/2000/svg">
          <rect width="640" height="420" fill="#f8f2e4"/>
          <rect x="175" y="110" width="290" height="205" rx="10" fill="#5982a6" stroke="#3c3228" stroke-width="2.5"/>
          <rect x="192" y="124" width="128" height="177" rx="4" fill="#fcf8ef" stroke="#3c3228" stroke-width="1.6"/>
          <rect x="322" y="124" width="128" height="177" rx="4" fill="#fcf8ef" stroke="#3c3228" stroke-width="1.6"/>
          <line x1="212" y1="160" x2="300" y2="160" stroke="#6c7a89" stroke-width="2.2"/>
          <line x1="212" y1="186" x2="294" y2="186" stroke="#6c7a89" stroke-width="2.2"/>
          <line x1="212" y1="212" x2="302" y2="212" stroke="#6c7a89" stroke-width="2.2"/>
          <path d="M345 195 Q375 165 405 195 T435 190" fill="none" stroke="#d98a4e" stroke-width="3"/>
          <circle cx="355" cy="242" r="8" fill="#d98a4e"/>
          <circle cx="395" cy="232" r="8" fill="#5e9c82"/>
          <line x1="363" y1="242" x2="363" y2="212" stroke="#3c3228" stroke-width="2.2"/>
          <line x1="403" y1="232" x2="403" y2="202" stroke="#3c3228" stroke-width="2.2"/>
        </svg>`
      },
      {
        caption: "Escena 3 · La canción guía el bote del tío Lalo hasta el muelle en calma.",
        svg: `<svg viewBox="0 0 640 420" xmlns="http://www.w3.org/2000/svg">
          <rect width="640" height="420" fill="#f4eee2"/>
          <rect x="0" y="265" width="640" height="155" fill="#7d9dad"/>
          <polygon points="120,265 142,118 182,118 204,265" fill="#ebe3d2" stroke="#3c3228" stroke-width="2.4"/>
          <rect x="140" y="92" width="44" height="28" rx="4" fill="#f6c86e" stroke="#3c3228" stroke-width="2.2"/>
          <polygon points="184,106 470,185 470,255" fill="#f7d58b" opacity="0.38"/>
          <polygon points="395,295 495,295 476,322 414,322" fill="#a86848" stroke="#3c3228" stroke-width="2.2"/>
          <circle cx="445" cy="282" r="10" fill="#f9d978" stroke="#3c3228" stroke-width="1.8"/>
        </svg>`
      }
    ]
  };

  function toast(msg) {
    const el = document.getElementById("globalToast");
    el.textContent = msg;
    el.style.display = "block";
    setTimeout(() => { el.style.display = "none"; }, 5200);
  }

  async function registerServiceWorker() {
    if ("serviceWorker" in navigator) {
      try {
        await navigator.serviceWorker.register("/sw.js");
        navigator.serviceWorker.addEventListener("message", (ev) => {
          if (ev.data && ev.data.type === "AUDIO_CACHED_OK") {
            toast("Cuento guardado en tu teléfono: ya puedes escucharlo sin conexión a internet.");
            document.getElementById("bedsideOfflineBadge").textContent = "Listo para escuchar sin internet";
          }
        });
      } catch (_err) {
        // Ignorar en entornos locales sin HTTPS
      }
    }
  }

  function switchMode(mode) {
    document.querySelectorAll(".pane").forEach((p) => p.classList.remove("active"));
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.getElementById("mainHeader").style.display = "flex";
    document.getElementById("mainFooter").style.display = "flex";

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

  function openPrivacyModal() {
    document.getElementById("privacyModal").classList.add("open");
  }

  function closePrivacyModal() {
    document.getElementById("privacyModal").classList.remove("open");
  }

  function formatParagraphsHtml(rawText) {
    const sentences = (rawText || "").match(/[^.!?…]+[.!?…]+/g) || [rawText || ""];
    const chunks = [];
    for (let i = 0; i < sentences.length; i += 4) {
      chunks.push(sentences.slice(i, i + 4).join(" ").trim());
    }
    return chunks.filter(Boolean).map((p) => `<p>${p}</p>`).join("");
  }

  function setStoryMode(mode) {
    currentStoryMode = mode;
    const cardMap = {
      solo_escuchar: "modeCardSleep",
      leer_imaginar: "modeCardReadImagine",
      mama_papa_lee: "modeCardParentReads",
      ilustraciones_audio: "modeCardIllustratedAudio"
    };
    Object.entries(cardMap).forEach(([key, id]) => {
      document.getElementById(id).classList.toggle("active", key === mode);
    });

    const audioDock = document.getElementById("mainAudioDock");
    const audioEl = document.getElementById("showcaseAudio");

    document.getElementById("stageSoloEscuchar").style.display = mode === "solo_escuchar" ? "block" : "none";
    document.getElementById("stageLeerImaginar").style.display = mode === "leer_imaginar" ? "grid" : "none";
    document.getElementById("stageMamaPapaLee").style.display = mode === "mama_papa_lee" ? "block" : "none";
    document.getElementById("stageIlustracionesAudio").style.display = mode === "ilustraciones_audio" ? "block" : "none";

    if (mode === "mama_papa_lee") {
      audioEl.pause();
      audioDock.style.display = "none";
    } else {
      audioDock.style.display = "flex";
    }
  }

  function selectScene(idx) {
    if (!selectedArchetype) return;
    const scenes = STORY_SCENES[selectedArchetype.profile_id] || STORY_SCENES.demo_lucas;
    currentSceneIdx = Math.max(0, Math.min(idx, scenes.length - 1));
    const scene = scenes[currentSceneIdx];

    document.getElementById("inkSplitSvg").innerHTML = scene.svg;
    document.getElementById("inkSplitCaption").textContent = scene.caption;
    document.getElementById("inkFullSvg").innerHTML = scene.svg;
    document.getElementById("inkFullCaption").textContent = scene.caption;

    const buttonsHtml = scenes.map((_s, i) =>
      `<button class="scene-pill${i === currentSceneIdx ? " active" : ""}" onclick="RoansoApp.selectScene(${i})">Ilustración ${i + 1}</button>`
    ).join("");
    document.getElementById("inkSplitControls").innerHTML = buttonsHtml;
    document.getElementById("inkFullControls").innerHTML = buttonsHtml;
  }

  function filterNarratorsByAccent() {
    const accent = document.getElementById("narratorAccentSelect").value;
    const vList = document.getElementById("ethicalVoiceList");

    if (accent === "es-CL") {
      vList.innerHTML = catalogVoices.map((v) => {
        const meta = NARRATOR_FRIENDLY_NAMES[v.id] || {
          name: v.display_name,
          desc: "Lectura cálida y pausada en español de Chile."
        };
        const isCurrent = selectedArchetype && selectedArchetype.voice_id === v.id;
        return `<div class="narrator-item${isCurrent ? " active-voice" : ""}">
          <strong style="color:var(--text);display:block;">${meta.name}</strong>
          <span>${meta.desc}</span>
        </div>`;
      }).join("");
      return;
    }

    if (accent === "es-LA") {
      vList.innerHTML = `
        <div class="narrator-item">
          <strong style="color:var(--text);display:block;">Clara · Español Neutro Suave</strong>
          <span>Separada de las voces chilenas locales. Disponible para familias que prefieren acento neutro latinoamericano.</span>
        </div>
      `;
      return;
    }

    vList.innerHTML = `
      <div class="narrator-item">
        <strong style="color:var(--text);display:block;">Voz de Mamá o Papá</strong>
        <span>Usa tu propia voz grabada de forma privada en la pestaña Espacio Mamá y Papá.</span>
      </div>
    `;
  }

  async function loadShowcase() {
    const [archRes, voiceRes] = await Promise.all([
      fetch("/v1/showcase/archetypes"),
      fetch("/v1/showcase/voices")
    ]);
    const archData = await archRes.json();
    const voiceData = await voiceRes.json();

    showcaseArchetypes = archData.archetypes || [];
    catalogVoices = voiceData.voices || [];

    const listEl = document.getElementById("archetypeList");
    listEl.innerHTML = "";

    showcaseArchetypes.forEach((item, idx) => {
      const mins = Math.max(1, Math.round((item.acoustic_qa?.duration_seconds || 180) / 60));
      const themeText = STORY_THEMES[item.profile_id] || (item.parent_guide && item.parent_guide.valor_del_capitulo) || "Amistad y calma";
      const card = document.createElement("div");
      card.className = "archetype-card" + (idx === 0 ? " selected" : "");
      card.dataset.pid = item.profile_id;
      card.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
          <strong>${item.child_alias} (${item.age_years} años)</strong>
          <span class="badge ok">${mins} min</span>
        </div>
        <div style="font-size:0.8rem;color:var(--text-muted);margin-top:5px;">
          ${themeText}
        </div>
      `;
      card.onclick = () => selectArchetype(item.profile_id);
      listEl.appendChild(card);
    });

    const audioEl = document.getElementById("showcaseAudio");
    audioEl.ontimeupdate = () => {
      if (!audioEl.duration || !selectedArchetype) return;
      const ratio = audioEl.currentTime / audioEl.duration;
      const nextIdx = ratio < 0.34 ? 0 : (ratio < 0.68 ? 1 : 2);
      if (nextIdx !== currentSceneIdx) {
        selectScene(nextIdx);
      }
    };

    if (showcaseArchetypes.length > 0) {
      selectArchetype(showcaseArchetypes[0].profile_id);
    } else {
      filterNarratorsByAccent();
    }
  }

  function selectArchetype(profileId) {
    const item = showcaseArchetypes.find((a) => a.profile_id === profileId);
    if (!item) return;
    selectedArchetype = item;

    document.querySelectorAll(".archetype-card").forEach((el) => {
      el.classList.toggle("selected", el.dataset.pid === profileId);
    });

    const mins = Math.max(1, Math.round((item.acoustic_qa?.duration_seconds || 180) / 60));
    const narratorMeta = NARRATOR_FRIENDLY_NAMES[item.voice_id] || {
      name: "Narrador Cálido",
      accentLabel: "Español Chile - Cálido"
    };
    const styleWord = FRIENDLY_WORDS[item.sensory_axes?.lenguaje] || "Claro y cercano";
    const styleOrder = FRIENDLY_ORDER[item.sensory_axes?.estructura] || "Rutina tranquila";
    const stylePace = FRIENDLY_PACE[item.sensory_axes?.estimulo_sonoro] || "Lectura pausada y suave";
    const themeText = STORY_THEMES[item.profile_id] || "Amistad y convivencia";

    document.getElementById("showcaseTitle").textContent = item.series_title;
    document.getElementById("showcaseMeta").textContent =
      `Historia pensada para ${item.child_alias} (${item.age_years} años) · Duración aproximada: ${mins} minutos`;

    document.getElementById("showcaseBadges").innerHTML = `
      <span class="badge accent">${themeText}</span>
      <span class="badge">${styleWord}</span>
      <span class="badge">${styleOrder}</span>
      <span class="badge ok">${stylePace}</span>
    `;

    document.getElementById("audioVoiceLabel").textContent =
      `Narrador: ${narratorMeta.name} (${narratorMeta.accentLabel})`;
    document.getElementById("showcaseAudio").src = item.audio_stream_url;

    const paragraphsHtml = formatParagraphsHtml(item.story_text);
    document.getElementById("storyStandardView").innerHTML = paragraphsHtml;
    document.getElementById("parentReadsBody").innerHTML = paragraphsHtml;

    selectScene(0);
    filterNarratorsByAccent();

    const guide = (item.parent_guide && item.parent_guide.preguntas) || [];
    document.getElementById("parentGuideList").innerHTML = guide.map((q) => `
      <div class="q-card">
        <strong>${q.tipo || "Para conversar"}</strong>
        <div style="font-size:0.93rem;">${q.pregunta}</div>
        ${q.pista_apoyo ? `<div style="font-size:0.79rem;color:var(--text-muted);margin-top:5px;">Idea para acompañar: ${q.pista_apoyo}</div>` : ""}
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
        toast("Cuento descargado para escucharlo sin conexión.");
      });
    }
  }

  function enterChildBedsideMode() {
    document.body.classList.add("amber-sleep");
    document.getElementById("mainHeader").style.display = "none";
    document.getElementById("mainFooter").style.display = "none";
    document.querySelectorAll(".pane").forEach((p) => p.classList.remove("active"));
    document.getElementById("paneBedside").classList.add("active");

    if (selectedArchetype) {
      document.getElementById("bedsideStoryTitle").textContent = selectedArchetype.series_title;
      document.getElementById("bedsideSubtitle").textContent =
        `Para ${selectedArchetype.child_alias} · Deja la pantalla a un lado y descansa`;
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
        alert("PIN de adulto incorrecto.");
        return;
      }
      switchMode("parent");
      return;
    }

    if (pin === "0000") {
      switchMode("showcase");
    } else {
      alert("En los cuentos de ejemplo, ingresa 0000 para volver.");
    }
  }

  async function apiAuth(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "No pudimos iniciar sesión. Revisa tus datos.");
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
      toast("¡Bienvenido/a! Tu espacio familiar ya está listo.");
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
      toast("Qué alegría tenerte de vuelta.");
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

    document.getElementById("childrenList").innerHTML = children.map((c) => {
      const w = FRIENDLY_WORDS[c.perfil_sensorial?.lenguaje] || "Claro y directo";
      const o = FRIENDLY_ORDER[c.perfil_sensorial?.estructura] || "Rutina tranquila";
      const p = FRIENDLY_PACE[c.perfil_sensorial?.estimulo_sonoro] || "Lectura pausada y suave";
      return `
        <div class="q-card">
          <strong>${c.alias} (${c.age_years} años)</strong>
          <div style="font-size:0.81rem;color:var(--text-muted);">${w} · ${o} · ${p}</div>
        </div>
      `;
    }).join("");

    const chSel = document.getElementById("jobChildSelect");
    chSel.innerHTML = children.map((c) => `<option value="${c.id}">${c.alias} (${c.age_years} años)</option>`).join("");
    chSel.onchange = () => loadChildChapters(chSel.value);
    if (children.length > 0) loadChildChapters(children[0].id);

    const activeVoices = voices.filter((v) => v.status === "ACTIVE");
    document.getElementById("jobVoiceSelect").innerHTML = activeVoices.map((v) => {
      const friendly = NARRATOR_FRIENDLY_NAMES[v.id]?.name || v.display_name;
      return `<option value="${v.id}">${friendly}</option>`;
    }).join("");

    document.getElementById("familyVoicesBox").innerHTML = voices.map((v) => {
      const friendly = NARRATOR_FRIENDLY_NAMES[v.id]?.name || v.display_name;
      const statusLabel = v.is_catalog
        ? "Narrador Español Chile - Cálido"
        : (v.status === "ACTIVE" ? "Voz familiar guardada" : "Voz eliminada");
      return `
        <div class="q-card" style="display:flex;justify-content:space-between;align-items:center;gap:10px;">
          <div>
            <strong>${friendly}</strong>
            <span style="font-size:0.78rem;color:var(--text-muted);">${statusLabel}</span>
          </div>
          ${(!v.is_catalog && v.status === "ACTIVE")
            ? `<button class="btn btn-danger" onclick="RoansoApp.revokeVoice('${v.id}')">Borrar mi voz</button>`
            : ""}
        </div>
      `;
    }).join("");
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
        valores_trabajar: ["amistad", "empatía y límites amables"],
        temas_evitar: [],
        closing_phrase: document.getElementById("childClosing").value,
        notas_libres_padre: document.getElementById("childNotes").value
      })
    });
    const data = await res.json();
    if (!res.ok) {
      alert(typeof data.detail === "string" ? data.detail : "Por favor usa solo el nombre o apodo y gustos cotidianos.");
      return;
    }
    toast(`¡Listo! Guardamos el estilo de lectura favorito de ${data.alias}.`);
    refreshParentPortal();
  }

  async function enqueueJob() {
    const childId = document.getElementById("jobChildSelect").value;
    const voiceId = document.getElementById("jobVoiceSelect").value;
    const seed = document.getElementById("jobSeed").value.trim();
    if (!childId || !seed) {
      alert("Elige para quién es la historia y cuéntanos en una frase qué vivió hoy.");
      return;
    }
    const res = await fetch("/v1/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${jwtToken}` },
      body: JSON.stringify({ child_id: childId, voice_id: voiceId, parent_seed: seed })
    });
    const job = await res.json();
    if (!res.ok) {
      alert(job.detail || "No pudimos preparar el cuento en este momento.");
      return;
    }
    document.getElementById("activeJobBox").textContent =
      `Estamos preparando el Capítulo ${job.chapter_number} para esta noche. Quedará guardado aquí mismo apenas esté listo.`;
    toast("¡Cuento de esta noche en preparación!");
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
          <strong>Capítulo ${ch.chapter_number}: ${ch.title}</strong>
          <audio controls preload="none" src="${streamUrl}" style="width:100%;margin-top:6px;"></audio>
        </div>
      `;
    }).join("");
  }

  async function enrollAdultVoice() {
    const displayName = (document.getElementById("voiceDisplayName").value || "").trim() || "Voz de Mamá";
    const consentText = (document.getElementById("voiceConsentText").value || "").trim();
    const sampleBytes = new Uint8Array(64);
    if (window.crypto && window.crypto.getRandomValues) {
      window.crypto.getRandomValues(sampleBytes);
    }
    let bin = "";
    sampleBytes.forEach((b) => { bin += String.fromCharCode(b); });
    const embeddingB64 = btoa(bin);

    const res = await fetch("/v1/voices/parental", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${jwtToken}` },
      body: JSON.stringify({
        display_name: displayName,
        adult_full_name: "Adulto Responsable",
        spoken_consent_transcript: consentText,
        is_minor_voice_declaration_false: true,
        embedding_b64: embeddingB64
      })
    });
    const data = await res.json();
    if (!res.ok) {
      alert(typeof data.detail === "string" ? data.detail : "Revisa la confirmación de voz adulta.");
      return;
    }
    toast(`Guardamos "${data.display_name}" de forma privada para tu familia.`);
    refreshParentPortal();
  }

  async function revokeVoice(voiceId) {
    await fetch(`/v1/voices/${voiceId}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${jwtToken}` }
    });
    toast("Tu voz fue borrada por completo.");
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
    a.download = "mis_cuentos_familiares.json";
    a.click();
  }

  async function eraseFamilyAccount() {
    if (!confirm("¿Quieres borrar tu cuenta familiar, voces guardadas y todos los cuentos de forma permanente?")) {
      return;
    }
    await fetch("/v1/families/me", {
      method: "DELETE",
      headers: { Authorization: `Bearer ${jwtToken}` }
    });
    logout();
    toast("Tu cuenta familiar y todos sus datos fueron eliminados.");
  }

  document.addEventListener("DOMContentLoaded", () => {
    registerServiceWorker();
    loadShowcase();
  });

  return {
    switchMode,
    toggleAmber,
    openPrivacyModal,
    closePrivacyModal,
    setStoryMode,
    selectScene,
    filterNarratorsByAccent,
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
