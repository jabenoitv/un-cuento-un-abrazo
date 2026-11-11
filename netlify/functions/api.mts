import crypto from "node:crypto";
import { getDeployStore, getStore } from "@netlify/blobs";
import { getDatabase } from "@netlify/database";
import type { Config, Context } from "@netlify/functions";

const CLINICAL_REGEX =
  /\b(tea|tdah|tda|autis(?:mo|ta)|asperger|trastorno|diagn[oó]stic[oa]s?|dsm-?5|cie-?10|melatonina|risperidona|metilfenidato|aripiprazol|sertralina|fluoxetina|atomoxetina|psiquiatr[aí]a?|neur[oó]log[oa]|medicaci[oó]n|f[aá]rmacos?|psicotr[oó]picos?)\b/gi;

const FRONTIER_HOSTS = [
  "api.openai.com",
  "api.anthropic.com",
  "generativelanguage.googleapis.com",
  "api.mistral.ai",
  "api.cohere.ai",
  "api.elevenlabs.io",
];

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}

function errorResponse(detail: unknown, status: number): Response {
  return jsonResponse({ detail }, status);
}

function sha256Hex(text: string | Uint8Array): string {
  return crypto.createHash("sha256").update(text).digest("hex");
}

function base64UrlEncode(buf: Buffer | string): string {
  const raw = typeof buf === "string" ? Buffer.from(buf, "utf8") : buf;
  return raw.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function base64UrlDecode(str: string): Buffer {
  const padded = str.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((str.length + 3) % 4);
  return Buffer.from(padded, "base64");
}

function hashSecretPbkdf2(secret: string): string {
  const salt = crypto.randomBytes(16);
  const derived = crypto.pbkdf2Sync(secret, salt, 210_000, 32, "sha256");
  return `pbkdf2_sha256$210000$${salt.toString("hex")}$${derived.toString("hex")}`;
}

function verifySecretPbkdf2(secret: string, stored: string): boolean {
  const parts = stored.split("$");
  if (parts.length !== 4 || parts[0] !== "pbkdf2_sha256") return false;
  const iterations = parseInt(parts[1], 10);
  if (!Number.isFinite(iterations) || iterations < 10_000) return false;
  const salt = Buffer.from(parts[2], "hex");
  const expected = Buffer.from(parts[3], "hex");
  const actual = crypto.pbkdf2Sync(secret, salt, iterations, expected.length, "sha256");
  return actual.length === expected.length && crypto.timingSafeEqual(actual, expected);
}

function signJwtHs256(
  claims: Record<string, unknown>,
  secretKey: string,
  expireMinutes = 720
): string {
  const nowSec = Math.floor(Date.now() / 1000);
  const header = base64UrlEncode(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const payload = base64UrlEncode(
    JSON.stringify({ ...claims, iat: nowSec, exp: nowSec + expireMinutes * 60 })
  );
  const signingInput = `${header}.${payload}`;
  const sig = crypto.createHmac("sha256", secretKey).update(signingInput).digest();
  return `${signingInput}.${base64UrlEncode(sig)}`;
}

function verifyJwtHs256(token: string, secretKey: string): Record<string, unknown> {
  const parts = token.split(".");
  if (parts.length !== 3) throw new Error("Formato JWT inválido");
  const [h, p, s] = parts;
  const signingInput = `${h}.${p}`;
  const expectedSig = crypto.createHmac("sha256", secretKey).update(signingInput).digest();
  const providedSig = base64UrlDecode(s);
  if (expectedSig.length !== providedSig.length || !crypto.timingSafeEqual(expectedSig, providedSig)) {
    throw new Error("Firma JWT inválida");
  }
  const claims = JSON.parse(base64UrlDecode(p).toString("utf8")) as Record<string, unknown>;
  const nowSec = Math.floor(Date.now() / 1000);
  if (typeof claims.exp !== "number" || claims.exp < nowSec) {
    throw new Error("Token expirado");
  }
  return claims;
}

function encryptVoiceEmbeddingAes256Gcm(
  plaintext: Buffer,
  familyId: string,
  masterKeyHex: string
): { ciphertextBlob: Buffer; keySaltHex: string; plaintextSha256: string } {
  const salt = crypto.randomBytes(16);
  const iv = crypto.randomBytes(12);
  const derivedKey = Buffer.from(
    crypto.hkdfSync(
      "sha256",
      Buffer.from(masterKeyHex, "utf8"),
      salt,
      Buffer.from(`roanso-voice-vault:${familyId}`, "utf8"),
      32
    )
  );
  const cipher = crypto.createCipheriv("aes-256-gcm", derivedKey, iv);
  cipher.setAAD(Buffer.from(familyId, "utf8"));
  const enc = Buffer.concat([cipher.update(plaintext), cipher.final()]);
  const tag = cipher.getAuthTag();
  return {
    ciphertextBlob: Buffer.concat([iv, tag, enc]),
    keySaltHex: salt.toString("hex"),
    plaintextSha256: sha256Hex(plaintext),
  };
}

function decryptVoiceEmbeddingAes256Gcm(
  blob: Buffer,
  familyId: string,
  keySaltHex: string,
  masterKeyHex: string
): Buffer {
  const iv = blob.subarray(0, 12);
  const tag = blob.subarray(12, 28);
  const ciphertext = blob.subarray(28);
  const derivedKey = Buffer.from(
    crypto.hkdfSync(
      "sha256",
      Buffer.from(masterKeyHex, "utf8"),
      Buffer.from(keySaltHex, "hex"),
      Buffer.from(`roanso-voice-vault:${familyId}`, "utf8"),
      32
    )
  );
  const decipher = crypto.createDecipheriv("aes-256-gcm", derivedKey, iv);
  decipher.setAAD(Buffer.from(familyId, "utf8"));
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(ciphertext), decipher.final()]);
}

function sanitizeLey21719Text(input: string): { sanitized: string; blockedCount: number } {
  let blockedCount = 0;
  const sanitized = input.replace(CLINICAL_REGEX, () => {
    blockedCount += 1;
    return "[OMITIDO_LEY_21719]";
  });
  return { sanitized, blockedCount };
}

function containsClinicalTerms(input: string): boolean {
  CLINICAL_REGEX.lastIndex = 0;
  return CLINICAL_REGEX.test(input);
}

function openVaultStore(context: Context) {
  if (context.deploy?.context === "production") {
    return getStore({ name: "roanso-vault-prod", consistency: "strong" });
  }
  return getDeployStore("roanso-vault-preview");
}

function computeSpanishReadability(text: string): {
  wordCount: number;
  szigriszt: number;
  fernandezHuerta: number;
} {
  const words = text.match(/[A-Za-záéíóúüñÁÉÍÓÚÜÑ]+/g) || [];
  const sentences = text.split(/[.!?…]+/).filter((s) => s.trim().length > 0);
  const wCount = Math.max(1, words.length);
  const sCount = Math.max(1, sentences.length);
  let syllables = 0;
  for (const w of words) {
    const groups = w.toLowerCase().match(/[aeiouáéíóúü]+/g);
    syllables += Math.max(1, groups ? groups.length : 1);
  }
  const szigriszt = Number((206.835 - 62.3 * (syllables / wCount) - wCount / sCount).toFixed(2));
  const fernandezHuerta = Number(
    (206.84 - 60.0 * (syllables / wCount) - 1.02 * (wCount / sCount)).toFixed(2)
  );
  return { wordCount: words.length, szigriszt, fernandezHuerta };
}

function buildMorningParentGuide(alias: string, closingPhrase: string) {
  return {
    nota_uso: "3 preguntas breves para conversar juntos a la mañana siguiente, sin moralejas ni interrogatorios.",
    preguntas: [
      {
        tipo: "Recuerdo sensorial",
        pregunta: `¿Qué sonido o lugar tranquilo del cuento de anoche le gustó más a ${alias}?`,
        pista_apoyo: "Aceptar respuestas cortas o señalar un objeto cotidiano.",
      },
      {
        tipo: "Puente cotidiano",
        pregunta: "¿Qué parte tranquila del día de hoy se pareció un poquito a la historia?",
        pista_apoyo: "Conectar con un momento concreto sin juzgar conductas.",
      },
      {
        tipo: "Vínculo afectivo",
        pregunta: `¿Dónde te gustaría escuchar hoy nuestra frase "${closingPhrase}"?`,
        pista_apoyo: "Reforzar seguridad y anticipación del ritual familiar.",
      },
    ],
  };
}

function buildChileanWriterPrompt(
  childRow: Record<string, any>,
  chapterNumber: number,
  parentSeed: string
): { writerPrompt: string; bibleSpec: Record<string, any> } {
  const interests: string[] = Array.isArray(childRow.special_interests_json)
    ? childRow.special_interests_json
    : [];
  const valores: string[] = Array.isArray(childRow.valores_json) ? childRow.valores_json : ["calma"];
  const evitar: string[] = Array.isArray(childRow.temas_evitar_json) ? childRow.temas_evitar_json : [];
  const closingPhrase = String(childRow.closing_phrase || "en casa te queremos mucho");

  const bibleSpec = {
    id_perfil: childRow.id,
    protagonista: {
      apodo: childRow.alias,
      edad: childRow.age_years,
    },
    ejes_sensoriales: {
      lenguaje: childRow.axis_lenguaje,
      estructura: childRow.axis_estructura,
      estimulo_sonoro: childRow.axis_estimulo_sonoro,
      foco_tematico: childRow.axis_foco_tematico,
    },
    intereses_especiales: interests,
    valores_trabajar: valores,
    temas_evitar: evitar,
    ritual_cierre: {
      frase_obligatoria: closingPhrase,
    },
  };

  const writerPrompt = [
    `Escribe en español natural de Chile el Capítulo ${chapterNumber} de un cuento nocturno relajante en TERCERA PERSONA para ${childRow.alias} (${childRow.age_years} años).`,
    `Semilla cotidiana observada hoy por su familia: ${parentSeed}.`,
    `Intereses especiales a integrar con naturalidad: ${interests.join(", ") || "estrellas del sur"}.`,
    `Eje de lenguaje: ${childRow.axis_lenguaje}. Eje de estructura: ${childRow.axis_estructura}.`,
    `Reglas estrictas Carol Gray & ROANSO: cero segunda persona directiva ("tú debes"), cero sermones moralizantes, cero léxico de doblaje neutro ("autobús", "nevera", "pastel"), cero suspenso ansioso.`,
    `Cierra obligatoriamente el último párrafo incluyendo la frase exacta: "${closingPhrase}".`,
  ].join("\n");

  return { writerPrompt, bibleSpec };
}

function mapChildRow(row: Record<string, any>) {
  return {
    id: row.id,
    alias: row.alias,
    age_years: Number(row.age_years),
    perfil_sensorial: {
      lenguaje: row.axis_lenguaje,
      estructura: row.axis_estructura,
      estimulo_sonoro: row.axis_estimulo_sonoro,
      foco_tematico: row.axis_foco_tematico,
    },
    intereses_especiales: row.special_interests_json || [],
    valores_trabajar: row.valores_json || [],
    temas_evitar: row.temas_evitar_json || [],
    closing_phrase: row.closing_phrase,
    preferred_voice_id: row.preferred_voice_id,
    sanitized_terms_blocked_count: Number(row.sanitized_terms_blocked_count || 0),
    created_at: new Date(row.created_at).toISOString(),
  };
}

function mapJobRow(row: Record<string, any>) {
  return {
    id: row.id,
    child_id: row.child_id,
    chapter_number: Number(row.chapter_number),
    status: row.status,
    stage_detail: row.stage_detail,
    attempts: Number(row.attempts || 0),
    leased_by_worker_id: row.leased_by_worker_id ?? null,
    slm_model_tag: row.slm_model_tag ?? null,
    slm_params_billions: row.slm_params_billions !== null ? Number(row.slm_params_billions) : null,
    gpu_seconds_used: row.gpu_seconds_used !== null ? Number(row.gpu_seconds_used) : null,
    estimated_cost_usd: row.estimated_cost_usd !== null ? Number(row.estimated_cost_usd) : null,
    last_error: row.last_error ?? null,
    created_at: new Date(row.created_at).toISOString(),
    completed_at: row.completed_at ? new Date(row.completed_at).toISOString() : null,
  };
}

export default async (req: Request, context: Context) => {
  const upstreamUrl = Netlify.env.get("ROANSO_UPSTREAM_CONTROL_PLANE_URL");
  if (upstreamUrl) {
    const incoming = new URL(req.url);
    const target = new URL(incoming.pathname + incoming.search, upstreamUrl);
    return fetch(new Request(target.toString(), req));
  }

  const jwtSecret = Netlify.env.get("ROANSO_JWT_SECRET");
  const vaultMasterKey = Netlify.env.get("ROANSO_VAULT_MASTER_KEY");
  const workerSharedToken = Netlify.env.get("ROANSO_WORKER_SHARED_TOKEN");
  if (!jwtSecret || !vaultMasterKey) {
    return errorResponse("Variables criptográficas de servidor no configuradas en Netlify.", 500);
  }

  const db = getDatabase();
  const url = new URL(req.url);
  const pathname = url.pathname.replace(/\/+$/, "") || "/";
  const method = req.method.toUpperCase();

  async function requireParent(): Promise<Record<string, any> | Response> {
    const auth = req.headers.get("authorization") || "";
    const match = /^Bearer\s+(.+)$/i.exec(auth.trim());
    const rawToken = match ? match[1] : url.searchParams.get("token");
    if (!rawToken) {
      return errorResponse("Token Bearer requerido.", 401);
    }
    try {
      const claims = verifyJwtHs256(rawToken, jwtSecret!);
      const rows = await db.sql`
        SELECT id, family_id, email, full_name, password_hash, parental_pin_hash
        FROM parent_users
        WHERE id = ${String(claims.sub || "")} AND family_id = ${String(claims.fam || "")}
        LIMIT 1
      `;
      if (rows.length === 0) {
        return errorResponse("Cuenta familiar no activa.", 401);
      }
      return rows[0];
    } catch (err: any) {
      return errorResponse(`JWT inválido: ${err?.message || "error"}`, 401);
    }
  }

  function verifyWorkerHeader(): boolean {
    if (!workerSharedToken) return false;
    const headerVal = req.headers.get("x-worker-token") || "";
    const a = Buffer.from(headerVal, "utf8");
    const b = Buffer.from(workerSharedToken, "utf8");
    return a.length === b.length && crypto.timingSafeEqual(a, b);
  }

  try {
    // 1. POST /v1/auth/register
    if (method === "POST" && pathname === "/v1/auth/register") {
      const body = (await req.json()) as Record<string, any>;
      if (!body.accepted_ley21719_notice) {
        return errorResponse("Debe aceptar el aviso de tratamiento mínimo Ley 21.719.", 422);
      }
      const email = String(body.email || "").trim().toLowerCase();
      const familyName = String(body.family_name || "").trim();
      const fullName = String(body.full_name || "").trim();
      const password = String(body.password || "");
      const pin = String(body.parental_pin || "").trim();
      const planTier = String(body.plan_tier || "familiar_voces");

      if (!email.includes("@") || familyName.length < 2 || fullName.length < 2) {
        return errorResponse("Completa nombre familiar, nombre adulto y correo válido.", 422);
      }
      if (password.length < 8) {
        return errorResponse("La contraseña debe tener al menos 8 caracteres.", 422);
      }
      if (!/^\d{4,8}$/.test(pin)) {
        return errorResponse("El PIN parental debe contener entre 4 y 8 dígitos numéricos.", 422);
      }

      const tenantId = crypto.randomUUID();
      const parentId = crypto.randomUUID();
      const consentId = crypto.randomUUID();
      const pwHash = hashSecretPbkdf2(password);
      const pinHash = hashSecretPbkdf2(pin);
      const consentStatement = `Acepto el tratamiento mínimo no clínico bajo Ley 21.719 para la familia ${familyName} (${email}).`;

      const client = await db.pool.connect();
      try {
        await client.query("BEGIN");
        const dup = await client.query("SELECT id FROM parent_users WHERE email = $1 LIMIT 1", [email]);
        if (dup.rowCount && dup.rowCount > 0) {
          await client.query("ROLLBACK");
          return errorResponse("El correo ya está registrado.", 409);
        }
        await client.query(
          "INSERT INTO family_tenants (id, family_name, plan_tier) VALUES ($1, $2, $3)",
          [tenantId, familyName, planTier]
        );
        await client.query(
          "INSERT INTO parent_users (id, family_id, email, full_name, password_hash, parental_pin_hash) VALUES ($1, $2, $3, $4, $5, $6)",
          [parentId, tenantId, email, fullName, pwHash, pinHash]
        );
        await client.query(
          "INSERT INTO consent_audit_logs (id, family_id, parent_user_id, consent_type, statement_sha256, metadata_json) VALUES ($1, $2, $3, $4, $5, $6::jsonb)",
          [
            consentId,
            tenantId,
            parentId,
            "ley21719_onboarding",
            sha256Hex(consentStatement),
            JSON.stringify({ plan_tier: planTier }),
          ]
        );
        await client.query("COMMIT");
      } catch (txErr) {
        await client.query("ROLLBACK");
        throw txErr;
      } finally {
        client.release();
      }

      const token = signJwtHs256({ sub: parentId, fam: tenantId, scope: "family_parent" }, jwtSecret);
      return jsonResponse(
        {
          access_token: token,
          token_type: "bearer",
          family_id: tenantId,
          parent_user_id: parentId,
          plan_tier: planTier,
        },
        201
      );
    }

    // 2. POST /v1/auth/login
    if (method === "POST" && pathname === "/v1/auth/login") {
      const body = (await req.json()) as Record<string, any>;
      const email = String(body.email || "").trim().toLowerCase();
      const password = String(body.password || "");
      const rows = await db.sql`
        SELECT u.id, u.family_id, u.password_hash, t.plan_tier
        FROM parent_users u
        JOIN family_tenants t ON t.id = u.family_id
        WHERE u.email = ${email}
        LIMIT 1
      `;
      if (rows.length === 0 || !verifySecretPbkdf2(password, String(rows[0].password_hash))) {
        return errorResponse("Credenciales inválidas.", 401);
      }
      const token = signJwtHs256(
        { sub: rows[0].id, fam: rows[0].family_id, scope: "family_parent" },
        jwtSecret
      );
      return jsonResponse({
        access_token: token,
        token_type: "bearer",
        family_id: rows[0].family_id,
        parent_user_id: rows[0].id,
        plan_tier: rows[0].plan_tier,
      });
    }

    // 3. POST /v1/auth/verify-pin
    if (method === "POST" && pathname === "/v1/auth/verify-pin") {
      const parentOrErr = await requireParent();
      if (parentOrErr instanceof Response) return parentOrErr;
      const body = (await req.json()) as Record<string, any>;
      const pin = String(body.parental_pin || "").trim();
      if (!verifySecretPbkdf2(pin, String(parentOrErr.parental_pin_hash))) {
        return errorResponse("PIN parental incorrecto.", 403);
      }
      const elevated = signJwtHs256(
        { sub: parentOrErr.id, fam: parentOrErr.family_id, scope: "parental_unlocked" },
        jwtSecret,
        30
      );
      return jsonResponse({ unlocked: true, elevated_token: elevated, expires_in_minutes: 30 });
    }

    // 4. GET & POST /v1/children
    if (pathname === "/v1/children") {
      const parentOrErr = await requireParent();
      if (parentOrErr instanceof Response) return parentOrErr;

      if (method === "GET") {
        const rows = await db.sql`
          SELECT * FROM child_profiles
          WHERE family_id = ${parentOrErr.family_id}
          ORDER BY created_at ASC
        `;
        return jsonResponse(rows.map(mapChildRow));
      }

      if (method === "POST") {
        const body = (await req.json()) as Record<string, any>;
        const alias = String(body.alias || "").trim();
        const ageYears = Number(body.age_years || 7);
        if (alias.length < 2 || alias.split(/\s+/).length > 2) {
          return errorResponse("Usa solo el primer nombre o apodo (sin apellidos por Ley 21.719).", 422);
        }
        if (ageYears < 3 || ageYears > 13) {
          return errorResponse("Edad fuera del rango infantil soportado (3 a 13 años).", 422);
        }
        const interests: string[] = Array.isArray(body.intereses_especiales) ? body.intereses_especiales : [];
        const valores: string[] = Array.isArray(body.valores_trabajar) ? body.valores_trabajar : [];
        const evitar: string[] = Array.isArray(body.temas_evitar) ? body.temas_evitar : [];

        for (const fieldText of [alias, ...interests, ...valores, ...evitar]) {
          if (containsClinicalTerms(String(fieldText))) {
            return errorResponse(
              "Ley 21.719: ROANSO no almacena diagnósticos clínicos ni fármacos. Describe preferencias sensoriales funcionales.",
              422
            );
          }
        }

        const rawNotes = String(body.notas_libres_padre || "");
        const { sanitized: cleanNotes, blockedCount } = sanitizeLey21719Text(rawNotes);
        const ps = body.perfil_sensorial || {};
        const childId = crypto.randomUUID();
        const closingPhrase = String(body.closing_phrase || "en casa te queremos mucho").trim();
        const prefVoice = String(body.preferred_voice_id || "catalog-chileno-sereno-m1");

        const inserted = await db.sql`
          INSERT INTO child_profiles (
            id, family_id, alias, age_years,
            axis_lenguaje, axis_estructura, axis_estimulo_sonoro, axis_foco_tematico,
            special_interests_json, valores_json, temas_evitar_json,
            closing_phrase, preferred_voice_id, bible_json, sanitized_terms_blocked_count
          ) VALUES (
            ${childId}, ${parentOrErr.family_id}, ${alias}, ${ageYears},
            ${String(ps.lenguaje || "literal_claro")},
            ${String(ps.estructura || "ritual_anticipado")},
            ${String(ps.estimulo_sonoro || "monovoz_estable")},
            ${String(ps.foco_tematico || "interes_profundo")},
            ${JSON.stringify(interests)}::jsonb,
            ${JSON.stringify(valores)}::jsonb,
            ${JSON.stringify(evitar)}::jsonb,
            ${closingPhrase},
            ${prefVoice},
            ${JSON.stringify({ notas_contexto_no_clinicas: cleanNotes })}::jsonb,
            ${blockedCount}
          )
          RETURNING *
        `;
        return jsonResponse(mapChildRow(inserted[0]), 201);
      }
    }

    // 5. GET /v1/children/:childId/chapters
    const chaptersMatch = /^\/v1\/children\/([^/]+)\/chapters$/.exec(pathname);
    if (method === "GET" && chaptersMatch) {
      const parentOrErr = await requireParent();
      if (parentOrErr instanceof Response) return parentOrErr;
      const childId = chaptersMatch[1];

      const childRows = await db.sql`
        SELECT * FROM child_profiles WHERE id = ${childId} AND family_id = ${parentOrErr.family_id} LIMIT 1
      `;
      if (childRows.length === 0) {
        return errorResponse("Perfil infantil no encontrado.", 404);
      }
      const chRows = await db.sql`
        SELECT * FROM story_chapters WHERE child_id = ${childId} ORDER BY chapter_number DESC
      `;
      return jsonResponse(
        chRows.map((ch) => ({
          chapter_id: ch.id,
          child_id: childId,
          child_alias: childRows[0].alias,
          chapter_number: Number(ch.chapter_number),
          title: ch.title,
          story_text: ch.story_text,
          word_count: Number(ch.word_count),
          readability: {
            szigriszt_pazos: Number(ch.szigriszt_score),
            fernandez_huerta: Number(ch.fernandez_huerta_score),
          },
          acoustic_qa: {
            whisper_wer: Number(ch.whisper_wer),
            speaker_similarity: Number(ch.speaker_similarity),
          },
          audio_url: `/v1/media/${ch.audio_storage_key}`,
          parent_guide: ch.parent_guide_json || {},
          qa_report: ch.qa_report_json || {},
          created_at: new Date(ch.created_at).toISOString(),
        }))
      );
    }

    // 6. GET /v1/voices & POST /v1/voices/parental & DELETE /v1/voices/:voiceId
    if (method === "GET" && pathname === "/v1/voices") {
      const parentOrErr = await requireParent();
      if (parentOrErr instanceof Response) return parentOrErr;
      const rows = await db.sql`
        SELECT id, display_name, is_catalog, locale, status, created_at, revoked_at
        FROM voice_profiles
        WHERE is_catalog = TRUE OR family_id = ${parentOrErr.family_id}
        ORDER BY is_catalog DESC, created_at ASC
      `;
      return jsonResponse(
        rows.map((v) => ({
          id: v.id,
          display_name: v.display_name,
          is_catalog: Boolean(v.is_catalog),
          locale: v.locale,
          status: v.status,
          created_at: new Date(v.created_at).toISOString(),
          revoked_at: v.revoked_at ? new Date(v.revoked_at).toISOString() : null,
        }))
      );
    }

    if (method === "POST" && pathname === "/v1/voices/parental") {
      const parentOrErr = await requireParent();
      if (parentOrErr instanceof Response) return parentOrErr;
      const body = (await req.json()) as Record<string, any>;
      if (!body.is_minor_voice_declaration_false) {
        return errorResponse("Prohibición ética absoluta: ROANSO jamás clona ni almacena voces de menores.", 422);
      }
      const consentTranscript = String(body.spoken_consent_transcript || "");
      const low = consentTranscript.toLowerCase();
      for (const reqWord of ["autorizo", "roanso", "voz"]) {
        if (!low.includes(reqWord)) {
          return errorResponse(
            `La declaración de consentimiento biométrico debe incluir la palabra "${reqWord}".`,
            422
          );
        }
      }
      const rawPt = Buffer.from(String(body.embedding_b64 || ""), "base64");
      if (rawPt.length < 12) {
        return errorResponse("Muestra biométrica inválida o demasiado corta.", 400);
      }

      const encrypted = encryptVoiceEmbeddingAes256Gcm(rawPt, parentOrErr.family_id, vaultMasterKey);
      const voiceId = crypto.randomUUID();
      const consentId = crypto.randomUUID();
      const storageKey = `vault/${parentOrErr.family_id}/${voiceId}.aes256gcm`;

      const store = openVaultStore(context);
      await store.set(
        storageKey,
        encrypted.ciphertextBlob.buffer.slice(
          encrypted.ciphertextBlob.byteOffset,
          encrypted.ciphertextBlob.byteOffset + encrypted.ciphertextBlob.byteLength
        ) as ArrayBuffer
      );

      await db.sql`
        INSERT INTO consent_audit_logs (id, family_id, parent_user_id, consent_type, statement_sha256, metadata_json)
        VALUES (
          ${consentId},
          ${parentOrErr.family_id},
          ${parentOrErr.id},
          ${"biometric_voice_clone"},
          ${sha256Hex(consentTranscript)},
          ${JSON.stringify({
            adult_full_name: String(body.adult_full_name || ""),
            embedding_plaintext_sha256: encrypted.plaintextSha256,
          })}::jsonb
        )
      `;

      const inserted = await db.sql`
        INSERT INTO voice_profiles (
          id, family_id, display_name, is_catalog, locale,
          storage_key, key_salt_hex, plaintext_sha256, consent_log_id, status
        ) VALUES (
          ${voiceId}, ${parentOrErr.family_id}, ${String(body.display_name || "Voz Adulta")}, FALSE, 'es-CL',
          ${storageKey}, ${encrypted.keySaltHex}, ${encrypted.plaintextSha256}, ${consentId}, 'ACTIVE'
        )
        RETURNING *
      `;
      const v = inserted[0];
      return jsonResponse(
        {
          id: v.id,
          display_name: v.display_name,
          is_catalog: false,
          locale: v.locale,
          status: v.status,
          created_at: new Date(v.created_at).toISOString(),
          revoked_at: null,
        },
        201
      );
    }

    const voiceDelMatch = /^\/v1\/voices\/([^/]+)$/.exec(pathname);
    if (method === "DELETE" && voiceDelMatch) {
      const parentOrErr = await requireParent();
      if (parentOrErr instanceof Response) return parentOrErr;
      const voiceId = voiceDelMatch[1];
      const rows = await db.sql`
        SELECT * FROM voice_profiles
        WHERE id = ${voiceId} AND family_id = ${parentOrErr.family_id} AND is_catalog = FALSE
        LIMIT 1
      `;
      if (rows.length === 0) {
        return errorResponse("Voz parental no encontrada.", 404);
      }
      const v = rows[0];
      if (v.storage_key) {
        const store = openVaultStore(context);
        await store.delete(String(v.storage_key));
      }
      await db.sql`
        UPDATE voice_profiles
        SET key_salt_hex = '', storage_key = '', status = 'REVOKED_SHREDDED', revoked_at = NOW()
        WHERE id = ${voiceId}
      `;
      await db.sql`
        INSERT INTO consent_audit_logs (id, family_id, parent_user_id, consent_type, statement_sha256, metadata_json)
        VALUES (
          ${crypto.randomUUID()},
          ${parentOrErr.family_id},
          ${parentOrErr.id},
          ${"voice_revocation"},
          ${sha256Hex(`revoke:${voiceId}`)},
          ${JSON.stringify({ voice_id: voiceId, shredded: true })}::jsonb
        )
      `;
      return jsonResponse({ revoked: true, voice_id: voiceId, status: "REVOKED_SHREDDED" });
    }

    // 7. POST /v1/jobs & GET /v1/jobs/:jobId
    if (method === "POST" && pathname === "/v1/jobs") {
      const parentOrErr = await requireParent();
      if (parentOrErr instanceof Response) return parentOrErr;
      const body = (await req.json()) as Record<string, any>;
      const childId = String(body.child_id || "");
      const voiceId = String(body.voice_id || "catalog-chileno-sereno-m1");
      const rawSeed = String(body.parent_seed || "").trim();
      if (!childId || rawSeed.length < 4) {
        return errorResponse("Ingresa la semilla cotidiana del día (mínimo 4 caracteres).", 422);
      }
      const { sanitized: cleanSeed } = sanitizeLey21719Text(rawSeed);

      const childRows = await db.sql`
        SELECT * FROM child_profiles WHERE id = ${childId} AND family_id = ${parentOrErr.family_id} LIMIT 1
      `;
      if (childRows.length === 0) {
        return errorResponse("Perfil infantil no encontrado.", 404);
      }

      const voiceRows = await db.sql`
        SELECT * FROM voice_profiles
        WHERE id = ${voiceId} AND (is_catalog = TRUE OR family_id = ${parentOrErr.family_id})
        LIMIT 1
      `;
      if (voiceRows.length === 0 || voiceRows[0].status !== "ACTIVE") {
        return errorResponse("La voz seleccionada fue revocada o no está activa.", 403);
      }

      const countRows = await db.sql`
        SELECT COALESCE(MAX(chapter_number), 0) AS max_ch FROM story_jobs WHERE child_id = ${childId}
      `;
      const nextChapter = Number(countRows[0].max_ch || 0) + 1;
      const { writerPrompt, bibleSpec } = buildChileanWriterPrompt(childRows[0], nextChapter, cleanSeed);

      const jobId = crypto.randomUUID();
      const inserted = await db.sql`
        INSERT INTO story_jobs (
          id, family_id, child_id, voice_id, chapter_number,
          parent_seed, writer_prompt, bible_spec_json, status, stage_detail
        ) VALUES (
          ${jobId}, ${parentOrErr.family_id}, ${childId}, ${voiceId}, ${nextChapter},
          ${cleanSeed}, ${writerPrompt}, ${JSON.stringify(bibleSpec)}::jsonb,
          'QUEUED', 'waiting_for_elastic_gpu'
        )
        RETURNING *
      `;
      return jsonResponse(mapJobRow(inserted[0]), 202);
    }

    const jobGetMatch = /^\/v1\/jobs\/([^/]+)$/.exec(pathname);
    if (method === "GET" && jobGetMatch) {
      const parentOrErr = await requireParent();
      if (parentOrErr instanceof Response) return parentOrErr;
      const rows = await db.sql`
        SELECT * FROM story_jobs
        WHERE id = ${jobGetMatch[1]} AND family_id = ${parentOrErr.family_id}
        LIMIT 1
      `;
      if (rows.length === 0) return errorResponse("Trabajo no encontrado.", 404);
      return jsonResponse(mapJobRow(rows[0]));
    }

    // 8. GET /v1/media/:storageKey (Private Family MP3 Streaming with HTTP 206 Range)
    if (method === "GET" && pathname.startsWith("/v1/media/")) {
      const parentOrErr = await requireParent();
      if (parentOrErr instanceof Response) return parentOrErr;
      const storageKey = decodeURIComponent(pathname.slice("/v1/media/".length));
      if (!storageKey.startsWith(`audio/${parentOrErr.family_id}/`)) {
        return errorResponse("Acceso denegado al objeto de otra familia.", 403);
      }
      const store = openVaultStore(context);
      const rawBuf = await store.get(storageKey, { type: "arrayBuffer" });
      if (rawBuf === null) {
        return errorResponse("Audio familiar no encontrado.", 404);
      }
      const total = rawBuf.byteLength;
      const rangeHeader = req.headers.get("range");
      if (rangeHeader) {
        const m = /bytes=(\d+)-(\d*)/.exec(rangeHeader);
        if (m) {
          const start = Math.min(parseInt(m[1], 10), Math.max(0, total - 1));
          const end = m[2] ? Math.min(parseInt(m[2], 10), total - 1) : total - 1;
          const slice = rawBuf.slice(start, end + 1);
          return new Response(slice, {
            status: 206,
            headers: {
              "Content-Type": "audio/mpeg",
              "Accept-Ranges": "bytes",
              "Content-Range": `bytes ${start}-${end}/${total}`,
              "Content-Length": String(slice.byteLength),
            },
          });
        }
      }
      return new Response(rawBuf, {
        status: 200,
        headers: {
          "Content-Type": "audio/mpeg",
          "Accept-Ranges": "bytes",
          "Content-Length": String(total),
        },
      });
    }

    // 9. GET /v1/families/me/export & DELETE /v1/families/me (Ley 21.719 ARCO+)
    if (method === "GET" && pathname === "/v1/families/me/export") {
      const parentOrErr = await requireParent();
      if (parentOrErr instanceof Response) return parentOrErr;
      const tenants = await db.sql`SELECT * FROM family_tenants WHERE id = ${parentOrErr.family_id} LIMIT 1`;
      const children = await db.sql`SELECT * FROM child_profiles WHERE family_id = ${parentOrErr.family_id} ORDER BY created_at ASC`;
      const consents = await db.sql`SELECT * FROM consent_audit_logs WHERE family_id = ${parentOrErr.family_id} ORDER BY created_at ASC`;
      return jsonResponse({
        law_compliance: "Chile Ley 21.719 — Portabilidad ARCO+",
        clinical_data_stored: false,
        family: {
          id: parentOrErr.family_id,
          family_name: tenants[0]?.family_name || "",
          plan_tier: tenants[0]?.plan_tier || "",
        },
        children: children.map(mapChildRow),
        consent_ledger: consents.map((c) => ({
          id: c.id,
          consent_type: c.consent_type,
          statement_sha256: c.statement_sha256,
          created_at: new Date(c.created_at).toISOString(),
        })),
      });
    }

    if (method === "DELETE" && pathname === "/v1/families/me") {
      const parentOrErr = await requireParent();
      if (parentOrErr instanceof Response) return parentOrErr;
      const store = openVaultStore(context);
      const voices = await db.sql`
        SELECT storage_key FROM voice_profiles WHERE family_id = ${parentOrErr.family_id} AND is_catalog = FALSE
      `;
      for (const v of voices) {
        if (v.storage_key) await store.delete(String(v.storage_key));
      }
      const chapters = await db.sql`
        SELECT sc.audio_storage_key
        FROM story_chapters sc
        JOIN child_profiles cp ON cp.id = sc.child_id
        WHERE cp.family_id = ${parentOrErr.family_id}
      `;
      for (const ch of chapters) {
        if (ch.audio_storage_key) await store.delete(String(ch.audio_storage_key));
      }
      await db.sql`DELETE FROM family_tenants WHERE id = ${parentOrErr.family_id}`;
      return jsonResponse({ erased: true, family_id: parentOrErr.family_id });
    }

    // 10. Elastic Rented GPU Worker Plane (/v1/worker/leases/*)
    if (pathname.startsWith("/v1/worker/leases")) {
      if (!verifyWorkerHeader()) {
        return errorResponse("X-Worker-Token inválido.", 401);
      }

      if (method === "POST" && pathname === "/v1/worker/leases/claim") {
        const body = (await req.json()) as Record<string, any>;
        const paramsB = Number(body.slm_params_billions || 7.6);
        const infEndpoint = String(body.inference_endpoint || "").toLowerCase();
        if (paramsB > 8.0) {
          return errorResponse("Gobernanza Human-Scale AI: prohibidos modelos > 8.0B parámetros.", 422);
        }
        if (FRONTIER_HOSTS.some((h) => infEndpoint.includes(h))) {
          return errorResponse("Gobernanza Human-Scale AI: prohibidas APIs propietarias frontier.", 422);
        }

        const workerId = String(body.worker_id || "gpu-worker");
        const leaseToken = crypto.randomBytes(24).toString("hex");
        const client = await db.pool.connect();
        let claimed: Record<string, any> | null = null;
        try {
          await client.query("BEGIN");
          // Reclaim orphan spot GPU leases whose TTL expired
          await client.query(`
            UPDATE story_jobs
            SET status = 'QUEUED', leased_by_worker_id = NULL, lease_token = NULL, stage_detail = 'requeued_after_spot_preemption'
            WHERE status LIKE 'LEASED_%' AND lease_expires_at < NOW() AND attempts < 3
          `);
          const res = await client.query(
            `
            UPDATE story_jobs
            SET status = 'LEASED_GENERATING',
                stage_detail = 'generating_verified_text',
                attempts = attempts + 1,
                leased_by_worker_id = $1,
                lease_token = $2,
                lease_expires_at = NOW() + INTERVAL '5 minutes'
            WHERE id = (
              SELECT id FROM story_jobs
              WHERE status = 'QUEUED'
              ORDER BY created_at ASC
              FOR UPDATE SKIP LOCKED
              LIMIT 1
            )
            RETURNING *
            `,
            [workerId, leaseToken]
          );
          await client.query("COMMIT");
          claimed = res.rows[0] || null;
        } catch (txErr) {
          await client.query("ROLLBACK");
          throw txErr;
        } finally {
          client.release();
        }

        if (!claimed) {
          return new Response(null, { status: 204 });
        }

        let ephemeralVoiceB64: string | null = null;
        const vRows = await db.sql`SELECT * FROM voice_profiles WHERE id = ${claimed.voice_id} LIMIT 1`;
        if (vRows.length > 0 && !vRows[0].is_catalog && vRows[0].status === "ACTIVE" && vRows[0].storage_key) {
          const store = openVaultStore(context);
          const encBuf = await store.get(String(vRows[0].storage_key), { type: "arrayBuffer" });
          if (encBuf !== null) {
            const pt = decryptVoiceEmbeddingAes256Gcm(
              Buffer.from(encBuf),
              String(claimed.family_id),
              String(vRows[0].key_salt_hex),
              vaultMasterKey
            );
            ephemeralVoiceB64 = pt.toString("base64");
          }
        }

        return jsonResponse({
          job_id: claimed.id,
          lease_token: leaseToken,
          family_id: claimed.family_id,
          child_id: claimed.child_id,
          chapter_number: Number(claimed.chapter_number),
          voice_id: claimed.voice_id,
          writer_prompt: claimed.writer_prompt,
          bible_spec: claimed.bible_spec_json,
          ephemeral_voice_embedding_b64: ephemeralVoiceB64,
        });
      }

      const hbMatch = /^\/v1\/worker\/leases\/([^/]+)\/heartbeat$/.exec(pathname);
      if (method === "POST" && hbMatch) {
        const body = (await req.json()) as Record<string, any>;
        const rows = await db.sql`
          UPDATE story_jobs
          SET stage_detail = ${String(body.stage_detail || "processing")},
              lease_expires_at = NOW() + INTERVAL '5 minutes'
          WHERE id = ${hbMatch[1]}
            AND leased_by_worker_id = ${String(body.worker_id || "")}
            AND lease_token = ${String(body.lease_token || "")}
          RETURNING lease_expires_at
        `;
        if (rows.length === 0) return errorResponse("Lease inválido o expirado.", 409);
        return jsonResponse({ ok: true, lease_expires_at: new Date(rows[0].lease_expires_at).toISOString() });
      }

      const upMatch = /^\/v1\/worker\/leases\/([^/]+)\/upload-audio$/.exec(pathname);
      if (method === "POST" && upMatch) {
        const body = (await req.json()) as Record<string, any>;
        const jobs = await db.sql`
          SELECT * FROM story_jobs
          WHERE id = ${upMatch[1]}
            AND leased_by_worker_id = ${String(body.worker_id || "")}
            AND lease_token = ${String(body.lease_token || "")}
          LIMIT 1
        `;
        if (jobs.length === 0) return errorResponse("Lease inválido.", 409);
        const mp3Buf = Buffer.from(String(body.audio_mp3_b64 || ""), "base64");
        const key = `audio/${jobs[0].family_id}/${jobs[0].child_id}/cap_${String(jobs[0].chapter_number).padStart(4, "0")}.mp3`;
        const store = openVaultStore(context);
        await store.set(
          key,
          mp3Buf.buffer.slice(mp3Buf.byteOffset, mp3Buf.byteOffset + mp3Buf.byteLength) as ArrayBuffer
        );
        return jsonResponse({ audio_storage_key: key });
      }

      const compMatch = /^\/v1\/worker\/leases\/([^/]+)\/complete$/.exec(pathname);
      if (method === "POST" && compMatch) {
        const body = (await req.json()) as Record<string, any>;
        const whisperWer = Number(body.whisper_wer ?? 1.0);
        const paramsB = Number(body.slm_params_billions ?? 7.6);
        if (whisperWer > 0.12) {
          return errorResponse(`Inteligibilidad insuficiente: Whisper WER=${whisperWer} excede 0.12.`, 422);
        }
        if (paramsB > 8.0) {
          return errorResponse("Modelo excede techo Human-Scale AI de 8.0B.", 422);
        }
        const jobs = await db.sql`
          SELECT * FROM story_jobs
          WHERE id = ${compMatch[1]}
            AND leased_by_worker_id = ${String(body.worker_id || "")}
            AND lease_token = ${String(body.lease_token || "")}
          LIMIT 1
        `;
        if (jobs.length === 0) return errorResponse("Lease inválido.", 409);
        const job = jobs[0];
        const childRows = await db.sql`SELECT * FROM child_profiles WHERE id = ${job.child_id} LIMIT 1`;
        const storyText = String(body.story_text || "").trim();
        const { wordCount, szigriszt, fernandezHuerta } = computeSpanishReadability(storyText);
        const parentGuide = buildMorningParentGuide(
          String(childRows[0]?.alias || "tu hijo/a"),
          String(childRows[0]?.closing_phrase || "en casa te queremos mucho")
        );
        const gpuSeconds = Number(body.gpu_seconds_used || 0);
        const estCostUsd = Number(((gpuSeconds / 3600) * 0.44).toFixed(5));
        const chapterId = crypto.randomUUID();

        await db.sql`
          INSERT INTO story_chapters (
            id, child_id, job_id, chapter_number, title, story_text,
            word_count, szigriszt_score, fernandez_huerta_score,
            whisper_wer, speaker_similarity, audio_storage_key,
            parent_guide_json, qa_report_json
          ) VALUES (
            ${chapterId}, ${job.child_id}, ${job.id}, ${job.chapter_number},
            ${String(body.title || `Capítulo ${job.chapter_number}`)},
            ${storyText}, ${wordCount}, ${szigriszt}, ${fernandezHuerta},
            ${whisperWer}, ${Number(body.speaker_similarity ?? 0.9)},
            ${String(body.audio_storage_key || "")},
            ${JSON.stringify(parentGuide)}::jsonb,
            ${JSON.stringify(body.editorial_review || {})}::jsonb
          )
        `;
        await db.sql`
          UPDATE story_jobs
          SET status = 'COMPLETED',
              stage_detail = 'verified_and_published',
              slm_model_tag = ${String(body.slm_model_tag || "roanso-chileno:7b")},
              slm_params_billions = ${paramsB},
              gpu_seconds_used = ${gpuSeconds},
              estimated_cost_usd = ${estCostUsd},
              completed_at = NOW()
          WHERE id = ${job.id}
        `;
        return jsonResponse({
          ok: true,
          chapter_id: chapterId,
          szigriszt_score: szigriszt,
          word_count: wordCount,
        });
      }

      const failMatch = /^\/v1\/worker\/leases\/([^/]+)\/fail$/.exec(pathname);
      if (method === "POST" && failMatch) {
        const body = (await req.json()) as Record<string, any>;
        const nextStatus = body.retryable ? "QUEUED" : "FAILED";
        const rows = await db.sql`
          UPDATE story_jobs
          SET status = ${nextStatus},
              stage_detail = 'worker_reported_failure',
              last_error = ${String(body.error_message || "error").slice(0, 900)},
              leased_by_worker_id = NULL,
              lease_token = NULL
          WHERE id = ${failMatch[1]}
          RETURNING id, status
        `;
        if (rows.length === 0) return errorResponse("Trabajo no encontrado.", 404);
        return jsonResponse({ ok: true, job_id: rows[0].id, status: rows[0].status });
      }
    }

    return errorResponse(`Ruta no encontrada: ${method} ${pathname}`, 404);
  } catch (err: any) {
    console.error("[roanso-api-error]", err);
    return errorResponse(err?.message || "Error interno del servidor.", 500);
  }
};

export const config: Config = {
  path: ["/v1/*"],
  excludedPath: ["/v1/showcase/*"],
};
