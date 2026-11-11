CREATE TABLE IF NOT EXISTS family_tenants (
  id TEXT PRIMARY KEY,
  family_name TEXT NOT NULL,
  plan_tier TEXT NOT NULL DEFAULT 'familiar_voces',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS parent_users (
  id TEXT PRIMARY KEY,
  family_id TEXT NOT NULL REFERENCES family_tenants(id) ON DELETE CASCADE,
  email TEXT NOT NULL UNIQUE,
  full_name TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  parental_pin_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS consent_audit_logs (
  id TEXT PRIMARY KEY,
  family_id TEXT NOT NULL REFERENCES family_tenants(id) ON DELETE CASCADE,
  parent_user_id TEXT,
  consent_type TEXT NOT NULL,
  statement_sha256 TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS voice_profiles (
  id TEXT PRIMARY KEY,
  family_id TEXT REFERENCES family_tenants(id) ON DELETE CASCADE,
  display_name TEXT NOT NULL,
  is_catalog BOOLEAN NOT NULL DEFAULT FALSE,
  locale TEXT NOT NULL DEFAULT 'es-CL',
  storage_key TEXT NOT NULL DEFAULT '',
  key_salt_hex TEXT NOT NULL DEFAULT '',
  plaintext_sha256 TEXT NOT NULL DEFAULT '',
  consent_log_id TEXT,
  status TEXT NOT NULL DEFAULT 'ACTIVE',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  revoked_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS child_profiles (
  id TEXT PRIMARY KEY,
  family_id TEXT NOT NULL REFERENCES family_tenants(id) ON DELETE CASCADE,
  alias TEXT NOT NULL,
  age_years INTEGER NOT NULL,
  axis_lenguaje TEXT NOT NULL,
  axis_estructura TEXT NOT NULL,
  axis_estimulo_sonoro TEXT NOT NULL,
  axis_foco_tematico TEXT NOT NULL DEFAULT 'interes_profundo',
  special_interests_json JSONB NOT NULL DEFAULT '[]'::jsonb,
  valores_json JSONB NOT NULL DEFAULT '[]'::jsonb,
  temas_evitar_json JSONB NOT NULL DEFAULT '[]'::jsonb,
  closing_phrase TEXT NOT NULL,
  preferred_voice_id TEXT NOT NULL DEFAULT 'catalog-chileno-sereno-m1',
  bible_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  sanitized_terms_blocked_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS story_jobs (
  id TEXT PRIMARY KEY,
  family_id TEXT NOT NULL REFERENCES family_tenants(id) ON DELETE CASCADE,
  child_id TEXT NOT NULL REFERENCES child_profiles(id) ON DELETE CASCADE,
  voice_id TEXT NOT NULL,
  chapter_number INTEGER NOT NULL,
  parent_seed TEXT NOT NULL,
  writer_prompt TEXT NOT NULL,
  bible_spec_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  status TEXT NOT NULL DEFAULT 'QUEUED',
  stage_detail TEXT NOT NULL DEFAULT 'waiting_for_elastic_gpu',
  attempts INTEGER NOT NULL DEFAULT 0,
  leased_by_worker_id TEXT,
  lease_token TEXT,
  lease_expires_at TIMESTAMPTZ,
  slm_model_tag TEXT,
  slm_params_billions DOUBLE PRECISION,
  gpu_seconds_used DOUBLE PRECISION,
  estimated_cost_usd DOUBLE PRECISION,
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS story_chapters (
  id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL REFERENCES child_profiles(id) ON DELETE CASCADE,
  job_id TEXT NOT NULL UNIQUE REFERENCES story_jobs(id) ON DELETE CASCADE,
  chapter_number INTEGER NOT NULL,
  title TEXT NOT NULL,
  story_text TEXT NOT NULL,
  word_count INTEGER NOT NULL,
  szigriszt_score DOUBLE PRECISION NOT NULL,
  fernandez_huerta_score DOUBLE PRECISION NOT NULL,
  whisper_wer DOUBLE PRECISION NOT NULL,
  speaker_similarity DOUBLE PRECISION NOT NULL,
  audio_storage_key TEXT NOT NULL,
  parent_guide_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  qa_report_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO voice_profiles (id, family_id, display_name, is_catalog, locale, status)
VALUES
  ('catalog-chileno-sereno-m1', NULL, 'Narrador Sureño Sereno (Catálogo Ético es-CL)', TRUE, 'es-CL', 'ACTIVE'),
  ('catalog-chilena-calida-f1', NULL, 'Narradora Valle Central Cálida (Catálogo Ético es-CL)', TRUE, 'es-CL', 'ACTIVE')
ON CONFLICT (id) DO NOTHING;
