-- ═══════════════════════════════════════════════════════════════════════════
--  YOLOCheck — Supabase Database Schema
--  Run this once in the Supabase SQL Editor to initialise your database.
-- ═══════════════════════════════════════════════════════════════════════════

-- ── Enable UUID generation ────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── users ─────────────────────────────────────────────────────────────────
-- Mirrors Supabase Auth; extend with profile data as needed.
CREATE TABLE IF NOT EXISTS public.users (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email       TEXT UNIQUE NOT NULL,
    full_name   TEXT,
    avatar_url  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE public.users IS 'Application user profiles.';

-- ── scans ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.scans (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id               UUID REFERENCES public.users(id) ON DELETE SET NULL,
    image_url             TEXT NOT NULL,
    image_filename        TEXT NOT NULL,
    total_moles_detected  INTEGER NOT NULL DEFAULT 0,
    highest_risk          TEXT NOT NULL CHECK (highest_risk IN ('Low', 'Moderate', 'High')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scans_user_id   ON public.scans(user_id);
CREATE INDEX IF NOT EXISTS idx_scans_created_at ON public.scans(created_at DESC);

COMMENT ON TABLE public.scans IS 'One row per uploaded scan image.';

-- ── detections ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.detections (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_id          UUID NOT NULL REFERENCES public.scans(id) ON DELETE CASCADE,
    mole_id          TEXT NOT NULL,                  -- "Mole #1", "Mole #2", …

    -- YOLO output
    confidence       FLOAT NOT NULL,

    -- Bounding box (pixel coordinates, relative to original image)
    bbox_x1          FLOAT NOT NULL,
    bbox_y1          FLOAT NOT NULL,
    bbox_x2          FLOAT NOT NULL,
    bbox_y2          FLOAT NOT NULL,
    bbox_width       FLOAT NOT NULL,
    bbox_height      FLOAT NOT NULL,

    -- ABCD scores (each 0–3)
    abcd_asymmetry   FLOAT NOT NULL,
    abcd_border      FLOAT NOT NULL,
    abcd_color       FLOAT NOT NULL,
    abcd_diameter    FLOAT NOT NULL,
    abcd_total       FLOAT NOT NULL,                 -- sum 0–12

    -- ABCD explanatory notes
    asymmetry_note   TEXT,
    border_note      TEXT,
    color_note       TEXT,
    diameter_note    TEXT,

    -- Risk
    risk_level       TEXT NOT NULL CHECK (risk_level IN ('Low', 'Moderate', 'High')),
    risk_score       FLOAT NOT NULL,

    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_detections_scan_id ON public.detections(scan_id);

COMMENT ON TABLE public.detections IS 'Individual mole detection results within a scan.';

-- ── chat_logs (optional audit trail) ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.chat_logs (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID REFERENCES public.users(id) ON DELETE SET NULL,
    scan_id     UUID REFERENCES public.scans(id) ON DELETE SET NULL,
    user_msg    TEXT NOT NULL,
    ai_reply    TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_logs_user_id ON public.chat_logs(user_id);

COMMENT ON TABLE public.chat_logs IS 'Audit log of AI Health Assistant conversations.';

-- ═══════════════════════════════════════════════════════════════════════════
--  Row-Level Security (RLS)
--  Users can only read/write their own data.
-- ═══════════════════════════════════════════════════════════════════════════

ALTER TABLE public.users      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.scans      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.detections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_logs  ENABLE ROW LEVEL SECURITY;

-- users: read own profile
CREATE POLICY "Users read own profile"
    ON public.users FOR SELECT
    USING (auth.uid() = id);

-- scans: full CRUD on own scans
CREATE POLICY "Users manage own scans"
    ON public.scans FOR ALL
    USING (auth.uid() = user_id);

-- detections: readable if parent scan belongs to the user
CREATE POLICY "Users read own detections"
    ON public.detections FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.scans s
            WHERE s.id = scan_id AND s.user_id = auth.uid()
        )
    );

-- chat_logs: read own messages
CREATE POLICY "Users read own chat logs"
    ON public.chat_logs FOR SELECT
    USING (auth.uid() = user_id);

-- ═══════════════════════════════════════════════════════════════════════════
--  Supabase Storage bucket
--  Run in Supabase Dashboard → Storage → New Bucket
--  OR execute via the Supabase JS/Python client on first deploy.
-- ═══════════════════════════════════════════════════════════════════════════
-- INSERT INTO storage.buckets (id, name, public)
-- VALUES ('scan-images', 'scan-images', TRUE)
-- ON CONFLICT (id) DO NOTHING;
