-- Pitch reports of vocals, mirrored from projects/<track>/vocal/<hash>/pitch.json.
CREATE TABLE vocal_reports (
    id TEXT PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    notes INTEGER NOT NULL,
    in_tune_share REAL,
    mean_cents REAL,
    mean_abs_cents REAL
);
CREATE INDEX vocal_reports_track ON vocal_reports(track_id, created_at);
