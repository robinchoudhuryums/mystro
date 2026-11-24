-- The DB is an index. Run manifests (run.json, analysis.json) are the source of truth,
-- so every row here must be rebuildable by `masterbench reindex`.

CREATE TABLE tracks (
    id          INTEGER PRIMARY KEY,
    slug        TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    genre       TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- id is the run directory name: "<timestamp>-<short-id>".
CREATE TABLE runs (
    id                 TEXT PRIMARY KEY,
    track_id           INTEGER NOT NULL REFERENCES tracks (id) ON DELETE CASCADE,
    kind               TEXT NOT NULL
                       CHECK (kind IN ('master', 'analysis_only', 'batch', 'version', 'tune')),
    settings_json      TEXT NOT NULL,
    path               TEXT NOT NULL,
    lib_versions_json  TEXT NOT NULL DEFAULT '{}',
    created_at         TEXT NOT NULL
);
CREATE INDEX runs_track_idx ON runs (track_id, created_at);

-- An analysis belongs to a run, or stands alone for a source file (references, inbox files).
CREATE TABLE analyses (
    id              INTEGER PRIMARY KEY,
    run_id          TEXT REFERENCES runs (id) ON DELETE CASCADE,
    source_path     TEXT,
    role            TEXT NOT NULL CHECK (role IN ('mix', 'master', 'reference', 'stem')),
    sample_rate     INTEGER,
    duration_s      REAL,
    lufs_i          REAL,
    lufs_s_max      REAL,
    lufs_m_max      REAL,
    true_peak_dbtp  REAL,
    lra             REAL,
    plr             REAL,
    crest_db        REAL,
    correlation     REAL,
    low_mono_corr   REAL,
    key             TEXT,
    key_confidence  REAL,
    tempo_bpm       REAL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    CHECK (run_id IS NOT NULL OR source_path IS NOT NULL)
);
CREATE INDEX analyses_run_idx ON analyses (run_id);
CREATE INDEX analyses_source_idx ON analyses (source_path);

CREATE TABLE band_deltas (
    analysis_id  INTEGER NOT NULL REFERENCES analyses (id) ON DELETE CASCADE,
    band         TEXT NOT NULL,
    delta_db     REAL NOT NULL,
    vs           TEXT NOT NULL CHECK (vs IN ('reference', 'target_curve')),
    section      TEXT NOT NULL DEFAULT 'full',
    PRIMARY KEY (analysis_id, band, vs, section)
) WITHOUT ROWID;

CREATE TABLE flags (
    id           INTEGER PRIMARY KEY,
    analysis_id  INTEGER NOT NULL REFERENCES analyses (id) ON DELETE CASCADE,
    code         TEXT NOT NULL,
    severity     TEXT NOT NULL CHECK (severity IN ('info', 'warn', 'problem')),
    message      TEXT NOT NULL,
    value        REAL
);
CREATE INDEX flags_analysis_idx ON flags (analysis_id);
CREATE INDEX flags_code_idx ON flags (code);

CREATE TABLE blind_tests (
    id               INTEGER PRIMARY KEY,
    track_id         INTEGER NOT NULL REFERENCES tracks (id) ON DELETE CASCADE,
    candidates_json  TEXT NOT NULL,
    ranking_json     TEXT NOT NULL,
    revealed_json    TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE notes (
    id            INTEGER PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    model         TEXT NOT NULL,
    cost_usd      REAL NOT NULL DEFAULT 0,
    payload_hash  TEXT NOT NULL,
    body_json     TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX notes_payload_idx ON notes (payload_hash);

-- "references" is an SQL keyword, hence the name.
CREATE TABLE reference_tracks (
    id             INTEGER PRIMARY KEY,
    genre          TEXT NOT NULL,
    path           TEXT NOT NULL UNIQUE,
    quality_json   TEXT NOT NULL DEFAULT '{}',
    features_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX reference_tracks_genre_idx ON reference_tracks (genre);

CREATE TABLE sessions (
    id          INTEGER PRIMARY KEY,
    started_at  TEXT NOT NULL,
    calibrated  INTEGER NOT NULL DEFAULT 0 CHECK (calibrated IN (0, 1)),
    minutes     REAL NOT NULL DEFAULT 0
);
