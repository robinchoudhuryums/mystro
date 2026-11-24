-- Content hash lets the watch folder and reindex recognise a file it has already seen.
ALTER TABLE analyses ADD COLUMN sha256 TEXT;
ALTER TABLE analyses ADD COLUMN sample_peak_dbfs REAL;
ALTER TABLE analyses ADD COLUMN channels INTEGER;
ALTER TABLE analyses ADD COLUMN bit_depth INTEGER;
CREATE INDEX analyses_sha_idx ON analyses (sha256);

ALTER TABLE reference_tracks ADD COLUMN name TEXT;
ALTER TABLE reference_tracks ADD COLUMN is_default INTEGER NOT NULL DEFAULT 0;
ALTER TABLE reference_tracks ADD COLUMN notes TEXT NOT NULL DEFAULT '';
