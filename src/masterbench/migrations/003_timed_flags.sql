-- Detector flags point at a moment in the track.
ALTER TABLE flags ADD COLUMN start_s REAL;
ALTER TABLE flags ADD COLUMN end_s REAL;
