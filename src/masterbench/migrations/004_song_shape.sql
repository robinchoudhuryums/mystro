-- Figures the progress tracker trends: how much the hook lifts, and how in tune the bass is.
ALTER TABLE analyses ADD COLUMN hook_contrast_lu REAL;
ALTER TABLE analyses ADD COLUMN bass_in_tune_share REAL;
