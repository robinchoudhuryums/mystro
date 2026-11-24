# MasterBench

A personal, local-first mastering, mix-analysis and learning workbench. Audio never leaves
the machine; the opt-in AI notes send numbers only.

- `HANDOFF.md` is the plan: goals, feature specs, phases and acceptance criteria.
- `NOTES.md` records working versions, API differences and what is still unverified.

**Status:** Phases 0 to 2 are done: mastering against a reference, loudness, frequency
balance and stereo analysis, streaming previews, a level-matched A/B/C player with blind
tests, a progress view, the reference library and the watch folder. Phase 3 is next.

## Setup (macOS)

```bash
xcode-select --install
```

```bash
brew install uv ffmpeg libsndfile rubberband
```

```bash
uv sync
```

```bash
uv run masterbench init
```

`init` creates `~/MasterBench` and the SQLite index. Set `MASTERBENCH_HOME` to put the data
somewhere else.

## Use

Add a reference track for a genre (the first one becomes that genre's default):

```bash
uv run masterbench ref add ~/Music/reference.wav --genre hip-hop
```

Master a bounce against the genre default, to -14 LUFS and -1 dBTP:

```bash
uv run masterbench master ~/Music/Bounces/song.wav --genre hip-hop
```

Open the web UI (127.0.0.1 only):

```bash
uv run masterbench ui
```

Analyze bounces as they land in `~/MasterBench/Inbox/<genre>/`:

```bash
uv run masterbench watch
```

See what the codecs and Spotify's normalization do to a master:

```bash
uv run masterbench preview 20261124-185126-7f55cf
```

Rebuild the index from the files on disk:

```bash
uv run masterbench reindex
```

Each master is a new folder under `~/MasterBench/projects/<track>/runs/` holding
`master_24.wav`, `master_16.wav`, `run.json` and `analysis.json`. The input is copied into
the project and never modified. `masterbench master --help` lists the options (`--ref`,
`--lufs`, `--ceiling`, `--match-loudness`, `--title`).

Defaults can be changed in `~/MasterBench/settings.toml`:

```toml
default_genre = "house"
watch_auto_master = true

[targets]
lufs_i = -11.0
true_peak_dbtp = -1.0
```

## Checks

```bash
uv run pytest
```

```bash
uv run python scripts/verify_stack.py
```

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

`pytest -m "not slow"` skips the test that drives the third-party libraries.
`verify_stack.py` exercises every library on generated audio and downloads one separation
model (about 65 MB) the first time.

## Layout

```
src/masterbench/
  cli.py            Typer entry point (`masterbench`)
  config.py         paths, targets, thresholds, settings.toml
  audio_io.py       load, save (TPDF dither), hashing
  validate.py       pre-flight checks on a mix
  analysis/         loudness, spectrum, stereo, flags, AnalysisReport
  master/           Matchering wrapper, true-peak limiter, pipeline
  references/       genre reference library
  projects.py       projects, runs, manifests
  db.py, index.py   SQLite migrations, indexing, reindex
  watch.py          watch folder
  preview/          codec round trips and platform normalization
  progress.py       per-track trends, recurring flags, blind-test history
  ui/               Gradio app, charts, Web Audio A/B/C player
  fixtures.py       synthetic test audio with known ground truth
  migrations/       numbered SQL files, applied in order
scripts/verify_stack.py   Phase 0 verification spike
scripts/check_player.js   browser check of the A/B/C player
tests/                    generated fixtures only; no audio is committed
```

Matchering is GPLv3. This project is for personal use and is not distributed.
