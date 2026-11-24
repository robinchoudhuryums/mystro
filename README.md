# MasterBench

A personal, local-first mastering, mix-analysis and learning workbench. Audio never leaves
the machine; the opt-in AI notes send numbers only.

- `HANDOFF.md` is the plan: goals, feature specs, phases and acceptance criteria.
- `NOTES.md` records working versions, API differences and what is still unverified.

**Status:** Phase 0 (scaffold and verification spike) is done. Phase 1 is next.

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
  cli.py          Typer entry point (`masterbench`)
  config.py       paths, targets, thresholds, settings.toml
  db.py           SQLite connection and migration runner
  fixtures.py     synthetic test audio with known ground truth
  migrations/     numbered SQL files, applied in order
scripts/verify_stack.py   Phase 0 verification spike
tests/                    generated fixtures only; no audio is committed
```

Matchering is GPLv3. This project is for personal use and is not distributed.
