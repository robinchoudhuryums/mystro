# MasterBench

A personal, local-first mastering, mix-analysis and learning workbench. Audio never leaves
the machine; the opt-in AI notes send numbers only.

- `HANDOFF.md` is the plan: goals, feature specs, phases and acceptance criteria.
- `NOTES.md` records working versions, API differences and what is still unverified.

**Status:** Phases 0 to 6 are done: mastering against a reference or a genre target curve,
processing chains with hosted VST3/AU plugins, stems, batch/EP mode, AI mix notes, and
analysis of loudness, balance, stereo, key, tempo, song sections, the 808 and common mix
problems, with streaming previews, speaker simulation, a level-matched A/B/C player with
blind tests, a progress view, the reference library and the watch folder. Everything is
tested on generated audio; none of it has met real music yet. Phase 7 (MCP server,
alternate versions, habit detection) is next.

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

`init` creates `~/MasterBench` and the SQLite index. To keep the data in iCloud Drive instead:

```bash
uv run masterbench init --icloud
```

`--home PATH` puts it anywhere else. With a synced folder the index stays on the Mac (in
`~/Library/Application Support/MasterBench`) and is rebuilt from the files when missing.
In Finder, set the MasterBench folder to "Keep Downloaded" so audio is always on disk.

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

Run a chain before or after matching, and build one around a plugin:

```bash
uv run masterbench master ~/Music/Bounces/song.wav --pre streaming-safe --post warm
```

```bash
uv run masterbench preset add-plugin my-chain "/Library/Audio/Plug-Ins/VST3/Some Plugin.vst3"
```

`masterbench preset list` shows the presets, `masterbench plugins` the plugins found. Presets
are TOML files in `~/MasterBench/presets/chains/`.

Separate a mix into stems and see how the vocal, drums and bass sit:

```bash
uv run masterbench separate ~/Music/Bounces/song.wav --genre hip-hop
```

Better, when you have them: import stems exported from Logic, then nudge the balance and
master the result as a new run of the same track.

```bash
uv run masterbench stems import ~/Music/Bounces/song-stems --title "Song" --genre hip-hop
```

```bash
uv run masterbench rebalance song --gain lead-vox=+2 --gain drums=-1
```

Build a genre's target curve from its references, then master toward it instead of
toward one song:

```bash
uv run masterbench curve build hip-hop
```

```bash
uv run masterbench master ~/Music/Bounces/song.wav --curve --genre hip-hop
```

Every analysis reports the key and tempo, the song's sections, out-of-tune 808 notes and
moments worth a listen. To look closer, or to correct what was detected:

```bash
uv run masterbench sections RUN_ID
```

```bash
uv run masterbench sections RUN_ID --set "0 intro, 0:14 verse, 0:41 hook, 1:08 verse"
```

```bash
uv run masterbench key night-drive "F minor"
```

```bash
uv run masterbench lowend RUN_ID
```

Hear a master as a phone, earbuds, a car, a laptop or a club system would roughly play it:

```bash
uv run masterbench speakers RUN_ID
```

`masterbench ref suggest song.wav --genre hip-hop` ranks your references by how close they
sit to a mix; `masterbench ref list` shows each reference's quality badge.

A genre can hold a bank of references, say a boom-bap one and a trap one. `--ref auto`
masters against whichever is closest to the mix and says which it picked and why:

```bash
uv run masterbench master ~/Music/Bounces/song.wav --genre hip-hop --ref auto
```

Closeness is judged on tempo, major or minor, tonal balance, dynamics and stereo width.
To make this the default everywhere (mastering, analysis, the watch folder, batch, vocal
mix), set `default_reference = "auto"` in `settings.toml`; a reference you name still wins.

Master an EP the same way and check that the tracks sit together:

```bash
uv run masterbench batch ~/Music/Bounces/EP/*.wav --genre hip-hop --name "My EP"
```

Ask Claude for mix notes on a run. Only measurements are sent, never audio; `--payload`
prints exactly what would go out, with a cost estimate, and sends nothing.

```bash
uv run masterbench notes 20261124-185126-7f55cf --payload
```

```bash
uv run masterbench notes 20261124-185126-7f55cf --intent "Dark and heavy, vocal up front"
```

The key is read from `ANTHROPIC_API_KEY`, or from the macOS Keychain if you store it there:

```bash
security add-generic-password -s masterbench-anthropic -a anthropic -w
```

Make an instrumental and an a cappella that get the same processing as a master. This
needs stems for the track (from Logic, or separated) and reports how closely the two add
up to the master:

```bash
uv run masterbench versions 20261124-185126-7f55cf
```

See what keeps happening across your recent tracks, and ask Claude for the top habits with
an exercise for each (numbers only; `--payload` shows what would be sent):

```bash
uv run masterbench habits
```

```bash
uv run masterbench trends --payload
```

Set a repeatable listening level: this writes 30 seconds of band-limited pink noise at
-20 dBFS RMS and prints the steps. The UI plays the same file in its Settings tab and
reminds you to take a break every 50 minutes.

```bash
uv run masterbench calibrate
```

Let Claude Code query your runs over MCP (read-only unless `mcp_allow_writes = true`):

```bash
claude mcp add masterbench -- uv run --project "$PWD" masterbench mcp
```

The tools are `list_tracks`, `list_runs`, `get_analysis`, `compare_runs`, `analyze_file`,
`suggest_reference`, `pitch_report`, `get_progress_trends`, `get_habits`, and, with writes on,
`master_file` and `run_batch`. Files outside the data folder are refused unless their
folder is listed in `mcp_allowed_dirs`. For the Claude desktop app, add this under
`mcpServers` in its config file:

```json
"masterbench": {
  "command": "uv",
  "args": ["run", "--project", "/path/to/mystro", "masterbench", "mcp"]
}
```

Check how in tune a vocal is, then tune it to the song's key. Use a dry vocal exported
from Logic. The report changes nothing; tuning writes a new file beside a copy of the
vocal, with the same length and start, ready to drag back into the session.

```bash
uv run masterbench vocal report ~/Music/Bounces/song-lead-vox.wav --track "Song"
```

```bash
uv run masterbench vocal tune ~/Music/Bounces/song-lead-vox.wav --track "Song"
```

`--preset tight` or `--preset hard-tune` change the character (`masterbench vocal presets`
lists them); `--amount`, `--retune` and `--humanize` adjust it; `--key "F minor"`,
`--scale dorian` or `--notes "F G# A# C"` set what counts as in tune; `--bypass 3,7` leaves
notes alone and `--set 5=G#3` sends one somewhere else. The Vocal tab does the same with an
A/B player and a piano-roll editor.

With only a full mix, the vocal is separated first and put back after tuning. That is
preview quality, for trying ideas:

```bash
uv run masterbench vocal tune-mix ~/Music/Bounces/song.wav --genre hip-hop
```

Compare the two tuning engines, and the same vocal tuned elsewhere, in a blind test:

```bash
uv run masterbench vocal benchmark ~/Music/Bounces/song-lead-vox.wav --add "Waves=~/Music/Bounces/song-waves.wav"
```

Mix a dry vocal into its instrumental and master the result in one go. Both files must be
bounced from the same point in the project. The vocal goes through a chain (`gentle`,
`modern-rap` or `aggressive`; `masterbench vocal chains` lists them), is ridden so it holds
its level against the beat, and gets reverb and a tempo-synced delay; the beat is dipped
slightly where the vocal lives. The processed vocal and instrumental are written beside
the mix, to take back into Logic.

```bash
uv run masterbench vocal mix ~/Music/Bounces/song-lead-vox.wav ~/Music/Bounces/song-beat.wav --title "Song" --genre hip-hop
```

`--target -1` sets how loud the vocal sits against the beat (in LU), `--ride` how far the
rider may move, `--carve` and `--duck` how much room is made in the beat, `--adlib FILE`
adds ad-libs, and `--notes` asks for AI mix notes on the result. Your own vocal chains are
TOML files in `~/MasterBench/presets/vocal/`; `insert = "name"` in one runs a chain preset
of yours (a Soundtoys plugin, say) after the built-in steps.

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
default_reference = "auto"
watch_auto_master = true

break_interval_min = 45
mcp_allow_writes = true
mcp_allowed_dirs = ["~/Music/Bounces"]

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

The full suite takes about seventeen minutes. `pytest -m "not slow"` skips the test that drives the
third-party libraries.
`verify_stack.py` exercises every library on generated audio and downloads one separation
model (about 65 MB) the first time.

## Layout

```
src/masterbench/
  cli.py            Typer entry point (`masterbench`)
  config.py         paths, targets, thresholds, settings.toml
  audio_io.py       load, save (TPDF dither), hashing
  validate.py       pre-flight checks on a mix
  analysis/         loudness, spectrum, stereo, key and tempo, sections, detectors,
                    low end, flags, AnalysisReport
  master/           Matchering wrapper, curve match, chains and presets, limiter, pipeline
  references/       reference library, quality checks, suggestions, target curves
  projects.py       projects, runs, manifests
  db.py, index.py   SQLite migrations, indexing, reindex
  watch.py          watch folder
  stems/            separation, Logic stem import, balance, rebalance
  preview/          codec round trips, platform normalization, speaker simulation
  track.py          hand-set key and sections, low-end report
  batch.py          EP mode and consistency report
  notes/            AI mix notes: payload, prompt, API call
  notes/trends.py   AI trend summary across tracks
  progress.py       per-track trends, recurring flags, habits, blind-test history
  release/          instrumental and a cappella versions with locked processing
  monitoring.py     calibration noise, listening sessions, break reminders
  mcp_server.py     MCP tools over the engine
  vocal/            pitch tracking, pitch report, tuning (WORLD or Rubber Band), presets,
                    and the vocal mix assistant (chain, de-esser, rider, sends)
  ui/               Gradio app, charts, Web Audio A/B/C player
  fixtures.py       synthetic test audio with known ground truth
  migrations/       numbered SQL files, applied in order
scripts/verify_stack.py   Phase 0 verification spike
scripts/check_player.js   browser check of the A/B/C player
scripts/check_vocal.js    browser check of the Vocal tab and pitch editor
quick_action/             Finder Quick Action script and install steps
.claude/commands/         cycle workflow slash commands (from claude-workflow-tools)
tests/                    generated fixtures only; no audio is committed
```

Matchering is GPLv3. This project is for personal use and is not distributed.
