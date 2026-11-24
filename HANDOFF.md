# Mystro project — Claude Code Handoff

> A personal, local-first mastering, mix-analysis and learning workbench for Robin's own tracks.
> New session: read this file and `NOTES.md`, then continue with the next unfinished phase.

**Status:** Phases 0 to 5 done (2026-11-24); see each phase for what still needs the Mac or an API key. Next: Phase 6.

**Changes made to this plan after the Phase 0 spike** (details in `NOTES.md`; each is also
edited in place below and marked *[P0]*):
- MCP SDK 2.x: `FastMCP` is now `MCPServer`; tools need typed returns for structured output.
- Matchering does not dither its 16-bit output, so we write that file ourselves.
- Pedalboard's `Limiter` is neither a hard ceiling nor true-peak aware, so the final limiter
  needs our own true-peak stage.
- 808 pitch uses pYIN plus a per-note spectral-peak refinement (raw pYIN made octave errors).
- Tempo detection needs a half/double-time rule.
- The `references` table is named `reference_tracks`; migrations live inside the package.
- Python is pinned to 3.12 and everything fits in one uv environment.

**Decisions made in Phase 1** (marked *[P1]* below):
- The final limiter is our own lookahead true-peak limiter (`master/limiter.py`), with a
  static trim that makes the ceiling a guarantee.
- Every master runs Matchering with its limiter off, then gain to the loudness target, then
  our limiter. A target needing more than 6 dB of limiting is not chased.
- Index writes live in `index.py`; `db.py` only connects and migrates.
- Loudness-matched playback is done with gain nodes in the player, so no `preview_lm.wav`
  is written.

**Decisions made in Phase 2** (marked *[P2]* below):
- Spectra and stereo figures come from one set of Welch auto/cross spectra per file. Band
  levels are stored relative to integrated loudness, so any two files compare at equal
  loudness by subtraction.
- The player is a Gradio 6 `gr.HTML` component with `js_on_load`; blind-test results go back
  through its `server_functions`. All sources play at once and switching is a 15 ms gain
  crossfade.
- Platform normalization is reported as numbers and messages, not rendered as audio.
- Blind tests are JSON files under `projects/<slug>/blind_tests/`, indexed like runs.
- Timestamps carry milliseconds so runs made in the same second keep their order.

**Decisions made in Phase 3** (marked *[P3]* below):
- Presets are TOML with `[[module]]` tables. Built-in modules are validated against fixed
  ranges; a `plugin` module stores the bundle path, its parameters and its opaque state.
- Order in a master: pre-chain, Matchering, post-chain, loudness and true-peak limiter. The
  final limiter always runs, so a chain's own limiter is optional.
- The UI edits presets as TOML text and can append a scanned plugin with its current
  parameters. Per-parameter sliders are not built.

**Decisions made in Phase 4** (marked *[P4]* below):
- A stem set is a folder of equal-length float WAVs at the mix's sample rate and level, plus
  `stems.json`, whether it came from a model or from Logic. Everything downstream treats
  the two the same, apart from an artifact warning on separated sets.
- audio-separator is wrapped: it rescales loud stems and always returns 44.1 kHz, so the
  input is staged at -6 dBFS and the stems are scaled, resampled and trimmed back.
- Model weights live in the local state folder, never in the (possibly synced) data root.
- Stem balance is measured in LUFS: vocal against the rest, drums and bass against the mix.

**Decisions made in Phase 5** (marked *[P5]* below):
- A batch masters each track as an ordinary run and writes `batches/<id>/batch.json` with
  the consistency report. Consistency is measured against the set's median.
- AI notes use the Messages API with structured output (`output_config.format`), effort
  `medium`, adaptive thinking left on, and the API's server-side refusal fallback on
  Sonnet. Notes are stored per run (`notes.json`, `notes.md`) and indexed for spend.
- The request carries measurements and labels only: no audio, paths or hashes.
- The repo carries Robin's cycle workflow commands in `.claude/commands/` (copied from
  `claude-workflow-tools` v1.56.0). They need a "Cycle Workflow Config" in a `CLAUDE.md`,
  which `/setup-cycle` produces; none exists yet.

---

## 1. Context

- **User:** Robin, a self-taught producer (hip-hop, plus growing electronic/house work). He uses **Logic Pro on a MacBook Pro** and bounces his own mixes. He masters his own tracks with no formal training. He wants better masters **and** to learn where his mixes fall short, including his habits across many songs.
- **Releases** go to streaming (Spotify etc.), so loudness normalization matters: target around **−14 LUFS integrated, ≤ −1.0 dBTP true peak**, with presets for louder genre targets.
- **Runs only on his own machine:** no cloud processing, unreleased audio never leaves the Mac. The one exception is the opt-in AI features, which send **numbers only, never audio**, to the Anthropic API.
- **Budget:** free or nearly free. The only paid piece is Claude API calls (a few cents per run).
- **Robin is technical** (TypeScript, Python, AWS, Postgres, Anthropic API, MCP servers). Keep the code clean and conventional; no hand-holding comments.

## 2. Goals and non-goals

**Goals**
1. Reference-based mastering with Matchering, wrapped in a convenient local web UI.
2. A **genre reference library** with quality checks, automatic reference suggestions, and **genre target curves** built from several references.
3. **Fair comparisons**: loudness-matched, synced, optionally blind A/B/C playback, plus **streaming previews** and **speaker simulation**.
4. **Objective analysis**: loudness, true peak, dynamics, stereo, **frequency balance vs. the reference or target curve**, analysis **by song section**, **problem detectors**, and **hip-hop low-end tools** (808 tuning, kick/808 clash).
5. A **custom processing chain** (Pedalboard): built-in EQ/compression/limiting plus his own VST3/AU plugins, saved as presets.
6. **Stem tools**: separate a stereo bounce into stems for diagnosis and light rebalancing, and accept real stems exported from Logic (the better path when available).
7. **Batch/EP mode**, plus a consistency report.
8. **AI mix notes**: send the metrics to Claude and get back prioritized, actionable feedback.
9. A **progress tracker** across songs that finds his recurring habits.
10. **Workflow integration**: a CLI, a Finder Quick Action, a **watch folder** for Logic bounces, and an **MCP server** so Claude Code or the Claude app can drive the engine.
11. **Release extras**: instrumental and a cappella versions, plus listening-level calibration and break reminders.
12. **Vocal tuning**: key-aware pitch analysis and correction for vocal stems, with natural correction (default) and a hard-tune effect, and simpler to use than typical graphical tuners.
13. **Vocal mix assistant**: dry vocal + instrumental → processed, balanced vocal mix → mastered track.

**Non-goals (for now)**
- Fully automatic mixing from multitracks (the open-source options are research-grade).
- A native SwiftUI app or multi-user support. (Possible later: put the Gradio app behind Tailscale for phone access.)
- Automatic clean/radio edits (censoring needs edits in the Logic session).
- Distributing the app. Matchering is GPLv3; this is personal use only.

## 3. Stack (verify current versions and APIs before coding)

Working versions are recorded in `NOTES.md` and pinned in `uv.lock`.

| Concern | Choice | Notes |
|---|---|---|
| Python | **3.12** via **uv** *[P0]* | Pinned in `.python-version`. All dependencies resolve in one environment. |
| Reference mastering | **matchering** | `mg.process(target, reference, results=[...])`, `Result(file, subtype, use_limiter=, normalize=)`. *[P0]* The 16-bit path does not dither, and the no-limiter path exceeds 0 dBFS, so write it as float. |
| DSP chain and plugin hosting | **pedalboard** (Spotify) | Built-in EQ/comp/limiter/filters, `load_plugin()` for VST3/AU, `pedalboard.io.AudioFile`. Also used for the speaker simulation curves. *[P0]* `Limiter` is not a ceiling and not true-peak aware. |
| Loudness | **pyloudnorm** | Integrated LUFS. Implement true peak (4× oversample via `scipy.signal.resample_poly`), short-term/momentary max, and LRA yourself. Cross-check against `ffmpeg -af ebur128=peak=true`. *[P0]* Both agree with ffmpeg on the fixtures (0.06 LU, 0.10 dB). |
| Spectral, tempo, key | numpy/scipy + **librosa** | Welch PSD with 1/6-octave smoothing; chroma-based key estimation (Krumhansl-Schmuckler profiles); tempo/beat tracking; structural segmentation. Optionally compare key accuracy against **essentia** if it installs cleanly on macOS. *[P0]* `beat_track` reported half time on the fixture loop; add a half/double-time rule. |
| Pitch tracking | **pYIN** (librosa) to start; evaluate **torchcrepe** (or another maintained CREPE port) for vocals | Used for 808 pitch detection and vocal tuning. *[P0]* For 808s, refine each note with a spectral peak (see 5.17). |
| Pitch shifting | **pyworld** (WORLD vocoder) and/or **pyrubberband** + `brew install rubberband` (R3 engine, formant preservation) | Benchmark both on vocals; formant preservation is required. *[P0]* pyworld needs `setuptools<81` and pads its output; Rubber Band R3 is `rbargs={"--fine": "", "--formant": ""}`. |
| Stem separation | **audio-separator** (`audio-separator[cpu]` on Apple Silicon; CoreML accel via onnxruntime) | Wraps the UVR models (MDX-Net, VR, Demucs, MDXC/Roformer). Benchmark a Roformer vocal model vs `htdemucs_ft`. The original facebookresearch/demucs repo is archived; the maintained fork is adefossez/demucs. Optional Apple-native alternative: `mlx-audio-separator`. *[P0]* Needs `audioread` declared explicitly. |
| Codec previews | **ffmpeg** (Homebrew) | Ogg Vorbis (Spotify-like), AAC 256 (Apple-like), Opus. Check which encoders the Homebrew build includes; prefer `aac_at` when present. |
| Database | **SQLite** (stdlib `sqlite3`) with numbered SQL migration files | An index of projects, runs, analyses and blind tests. The JSON run manifests stay the source of truth; `masterbench reindex` rebuilds the DB from them. |
| Watch folder | **watchdog** | FSEvents backend on macOS. |
| MCP server | Official **MCP Python SDK** 2.x (`MCPServer`, formerly FastMCP) *[P0]* | stdio transport, for Claude Code and the Claude desktop app. Typed tool returns give structured output. |
| UI | **Gradio** (Blocks) | Bind to `127.0.0.1` only. Custom `gr.HTML` + Web Audio JS for the A/B player, section markers and the pitch editor. |
| CLI | **Typer** | `masterbench master|analyze|batch|separate|tune|versions|reindex|watch|mcp|ui`. |
| AI notes | **anthropic** SDK | Key from `ANTHROPIC_API_KEY` or macOS Keychain. Default model `claude-sonnet-5-5`; let the user switch to `claude-haiku-5-5` for cheaper runs. |
| Notifications | `osascript` (display notification) or `pync` | For the watch folder, Quick Action, batch completion and break reminders. |
| Plots | Plotly | Spectrum overlays, band deltas, loudness-over-time, trends, pitch curves. |
| System deps | Homebrew: `ffmpeg`, `libsndfile`, `rubberband`; Xcode Command Line Tools | *[P0]* `pyworld` and `diffq` compile from source on macOS arm64. |
| Quality | ruff, pyright (basic), pytest | |

## 4. Architecture

Keep the **engine** (pure Python, no UI imports) separate from the **interfaces** (Gradio, CLI, Quick Action, watch folder, MCP). Every interface calls the same engine functions.

*[P0]* The repo root is the project root. Files that exist today are marked ✓.

```
pyproject.toml           ✓
uv.lock                  ✓
scripts/verify_stack.py  ✓ Phase 0 verification spike
scripts/check_player.js  ✓ [P2] browser check of the player (needs a running UI)
src/masterbench/
  migrations/            ✓ 001_init.sql, 002_..., applied in order (packaged)
  config.py            ✓ paths, defaults, targets; loads ~/MasterBench/settings.toml
  db.py                ✓ SQLite connection and migrations
  index.py             ✓ [P1] writes runs/analyses/references to the DB; reindex; queries
  fixtures.py          ✓ synthetic test audio with ground truth
  cli.py               ✓ version, init, fixtures, master, analyze, reindex, watch, ui, ref
  audio_io.py          ✓ load (ffmpeg fallback), info, save with TPDF dither, hashing
  validate.py          ✓ pre-flight checks on input mixes (5.1)
  summary.py           ✓ [P1] text/Markdown run summaries for CLI, UI, notifications
  notify.py            ✓ [P1] macOS notifications (terminal-notifier or osascript)
  analysis/
    flags.py           ✓ [P1] Flag dataclass
    loudness.py        ✓ LUFS (I/S/M), true peak, LRA, PLR/crest, short-term curve
    spectrum.py        ✓ smoothed spectrum, band energies, deltas vs reference
                       # still to add: deltas vs target curve (Phase 6)
    stereo.py          ✓ correlation (overall, per band), side/mid, low-end mono check
    compare.py         ✓ [P2] band and stereo flags in plain language
    musical.py         # key, tempo, beat grid (shared by 808 tools, vocal tuning, ref matching)
    sections.py        # automatic section detection + manual markers (5.15)
    detectors.py       # sibilance, resonances, mud, harshness, vocal consistency (5.16)
    lowend.py          # 808 pitch vs key, kick/808 clash (5.17)
    report.py          ✓ AnalysisReport: loudness, spectrum, bands, stereo, deltas, flags
  master/
    matchering_engine.py  ✓ wrapper around mg.process with our options
    limiter.py         ✓ [P1] lookahead true-peak limiter and loudness targeting
    curve_match.py     # matching EQ toward a target curve, deterministic, linear phase (5.14)
    chain.py           ✓ built-in modules, VST3/AU hosting, plugin scan, state capture
    presets.py         ✓ load/save chain presets (TOML); packaged starters in presets/
    pipeline.py        ✓ validate -> pre-chain -> Matchering -> post-chain -> limit ->
                       #   export -> analyze -> index. Still to add: curve match (Phase 6)
  preview/
    streaming.py       ✓ codec round trips, post-encode peaks, platform gains (5.11)
    speakers.py        # phone/earbuds/car/laptop/club/mono simulation (5.12)
  references/
    library.py         ✓ reference CRUD; caches each reference's analysis
    quality.py         # detects lossy rips, brickwalled or clipped references (5.13)
    suggest.py         # suggests the best-matching reference for a mix (5.13)
    target_curve.py    # genre target curves from several references (5.14)
  stems/
    separate.py        ✓ audio-separator wrapper, cache per input hash and model, Logic import
    rebalance.py       ✓ per-stem gain (±6 dB) and chain, re-sum, headroom trim
    stem_analysis.py   ✓ vocal-to-beat, drums and bass vs mix, comparison with a reference
    workflow.py        ✓ [P4] separate / import / diagnose / rebalance-and-master
  vocal/
    pitch_track.py     # f0 tracking, note segmentation
    tuning.py          # scale snapping, retune speed, humanize, formant-preserving shift (5.23)
    pitch_report.py    # cents-off per note, out-of-key notes, read-only report
    presets/           # natural.toml, tight.toml, hard_tune.toml
    mix_assistant.py   # vocal chain, de-esser, rider, dynamic EQ carve, sum (5.24)
  release/
    versions.py        # instrumental / a cappella with locked processing (5.20)
  progress.py          ✓ per-track points, recurring flags, blind-test history
                       # still to add: habit detection, AI summary (Phase 7)
  blind.py             ✓ [P2] blind-test records (JSON, source of truth)
  monitoring.py        # calibration tone, session timer, break reminders (5.21)
  projects.py          ✓ project and run storage, run manifests
  batch.py             ✓ EP mode and consistency report
  watch.py             ✓ watch folder: debounce, analyze, optional auto-master, notify
  mcp_server.py        # MCP tools over the engine (5.22)
  notes/
    mix_notes.py       ✓ payload, estimate and cap, API call, cache, render
    prompt.md          ✓ versioned prompt template
    trends_prompt.md   # versioned prompt for the progress summary
  ui/
    app.py             ✓ Gradio Blocks app: Master, Compare, Analyze, Progress, Library
    player.py          ✓ [P2] player component, sources per run, blind-test saving
    charts.py          ✓ [P2] Plotly figures
    ab_player.js       ✓ synced, gain-matched A/B/C player (Web Audio)
    pitch_editor.js    # note-level pitch view and editor
quick_action/          ✓ Automator script + install instructions (untried on macOS)
launchd/               # optional LaunchAgent plist for the watch folder
tests/                 ✓ fixtures are generated per test session, never Robin's music
```

### Data layout (outside the repo)

The root is `$MASTERBENCH_HOME` when set, else the folder saved by `masterbench init --home PATH` (or `--icloud`), else `~/MasterBench`. When the root is in a synced folder, `masterbench.db` lives in the local state folder instead.

```
~/MasterBench/
  settings.toml
  masterbench.db               # SQLite index (rebuildable)
  Inbox/<genre>/               # default watch folder (genre inferred from subfolder)
  references/<genre>/<file> + <file>.analysis.json  (cached analysis, notes)
  references/defaults.json     # default reference per genre
  references/<genre>/.stems/   # separated stems of references (cached)
  target_curves/<genre>.json
  presets/chains/<name>.toml
  batches/<id>/batch.json      # runs in the set, failures, consistency report
  projects/<track-slug>/
    track.json                 # slug, title, genre; source of truth for the tracks table
    blind_tests/<id>.json      # one finished blind test each
    input/<original bounce>    # same name + different content gets a hash suffix
    stems/<model>-<hash> or logic-<hash>/     <stem>.wav, stems.json, diagnosis.json,
                                              rebalanced-<settings hash>.wav
    runs/<timestamp>-<short-id>/
      run.json        # all settings: reference/curve, chain preset, targets, lib versions
      master_24.wav   master_16.wav
      previews/       # codec round trips + streaming.json (cached, deletable);
                      # speaker-sim renders join them in Phase 6
      analysis.json   notes.json + notes.md (if generated)
    versions/<run-id>/instrumental_24.wav, acapella_24.wav
    vocal/<take>/tuned_<settings-hash>.wav, pitch_report.json
```
Make every run reproducible from `run.json`. Never overwrite the input file.

### Database schema (initial; extend through migrations)

Implemented in `src/masterbench/migrations/` (001 init; 002 adds content hashes and reference names).

- `tracks` (id, slug, title, genre, created_at)
- `runs` (id = run folder name, track_id, kind [master|analysis_only|batch|version|tune], settings_json, path, created_at, lib_versions_json)
- `analyses` (id, run_id or source_path, role [mix|master|reference|stem], lufs_i, lufs_s_max, lufs_m_max, true_peak_dbtp, lra, plr, crest_db, correlation, low_mono_corr, key, key_confidence, tempo_bpm, …)
- `band_deltas` (analysis_id, band, delta_db, vs [reference|target_curve], section [full|hook|verse|drop|…])
- `flags` (analysis_id, code, severity [info|warn|problem], message, value)
- `blind_tests` (id, track_id, candidates_json, ranking_json, revealed_json, created_at)
- `notes` (id, run_id, model, cost_usd, payload_hash, body_json)
- `reference_tracks` (id, genre, path, quality_json, features_json) *[P0: `references` is an SQL keyword]*
- `sessions` (id, started_at, calibrated bool, minutes)

## 5. Feature specs

### 5.1 Input validation (runs on every upload)
- Accept WAV/AIFF (preferred) and FLAC. Allow MP3/AAC **as references only**; if an MP3 is used as the *target*, show a warning that explains why (lossy artifacts get exaggerated by mastering, and the file gets encoded again on streaming).
- Report sample rate, bit depth, duration, channels, and peak/true peak.
- Warn when the mix looks already mastered or limited: integrated loudness above about −12 LUFS, true peak near 0 dBFS, or very low crest factor. Suggest "bounce from Logic with the master bus bypassed."
- Warn on clipping (runs of consecutive full-scale samples), DC offset, or large silence at the head or tail.

### 5.2 Mastering pipeline
1. Optional **pre-chain** (Pedalboard preset).
2. **Matchering** against the chosen reference, **or** **curve match** toward a genre target curve (5.14). When a post-chain limiter is enabled, run Matchering with its limiter **off** so audio isn't limited twice. *[P0]* That output exceeds 0 dBFS, so keep it in float until the limiter.
3. Optional **post-chain**: tone or saturation, then a final limiter with a true-peak ceiling (default −1.0 dBTP). *[P1]* The final limiter is `master/limiter.py`: linked-channel gain from the 4×-oversampled peak, held over a 5 ms lookahead, released at 40 dB/s, smoothed, then a static trim so the ceiling always holds (tested). `finalize()` sets make-up gain for the loudness target and re-runs the limiter until it lands within 0.05 LU, backing off when that would need more than 6 dB of reduction. Pedalboard's `Limiter` is not used for the ceiling.
4. Export 24-bit WAV (main) and 16-bit dithered WAV. *[P0]* Matchering's 16-bit path does not dither; write the 16-bit file ourselves with TPDF dither.
5. Analyze the output, write `analysis.json`, and index it in SQLite.
6. Run jobs in a background worker and show the progress of each stage.

### 5.3 Reference library
- Genre folders (hip-hop, house, electronic, plus custom genres), with optional notes on each reference.
- Cache each reference's analysis, quality check (5.13) and features on add.
- Pickers are dropdowns grouped by genre, with a default reference per genre and a **"Suggested"** section at the top (5.13).

### 5.4 Analysis (core)
- **Loudness:** integrated LUFS, short-term max, momentary max, LRA, true peak, PLR.
- **Dynamics:** crest factor, and a loudness-over-time plot.
- **Stereo:** correlation (overall and per band), side/mid ratio, and a **low-end mono check** (<120 Hz).
- **Musical:** key (with a confidence score), tempo, beat grid. *[P0]* Resolve half/double-time tempo with a genre prior or onset density.
- **Frequency balance:** compare against the reference **or** the genre target curve, at **equal integrated loudness**, smoothed to 1/6 octave. Report dB deltas in named bands (defined in `config.BANDS`):
  - Sub 20–60 Hz
  - Bass 60–250 Hz
  - Low-mids 250–500 Hz
  - Mids 500 Hz–2 kHz
  - Upper-mids 2–6 kHz
  - Presence 6–10 kHz
  - Air 10–20 kHz
- **Plain-language flags**, with thresholds in config.

### 5.5 A/B/C player (the centerpiece)
- Load any combination: input mix, masters from several runs, the reference, stem solos, streaming previews (5.11) and speaker sims (5.12).
- **Loudness-matched playback by default** (preview gain only, never applied to exports), with a toggle to hear the real levels.
- **Synced position**, keyboard shortcuts 1–4, space to play/pause, loop regions, and **jump-to-section** buttons (5.15).
- **Blind mode:** shuffle the labels, rank them, reveal, and save to `blind_tests`.
- Implement as a custom `gr.HTML` component with a Web Audio JS player (shared clock, per-source gain nodes).

### 5.6 Custom chains (Pedalboard)
- Built-in modules: high-pass, shelves, peak EQ, compressor, gain, saturation, limiter. *[P3]* Saturation is our own level-compensated tanh with a dry/wet mix; the limiter module is the true-peak limiter from 5.2.
- **Third-party plugins:** scan the standard AU/VST3 folders, load them with `load_plugin`, expose parameters as UI controls, and save plugin state in presets. *[P3]* Done for VST3 on Linux: parameters are listed with ranges and units (`masterbench plugins --params`), captured into the preset with `raw_state`, and restored on load. Mono-only plugins run once per channel. A plugin that returns NaN stops the chain with an error. Controls in the UI are TOML text, not sliders.
- **Risk:** `show_editor()` needs the main thread. Ship parameter sliders first; a native editor is a stretch goal (side process).
- Presets are TOML files, used as the pre-chain or post-chain.

### 5.7 Stems
- **Separate** into 4 stems (vocals/drums/bass/other) or 2 (vocals/instrumental), with a selectable model. Cache by input hash and model. *[P4]* Models: `mdx-2` (default, small), `roformer-2`, `demucs-4`, `demucs-4-fast`.
- **Diagnose** stem balance against the reference's stems (vocal-to-instrumental, drums-to-total, bass-to-total).
- **Rebalance:** per-stem gain (±6 dB) and simple EQ, re-summed with headroom management, then sent to mastering. Warn clearly about separation artifacts.
- **Real stems:** import a folder of Logic-exported stems. Prefer this path whenever stems exist; it also powers alternate versions (5.20) and vocal tuning (5.23).

### 5.8 Batch / EP mode
- Several tracks, one reference or curve (or one per track), and one chain preset.
- **Consistency report:** LUFS spread, true peaks, band deltas between tracks, section loudness (hook vs hook), with outliers flagged.

### 5.9 AI mix notes (opt-in, numbers only)
- Input: the `AnalysisReport`s (mix, master, reference/curve), band deltas (full and by section), detector flags, low-end findings, stem diagnostics, genre, and Robin's optional free-text intent.
- **Never upload audio.** Show the exact payload in a "what gets sent" panel.
- Output as structured JSON, then rendered: top 3–5 issues ranked by impact, each with *what / evidence / fix in the mix (Logic-specific where reasonable) / fix in the master*, plus "what's already good."
- Cache by payload hash, show the estimated cost, apply a monthly spend cap, and keep the prompt versioned. Tell the model not to invent measurements.
- *[P5]* Implemented in `notes/mix_notes.py`; `masterbench notes RUN_ID --payload` prints exactly what would be sent. The key comes from `ANTHROPIC_API_KEY`, an `ant auth login` profile, or the Keychain item `masterbench-anthropic`. Never run against the live API yet.

### 5.10 CLI and Finder Quick Action
- `masterbench master path/to/bounce.wav --ref hiphop/default --preset streaming-safe` writes a run and prints a summary.
- `masterbench ui` launches Gradio and opens the browser.
- Quick Action: right-click a WAV, choose "Master with MasterBench", and get a macOS notification when done.

### 5.11 Streaming preview
- **Codec preview:** encode the master as Ogg Vorbis ~320 kbps (Spotify-like), AAC 256 (Apple-like) and Opus. Decode back and add each to the A/B player.
- **Peaks after encoding:** measure the true peak of each decoded file. Flag when encoding pushes peaks above −1 dBTP (or to clipping), and suggest a lower ceiling (e.g. −1.5 or −2 dBTP) for loud masters.
- **Platform normalization:** simulate Spotify's Loud (−11 LUFS), Normal (−14) and Quiet (−19) settings, plus a generic −14 profile for other platforms. Show how much gain each platform would apply ("Spotify will turn this down 4.2 dB"). *[P4]* Checked against Spotify's "Loudness normalization" support article on 2026-11-24: louder masters are turned down to −14; quieter ones are turned up only while 1 dB of peak headroom remains (their example: −20 LUFS with a −5 dBFS true peak plays at −16); Loud (−11) applies the full gain with a limiter (−1 dB, 5 ms attack, 100 ms decay); Normal is −14 and Quiet −19; albums played in order get one gain for the whole album (not modelled yet; relevant to Phase 5); the web player and third-party devices do not normalize. Their mastering tips: −14 LUFS integrated and under −1 dBTP, or under −2 dBTP when louder than −14. A master louder than −14 with a peak above −2 dBTP now gets a `loud_for_spotify` flag.
- Cache the renders under `runs/<id>/previews/`.

### 5.12 Speaker simulation
- Approximate profiles built from Pedalboard filters/EQ, with optional mono summing:
  - **Phone speaker:** steep high-pass around 300–500 Hz, upper-mid resonance, mono
  - **Earbuds:** mild bass lift, treble peaks
  - **Car:** cabin low-end boost, slight treble roll-off
  - **Laptop:** high-pass around 150–200 Hz, narrow width
  - **Club/PA** (for house): extended sub emphasis
  - **Mono**
- Add each profile as a player source, with level matching. Label them clearly as **approximations**, not replacements for real checks.
- Pair each sim with analysis: e.g. "with phone sim, the 808 fundamental is gone; perceived bass relies on harmonics at X Hz. Consider saturation."
- Profiles live in TOML so Robin can tune them.

### 5.13 Reference quality checks and suggestions
- **Quality check** on add:
  - lossy-rip detection: a persistent spectral cutoff around 15–17 kHz, or the kind of holes codecs leave
  - heavy brickwall limiting or clipping
  - mono or fake-stereo files
  - very short files
- Show a quality badge, and exclude low-quality references from target curves by default.
- **Suggestions:** build a feature vector per reference (tempo, key or mode, band balance, LUFS, crest, stereo width). Rank the library by similarity to the input mix within its genre (cosine distance on normalized features), and show the top 3 with the reasons ("closest tempo and low-end balance").

### 5.14 Genre target curves
- Pick N references in a genre, normalize each to equal loudness, and compute the **median spectrum** plus a **25th–75th percentile band**. Save the result to `target_curves/<genre>.json`.
- **Analysis use:** plot the mix against the median and the band. A delta only counts as an issue when the mix falls **outside the band**, which is steadier than matching one song.
- **Mastering use (curve match mode):** design a smoothed matching EQ (limited to about ±4 dB, linear-phase FIR), then gain and the post-chain limiter. It's deterministic and repeatable, and it's the basis for locked processing (5.20). Offer it as an alternative to Matchering and in comparisons.
- Track section-level target curves too, when sections are labeled (hook/drop).

### 5.15 Analysis by song section
- **Automatic sections:** combine librosa structural segmentation with loudness to propose sections (intro/verse/hook/drop/bridge/outro). Always find the **loudest 30-second window** (usually the hook or drop) and the quietest full section.
- **Manual markers:** drag markers on a waveform in the UI. Save them to the project and reuse them on later runs of the same track (aligned by duration and cross-correlation).
- Run loudness, band deltas, stereo and the detectors **per section**. Compare hook to reference hook, drop to drop.
- Flags such as "hook is only 0.8 LU louder than verse (reference: 2.5 LU)" or "drop has 2 dB less sub than the reference drop."

### 5.16 Problem detectors
- **Sibilance:** short-time energy spikes in 5–10 kHz on the vocal stem (or the full mix if there are no stems), with timestamps and severity.
- **Resonances:** narrow peaks rising above the smoothed spectrum that persist over time, reported as frequency, Q estimate and gain.
- **Mud:** excess 200–500 Hz compared with the target curve band, per section.
- **Harshness:** excess 2–5 kHz, with short-term spikes.
- **Vocal level consistency:** the vocal stem's short-term loudness relative to the instrumental over time. Flag phrases that drop or jump by more than a set number of dB.
- **Phase:** correlation going negative in any band, or a low-end correlation problem.
- Each detector produces flags with timestamps. The UI lists them and clicking one jumps the player there.

### 5.17 Hip-hop low-end tools
- **808 tuning:**
  1. Isolate the bass (separated bass stem, Logic bass stem, or a low-pass under 150 Hz).
  2. Track the fundamental (pYIN, 25–150 Hz) and segment it into notes.
  3. *[P0]* Refine each note's pitch with a zero-padded FFT peak near the pYIN estimate and its octave above. Raw pYIN slipped an octave on some notes and read about 5 cents sharp; the refined values were within 0.1 cents on the fixture. Reference code: `refine_pitch` in `scripts/verify_stack.py`.
  4. Map each note to a pitch and compare it with the detected key (5.4).
  5. Report out-of-key or detuned notes (cents off) with timestamps, plus an overall "808 tuned to key" verdict.
- **Kick/808 clash:**
  1. Detect kick hits from the drums stem.
  2. At each hit, measure overlapping 40–120 Hz energy from the bass/808.
  3. Report the clash amount and the frequencies involved.
  4. Suggest fixes (sidechain, EQ carve, retuning, kick choice).
- **Sub mono and harmonics:** confirm the sub is mono below 120 Hz, and estimate how much of the bass will survive small speakers (pairs with 5.12).

### 5.18 Progress tracker
- Uses the SQLite index. Views:
  - band deltas vs. target curve per track over time
  - recurring flags (frequency and trend)
  - section loudness patterns
  - 808 tuning and clash history
  - blind-test results (how often Robin's own master beats the AI masters, and how often he picks the reference-closest version)
- **Habit detection:** a rule-based pass that finds consistent biases ("sub under the target band on 7 of the last 8 tracks", "upper-mids over on most vocal-heavy tracks"), with a confidence score based on sample count.
- Optional **AI trend summary** (numbers only, versioned prompt): "your top 3 habits, and one practice exercise for each."

### 5.19 Watch folder
- Watches `~/MasterBench/Inbox/` (and optionally Robin's Logic bounce folder). The subfolder name sets the genre.
- Debounce until the file size is stable (Logic writes in chunks), then validate and analyze automatically. Optionally auto-master with the genre's default reference or curve (off by default).
- macOS notification with the headline flags and a link that opens the run in the UI.
- Run it with `masterbench watch`, or install it as a LaunchAgent (plist in `launchd/`).

### 5.20 Alternate versions (instrumental / a cappella)
- Requires stems from Logic (preferred) or separated stems (flagged as lower quality).
- **Locked processing:** alternate versions must get the *same* processing as the main master. Matchering is adaptive, so for these:
  - **Curve-match runs:** apply the identical matching EQ, gain and chain.
  - **Matchering runs:** derive a static transfer function (master spectrum ÷ mix spectrum, smoothed into a linear-phase FIR, plus gain) and apply it with the same limiter settings. Check that (instrumental + a cappella) before the limiter is close to the main master, and report the error.
- Export `instrumental_24.wav` and `acapella_24.wav`, with loudness and peak reports. A cappella versions skip heavy limiting by default.

### 5.21 Listening level and breaks
- **Calibration:** play pink noise at a set reference level (e.g. −20 dBFS RMS), and guide Robin to set his monitors or headphones to a consistent level (about 75–80 dB SPL measured with a phone SPL meter app). Save "calibrated" for the session.
- **Session timer:** remind him to take a break every 45–60 minutes (configurable), and log session length.
- Suggest a **fresh-ears re-check**: queue a blind A/B of today's final picks for the next session.

### 5.22 MCP server
- `masterbench mcp` runs a stdio MCP server built on the engine. *[P0]* Use `MCPServer` from `mcp.server.mcpserver` (SDK 2.x) and give every tool a typed return so clients get structured output. Tools:
  - `list_tracks`, `list_runs`
  - `analyze_file`, `get_analysis`, `compare_runs` (band and section deltas, flags)
  - `master_file` (always creates a new run, never overwrites)
  - `suggest_reference`
  - `get_progress_trends`, `get_habits`
  - `pitch_report`
  - `run_batch`
- Read-only by default. Tools that write need a config flag (`mcp_allow_writes`). File access is limited to `~/MasterBench` and explicitly allowed folders.
- Document the setup for Claude Code (`claude mcp add`) and the Claude desktop app config.
- Return compact JSON summaries, with file paths instead of audio.

### 5.23 Vocal tuning
- **Input:** a dry vocal stem from Logic (best), or a separated vocal (lower quality: reverb and bleed confuse pitch tracking; warn about this).
- **Full-mix path:** pitch correction can't be applied to a full mix directly, because it would retune the beat too. The path is: separate (5.7) into vocal and instrumental, tune the separated vocal, then recombine. Label the result **"preview quality"**: tuning makes separation artifacts and reverb smear more audible. It's good for the pitch report and for auditioning ideas, not for release. Recommend re-exporting a dry vocal from Logic for the real version.
- **Modes and presets (default = Natural):**
  - **Natural:** retune speed of about 80–150 ms, humanize on, amount about 70–85%, vibrato kept
  - **Tight:** about 20–40 ms, humanize partly on
  - **Hard-tune effect:** 0 ms, humanize off, amount 100%, hard note transitions; optional formant shift and "note hold" for the stepped effect
  - **Custom:** every control exposed

  Presets are TOML like chain presets, and Robin can save his own.
- **Simple by default:** one screen with the detected key (with confidence and an override), the preset picker, an amount slider, and an A/B against the original. The pitch editor and the advanced controls sit behind an "Advanced" toggle. Ease of use is an explicit goal; Robin has found graphical tuners hard to navigate.
- **Key and scale:** the detected song key (5.4), with manual override, scale choice (major, minor, modes, chromatic) and custom note sets.
- **Pitch report (read-only, default first step):**
  - f0 curve
  - segmented notes
  - cents-off per note relative to the scale
  - % of time within ±20 cents
  - the worst phrases with timestamps

  This also feeds the progress tracker.
- **Correction:**
  - snap each note's center pitch to the nearest scale note
  - **retune speed** (how fast pitch moves to the target: 0 ms gives the hard "auto-tune" effect, 50–200 ms sounds natural)
  - **humanize** (keeps vibrato and natural drift, only corrects sustained deviation)
  - **amount** (0–100%)
  - per-note bypass
  - **formant preservation** is required
- **Engines:** benchmark the WORLD vocoder (pyworld) against Rubber Band R3 (pyrubberband) on Robin's vocals, and keep the better one as default. *[P0]* WORLD output comes back padded (220 samples on the fixture); trim it to the input length. Rubber Band returned the exact length.
- **Pitch editor:** a piano-roll view of detected notes over the f0 curve. Click a note to bypass it, change its target, or drag it. Implement in `pitch_editor.js`.
- **Output:** `tuned_<settings-hash>.wav`, same length and start, ready to drag back into Logic. Add the original vs tuned A/B in the player.
- **Benchmark against commercial tuners:** Robin has used a Waves tuning plugin (likely Waves Tune LT). Phase 8 includes a blind A/B of our Natural and Hard-tune output against his Waves result and Logic's Flex Pitch on the same vocal. Optionally, try hosting his Waves tuner through Pedalboard: Waves ships plugins inside a WaveShell, so check `load_plugin`'s `plugin_name` support. Graphical-capture tuners like Tune LT probably can't run headless; real-time auto tuners might.
- Position it honestly: Logic's Flex Pitch and Pitch Correction remain the tools for detailed manual work. This feature's value is ease of use, the pitch report, quick scale-aware correction, and the learning data.

### 5.24 Vocal mix assistant (dry vocal + instrumental → mixed → mastered)
- **Input:** a dry vocal (tuned or not) plus the instrumental, both from Logic and aligned to the same start. Separated stems are allowed, labeled preview quality. Optional ad-lib or backing-vocal stems.
- **Vocal chain** (Pedalboard; built-in modules first, then Robin's plugins via presets):
  1. high-pass
  2. corrective EQ seeded from the resonance detector (5.16)
  3. **de-esser** (Pedalboard has no built-in, so build a band-limited compressor keyed on 5–10 kHz, or host a plugin)
  4. compression (with "gentle / modern rap / aggressive" presets)
  5. tone EQ
  6. saturation
  7. reverb and delay as parallel sends with tempo-synced delay times from the beat grid (5.4)
- **Vocal rider:** automatic gain so the vocal sits at a **target vocal-to-instrumental ratio over time**. Take the target from the reference's separated stems (5.7) or a genre default. Cap the gain changes to a configurable range and smooth them.
- **Making space in the beat:** a dynamic EQ dip (about 1–4 kHz, a few dB) on the instrumental, triggered by the vocal. An optional broadband sidechain duck for hooks.
- **Ad-libs:** level offset, width and pan presets relative to the lead.
- **Output:** a mixed stereo file, then the standard mastering pipeline (5.2). Save every setting in `run.json`. Export the processed vocal and instrumental separately too, so Robin can pull them back into Logic.
- **Feedback loop:** run the analysis, detectors and AI notes on the result, and show the vocal-to-instrumental ratio against the reference by section.
- Positioning: a fast "rough to release-candidate" path for vocal-plus-beat tracks. Logic stays the place for creative mixing decisions.

## 6. UI tabs (Gradio Blocks)
1. **Master:** select a mix (or pick from the watch-folder inbox); the suggested reference or genre curve; pre/post presets; target; run; results, flags and the player.
2. **Compare:** the full A/B/C player across runs, references, stems, streaming previews and speaker sims; blind mode; section jumps.
3. **Analyze:** spectrum vs reference or curve (with the percentile band), band deltas (full and by section), section markers, detector flags, low-end tools, and "Get AI mix notes."
4. **Stems:** separate or import from Logic, diagnose, rebalance, send to Master.
5. **Vocal:** a simple view (key, preset, amount, A/B) and an Advanced view (pitch editor, all controls); the full-mix "separate → tune → recombine" path; and the **Vocal Mix** sub-tab (5.24) with vocal chain presets, rider target, beat carve, mix → master.
6. **Release:** alternate versions, streaming preview summary, final loudness/peak checklist.
7. **Batch:** EP runs and the consistency report.
8. **Progress:** trends, habits, blind-test history, AI trend summary.
9. **Library:** references (quality badges), target curves, chain presets, plugin scan, speaker-sim profiles.
10. **Settings:** paths, watch folder, targets, thresholds, API key status, spend cap, models, MCP write permission, calibration and break timer.

## 7. Phases and acceptance criteria

**Phase 0: Scaffold and verify — DONE (2026-11-24)**
- uv project, src layout, ruff/pytest/pyright, `masterbench` entry point, SQLite migrations runner plus `001_init.sql`.
- **Verification spike** (`scripts/verify_stack.py`): exercises matchering, pedalboard (including loading one AU/VST3 if present), pyloudnorm, librosa (key and tempo), audio-separator, pyworld and pyrubberband, the MCP SDK, and the ffmpeg codecs, all on generated audio. Working versions and API differences are in `NOTES.md`.
- Test-fixture generator (`masterbench.fixtures`): pink noise, a sine sweep, a synthetic drum/808 loop **in F minor, with the bar-6 808 note detuned +35 cents**, a synthetic "vocal" (a sawtooth with vibrato, +30 cents on notes 3 and 6), a louder "reference" take on the loop, and a clipped file.
- *Done when:* `uv run pytest` passes and the spike runs end to end. **Met on Linux; the spike still has to be re-run on the Mac** (plugin loading, Homebrew codecs, FSEvents; see `NOTES.md`).

**Phase 1: Core mastering MVP + database + watch folder — DONE (2026-11-24)**
- Validation, the Matchering pipeline, exports, run manifests, **SQLite indexing and `reindex`**, the reference library, the core loudness analysis, the Master tab with simple players, and the **watch folder** (5.19, analyze-only by default).
- *Done when:* Robin can master a Logic bounce against a saved genre reference in under 3 clicks and get a 24-bit WAV at about −14 LUFS / ≤ −1 dBTP; a file dropped into the Inbox gets analyzed with a notification; deleting the DB and running `reindex` restores it.
- **Met on fixtures:** in the browser, choosing a file and clicking Master (2 clicks, default reference preselected) gave −14.0 LUFS / −1.0 dBTP; Inbox files are analyzed once with the genre from the subfolder; `reindex` after deleting the DB restores identical rows (tested).
- **Not yet checked on the Mac:** a real Logic bounce, the macOS notification itself (the tests stub it), and FSEvents. The UI has no Inbox-specific list beyond the "recent run or Inbox file" picker.

**Phase 2: Compare, analyze, streaming preview, progress v1 — DONE (2026-11-24)**
- The synced A/B/C player with blind mode (saved to the DB), the spectrum and band-delta analysis, the stereo checks, the flags, **streaming previews** (5.11), and **progress tracker v1** (trend charts and blind-test history).
- *Done when:* LUFS matches ffmpeg ebur128 within ±0.1 LU on the fixtures, switching sources keeps the playhead with no level jump, codec previews show post-encode true peak, and the Progress tab shows trends across at least 3 tracks.
- **Met:** loudness within ±0.1 LU (tested); in headless Chromium the playhead moved 9 ms across a switch and matched sources sat at one loudness (`scripts/check_player.js`, 10 checks); codec previews report post-encode true peak and are sample-aligned (tested); the Progress tab charted three tracks.
- **Not done, by choice or limit:** nobody has listened to the player (headless browser only); loop points are set at the playhead with buttons, with no waveform or drag yet; section-jump buttons wait for Phase 6; platform normalization is numbers, not audio.

**Phase 3: Chains and plugins — DONE for free VST3s (2026-11-24); Robin's plugins unchecked**
- Built-in modules, plugin scan and loading, presets, and pre/post-chain integration (the limiter-double-up rule).
- *Done when:* a preset using one of Robin's plugins round-trips with the same output within tolerance.
- **Met with a stand-in:** a preset hosting ZamCompX2 (a free VST3) saved to TOML, reloaded and produced output within 1e-6 of the original (tested; the test skips where that plugin is not installed).
- **Still open:** the same round trip with one of Robin's own plugins, Audio Units, and Waves inside its WaveShell (`--plugin-name`). Plugins that need their editor open to initialize, or that use licensing dongles, may not load headless.

**Phase 4: Stems — DONE (2026-11-24); timing on the MacBook open**
- Separation plus cache, stem diagnostics, rebalance, and Logic stem import.
- *Done when:* a 3-minute track separates in a reasonable time on his MacBook (log the time), and the diagnostics show sensible ratios.
- **Met here:** real separations ran with `mdx-2` and `demucs-4-fast` (about real time on 4 Linux CPU cores; the time is stored in `stems.json` and printed). On the fixture, Demucs put drums at -7.6 dB and bass at -3.7 dB against true values of -7.6 and -3.6. Stems from Logic give exact ratios (tested).
- **Still open:** the timing on the M2, `roformer-2` and `demucs-4` (not run here), and ratios on a real song with a real voice. The synthetic vocal is not recognised as a voice by either model, so vocal-to-beat on separated stems is unproven.

**Phase 5: Batch, AI notes, CLI, Quick Action — DONE (2026-11-24); live API call and Quick Action untried**
- *Done when:* a 4-track EP runs unattended with a consistency report, and mix notes return structured JSON with costs logged.
- **Met:** a 4-track fixture EP masters unattended, survives a broken file, and reports spread and outliers (tested).
- **Met against a stand-in only:** the notes request is built by the real Anthropic SDK and answered by a local fake server in the tests: structured JSON parsed, cost logged, cache and cap enforced. **No request has reached the real API**, so whether the API accepts this exact parameter combination, and how good the notes are, is unknown.
- **Untried:** the Automator Quick Action (macOS only).

**Phase 6: Deep analysis**
- Key and tempo (5.4 musical), **sections** (5.15), **detectors** (5.16), **hip-hop low-end tools** (5.17), **reference quality and suggestions** (5.13), **genre target curves plus curve match mode** (5.14), and **speaker simulation** (5.12).
- *Done when:*
  - key detection is right on the fixtures
  - the detuned 808 note is found within ±10 cents
  - the clipped or lossy-rip fixture references get flagged
  - curve match lands within ±1 dB of the target median in every band on pink noise
  - the section-loudness flags work on a fixture with a known quiet verse and loud hook

**Phase 7: Integrations and release**
- **MCP server** (5.22), **alternate versions with locked processing** (5.20), **listening-level calibration and break reminders** (5.21), and **progress tracker v2** (habit detection plus AI trend summary, 5.18).
- *Done when:* Claude Code can call `compare_runs` and `get_habits` through MCP; instrumental + a cappella before the limiter sums to within a set error of the main master on the fixture stems; and habit detection finds a deliberate bias seeded across fixture tracks.

**Phase 8: Vocal tuning**
- Pitch tracking, the pitch report, correction (retune speed, humanize, amount, formant preservation), the engine benchmark, the pitch editor, and export.
- Includes the Natural/Tight/Hard-tune presets, the simple and Advanced views, and the full-mix separate → tune → recombine path.
- *Done when:*
  - the synthetic vocal's +30-cent notes are corrected to within ±5 cents under Natural with vibrato kept, and snap with stepped transitions under Hard-tune
  - output length and alignment are sample-exact
  - a blind A/B of both engines, plus Robin's Waves result and Flex Pitch, on one real vocal is logged so he can choose the default engine

**Phase 9: Vocal mix assistant**
- The vocal chain and presets, the de-esser, the vocal rider, the dynamic EQ carve, ad-lib handling, sum → master, separate exports, and the feedback loop (5.24).
- *Done when:*
  - on fixture stems, the rider holds the vocal-to-instrumental ratio within ±1 dB of target in each section
  - the de-esser reduces planted sibilance spikes by a measurable amount
  - an end-to-end dry vocal + instrumental → mastered file runs in one click and produces a full analysis with AI notes

## 8. Testing and quality
- Use **only generated fixtures** in the repo; never commit Robin's music. `.gitignore` blocks audio extensions everywhere, and fixtures are generated per test session.
- Unit tests:
  - loudness math (against ffmpeg)
  - band deltas, percentile curves, true-peak oversampling
  - key, tempo and pitch detection on fixtures with known answers
  - detectors on fixtures with planted problems
  - preset round-trips
  - DB migrations and `reindex`
  - MCP tool schemas
- Golden tests for the pipeline: same input, settings and versions give the same output within a small tolerance.
- Keep the UI thin and push logic into the engine, so most features can be tested without Gradio.
- `tests/test_spike.py` (marked `slow`) runs the spike's offline checks, so a dependency bump that breaks an API fails the suite.

## 9. Risks and open questions
- ~~**Python version conflicts**~~ *[P0]* Resolved: one uv environment on Python 3.12 holds everything.
- **Plugin editor windows** under Gradio (5.6).
- **Pitch tracking on separated vocals** is unreliable; steer toward Logic vocal stems.
- **Key detection** is ambiguous for some hip-hop beats (modal loops, relative major/minor). Always show the confidence score and allow manual override, which saves per track.
- **Spotify normalization details** change; verify against Spotify's current documentation.
- **Codec availability** in the Homebrew ffmpeg build (libvorbis/libopus/AAC). Ubuntu's build has all three; the Mac is unchecked.
- **Separated vocals** *[P4]*: every vocal figure from a model is an estimate until checked on real songs.
- **Final limiter** *[P0]*: Pedalboard's `Limiter` overshoots the ceiling, so the true-peak guarantee is ours to build (5.2).
- **Robin's answers (2026-11-24):**
  1. MacBook Pro, M2, 16 or 32 GB RAM (to be confirmed).
  2. Plugins: many purchased third-party plugins used in Logic. These are the ones to host
     (their AU or VST3 builds). Logic's stock plugins cannot be loaded outside Logic. Which
     of them load headless is found by running `masterbench plugins` on the Mac.
  3. Loudness: left to our recommendation, with Spotify as the goal. **Decided: −14 LUFS and
     −1.0 dBTP for every genre.** Spotify turns louder masters down to −14, so extra loudness
     only costs dynamics there. `--lufs` stays available for comparing a louder version in
     the blind player.
  4. Data location: iCloud preferred. **Decided:** `masterbench init --icloud` puts the data
     root in iCloud Drive and keeps the SQLite index in `~/Library/Application Support/
     MasterBench` (a database inside a synced folder gets corrupted). Logic can bounce
     straight into `<data root>/Inbox/<genre>/`. Absolute paths in manifests assume one Mac.
  5. ~~Natural vs hard-tune?~~ **Decided: both, with Natural as the default** (5.23).
  6. Tuner: **Waves Tune LT**. It is a graphical tuner that needs its editor, so it cannot be
     hosted headless; the Phase 8 comparison uses a Waves-tuned vocal exported from Logic.
     Still unanswered: a de-esser or vocal compressor plugin for the vocal chain (ask at
     Phase 9).

## 10. Working agreement
1. Proceed phase by phase, committing at each phase's acceptance criteria and summarizing what changed.
2. Update `NOTES.md` and this plan whenever a library behaves differently from what is written here.
3. Anything that can only be checked on the Mac goes into the "Still to verify on the Mac" list in `NOTES.md` instead of being assumed.
