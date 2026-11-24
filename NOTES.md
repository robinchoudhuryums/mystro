# Build notes

Working versions, API differences from the plan, and things still to verify on the Mac.
Updated at the end of each phase.

## Phase 0: verification spike (2026-11-24)

Run with `uv run python scripts/verify_stack.py`. It generates the fixtures, exercises each
library on them, and prints PASS / SKIP / FAIL per check.

**Where it ran:** a Linux x86_64 container, not the MacBook. 13 checks passed and 1 skipped
(plugin loading: no VST3/AU plugins exist there). Everything in
[Still to verify on the Mac](#still-to-verify-on-the-mac) is untested until the spike is
re-run there.

### Versions that work together

One uv environment on Python 3.12 holds everything; no subprocess environments were needed.

| Package | Version | | Package | Version |
|---|---|---|---|---|
| python | 3.12 | | audio-separator | 0.48.0 |
| matchering | 2.0.6 | | torch | 2.15.0 |
| pedalboard | 0.9.26 | | onnxruntime | 1.32.0 |
| pyloudnorm | 0.2.0 | | mcp | 2.3.0 |
| librosa | 1.0.1 | | gradio | 6.32.0 |
| numpy | 2.5.5 | | typer | 0.29.2 |
| scipy | 1.18.1 | | watchdog | 7.1.1 |
| numba | 0.68.1 | | anthropic | 2.13.0 |
| soundfile | 0.14.0 | | plotly | 7.1.0 |
| pyworld | 0.3.5 | | ffmpeg (Ubuntu) | 6.1.1 |
| pyrubberband | 0.4.0 | | rubberband CLI | 3.3.0 |

Exact pins are in `uv.lock`.

### Findings that change the plan

| Area | Finding | Consequence |
|---|---|---|
| MCP SDK | 2.x renamed `FastMCP` to `MCPServer` (`from mcp.server.mcpserver import MCPServer`). `mcp.server.fastmcp` no longer exists. | Phase 7 uses `MCPServer`. `server.run("stdio")` and the `@server.tool()` decorator are unchanged. |
| MCP SDK | A tool returning bare `dict` sends text only. A typed return (`dict[str, float]`, a TypedDict or a pydantic model) is needed for `structured_content` and an output schema. | Give every MCP tool a typed return. |
| MCP SDK | `mcp.Client` accepts an `MCPServer` instance directly and connects in-process. | Tool-schema tests need no subprocess. |
| Matchering | The 16-bit result is a plain `sf.write(..., "PCM_16")`: no dither. | We write the 16-bit file ourselves with TPDF dither from the 24-bit/float result (5.2 step 4). |
| Matchering | `Result(use_limiter=False, normalize=False)` peaked at +6.2 dBFS on the fixture. | That path must be written as float and always followed by our limiter. Never write it as PCM. |
| Pedalboard | `Limiter(threshold_db=-1.0)` still produced a 0.00 dBFS sample peak and +1.72 dBTP after a +12 dB push. It is not a ceiling and not true-peak aware. | The final limiter needs our own true-peak stage: measure 4x-oversampled peak after the chain and trim, or build a lookahead limiter. Decide in Phase 1. |
| pYIN on 808s | On the bass fixture, raw `librosa.pyin` slipped an octave low on 1 of 8 notes and read about +5 cents on the rest. | 5.17 uses pYIN for note finding, then a zero-padded FFT peak per note (checking the octave above). That gave under 0.1 cents error. The spike has the reference implementation (`refine_pitch`). |
| Tempo | `librosa.beat.beat_track` returned 69.8 BPM for the 140 BPM loop. | Tempo needs a half/double-time rule (genre prior or onset density) in Phase 6. |
| pyworld | 0.3.5 imports `pkg_resources`, which setuptools 81+ removed. | `setuptools<81` is pinned as a dependency. |
| pyworld | Synthesis output was 220 samples longer than the input (padding to a frame boundary). Rubber Band returned the exact length. | Phase 8 trims or pads WORLD output to meet the sample-exact criterion. |
| audio-separator | 0.48.0 imports `audioread` without declaring it (librosa 1.0 dropped it). | `audioread` is an explicit dependency. |
| Loudness | pyloudnorm matched `ffmpeg ebur128` within 0.06 LU on all five fixtures; the 4x `resample_poly` true peak matched within 0.10 dB. | The Phase 2 criterion (±0.1 LU) is reachable with the planned approach. |
| SQLite | `references` is an SQL keyword. | The table is named `reference_tracks`. |

### Layout decisions

- The repo root is the project root (`pyproject.toml` at the top), not a nested `masterbench/`.
- Migrations live in `src/masterbench/migrations/` and load through `importlib.resources`,
  so they ship inside the package instead of depending on the repo checkout.
- Run IDs are text primary keys (the run folder name), since the manifests are the source
  of truth and the DB is rebuilt from them.
- Fixtures come from `masterbench.fixtures` and are generated per test session into a temp
  folder. No audio is committed; `.gitignore` blocks audio extensions repo-wide.
- Stem separation is a normal dependency, not an extra, so `uv run` never uninstalls it.
- `MASTERBENCH_HOME` overrides the `~/MasterBench` data root (tests use it).

### Other observations

- Separation: `UVR-MDX-NET-Inst_HQ_3.onnx` split the 14 s loop in 13.7 s on 4 CPU cores
  (about real time). Models download from GitHub releases on first use.
- Codecs: the true-peak change after an encode/decode round trip on the fixture reference was
  +0.05 dB (Vorbis q9), +0.01 dB (AAC 256k) and -0.33 dB (Opus 160k). Real masters will move
  more; this only proves the measurement path.
- Rubber Band: `pyrubberband.pitch_shift(..., rbargs={"--fine": "", "--formant": ""})` selects
  the R3 engine with formant preservation and hit the requested shift exactly.
- Matchering imports `scipy.ndimage.filters`, which SciPy 2.0 will remove. It works on
  1.18 with a deprecation warning; hold SciPy below 2 when that release arrives.

## Phase 1: mastering MVP (2026-11-24)

### What was built

`masterbench master|analyze|reindex|watch|ui|ref`, the engine behind them, and a Gradio app
with Master and Library tabs. 110 tests pass.

### Decisions

- **Final limiter:** our own (`master/limiter.py`), because Pedalboard's does not hold a
  ceiling. Per-sample required gain comes from the 4x-oversampled peak across both channels
  (linked, so the stereo image does not move), is held over a 5 ms lookahead, released at
  40 dB/s and smoothed. A last static trim absorbs the overshoot the gain changes themselves
  cause, which makes the ceiling a guarantee. The release is deliberately slow for a peak
  limiter: 808 material distorts when gain recovers inside a bass cycle.
- **Loudness targeting:** `finalize()` applies make-up gain, limits, measures, and corrects
  until within 0.05 LU. If the target would need more than 6 dB of limiting it backs off and
  the master comes out quieter than asked, with `off_target` and `heavy_limiting` flags.
- **Matchering always runs with its limiter and normalization off.** One path, one limiter.
  Its internal sample rate is set to the mix's rate, so a 48 kHz bounce stays 48 kHz.
- **Short-term, momentary and LRA** are computed here (K-weighting from pyloudnorm's filter
  class, sliding windows at a 100 ms hop, EBU Tech 3342 gating for LRA). Integrated loudness
  still comes from pyloudnorm.
- **Takes of a song share a track:** `Song v3.wav`, `Song_mix2.wav` and `Song bounce 4.wav`
  all land in project `song`. Only suffixes introduced by v/ver/mix/bounce/master are
  stripped, so `Track 7.wav` stays `track-7`. `--title` overrides it.
- **Watch folder dedupes by content hash**, so restarting the watcher or re-copying a file
  does not create another run.
- **16-bit export** is TPDF-dithered here, seeded from the audio so identical input gives an
  identical file.

### Measurements

| Check | Result |
|---|---|
| Loudness vs `ffmpeg ebur128`, six fixtures | integrated ±0.1 LU, short-term and momentary max ±0.15, LRA ±0.2, true peak ±0.15 dB (ffmpeg prints one decimal) |
| Fixture loop mastered to the default target | -14.0 LUFS, -1.0 dBTP, 0.7 dB of limiting, one limiter pass |
| Ceiling under heavy limiting (+18 dB push) | held at 4x oversampling; an 8x measurement read up to 0.09 dB higher |
| 206 s, 48 kHz track end to end (4 CPU cores, Linux) | 20 s: validate 5, Matchering 4, limiter 7, export and analysis 4 |
| Same input and settings twice | byte-identical 24-bit and 16-bit files |

### Known gaps

- The limiter's true peak is a 4x estimate. Against the default -1.0 dBTP ceiling the
  remaining error is under 0.1 dB; raise `OVERSAMPLE` if a stricter figure is ever needed.
- `mg.log` is process-global, so two masters in parallel threads would share Matchering's
  log handlers. Jobs run one at a time today.
- The UI stores an uploaded file's Gradio temp path as `original_path` in the manifest.
  The copy under `projects/<slug>/input/` is what matters.

## Phase 2: compare, analyze, streaming previews, progress (2026-11-24)

### What was built

Spectrum, band and stereo analysis with flags; codec previews and platform gains
(`masterbench preview RUN_ID`); the A/B/C player; Compare, Analyze and Progress tabs.
140 tests pass.

### Decisions

- **One spectral pass per file:** Welch auto- and cross-spectra (32768-sample segments) feed
  the smoothed spectrum, the band levels and every stereo figure. Band correlation is the
  real part of the cross-spectrum over the band, so no per-band filtering is needed.
- **Equal-loudness comparison by construction:** band levels and spectra are stored in dB
  relative to the file's integrated loudness. A delta between two files is a subtraction.
- **Stereo flags have a dead zone:** correlation below -0.2 is a phase problem; between
  there and +0.3 it is "wide". Uncorrelated noise sits near 0 and must not read as broken.
- **Player:** every source is decoded up front and all play together from one start time,
  each through a gain node. The selected one is audible; switching is a 15 ms crossfade.
  That makes sync exact and removes the need for seeking on a switch. Matching plays every
  source at the level of the quietest one, so nothing clips. Cost: decoded audio for all
  sources sits in memory (about 90 MB per 4-minute stereo source at 48 kHz).
- **Blind mode** shuffles the order, hides names and levels, forces level matching, and
  records the ranking and the mapping.
- **Analysis-only runs compare against the genre's default reference** when there is one,
  so watch-folder results already say where the balance is off.

### Measurements

| Check | Result |
|---|---|
| Pink noise band levels | equal energy per octave within 0.6 dB across the seven bands |
| Matchering on the fixture loop | worst band delta vs the reference drops from 2.5 dB (mix) to 1.2 dB (master) |
| Codec round trip of the -14 LUFS fixture master (-1.05 dBTP) | Vorbis -0.98, AAC -0.88, Opus 128k -0.64 dBTP; all within 2 samples of the master's timing |
| Player in headless Chromium | playhead +9 ms across a switch, one audible source, matched loudness spread 0.000 LU |
| Full analysis of a 206 s, 48 kHz file | 6.6 s |

The Opus result is worth knowing: even a master held to -1 dBTP came back at -0.64 dBTP on
this fixture, and the preview then suggests a -1.5 dBTP ceiling.

### Known gaps

- Spotify's rules were first taken from search-result quotes; they were checked against
  the live article in Phase 4 and match.
- The player has only been run headless. Crossfade clicks, the feel of the loop buttons and
  memory use with long tracks need ears and a real browser.
- Loop points are set at the playhead with two buttons; there is no waveform or drag.
- Platform normalization is reported, not auditioned. Spotify's Loud-mode limiter is
  described in the message but not simulated.
- Runs made in Phase 1 have no stored spectrum; the Analyze tab says so and a new
  `analyze` or `master` run fills it in.

## Phase 3: chains and plugins (2026-11-24)

### What was built

`master/chain.py` and `master/presets.py`, three packaged presets (`streaming-safe`,
`gentle-glue`, `warm`), `--pre` / `--post` (alias `--preset`) on `master`,
`masterbench preset list|show|add-plugin`, `masterbench plugins`, chain pickers on the
Master tab and a preset editor with plugin scan in the Library tab. 172 tests pass.

### Findings

Plugin hosting was tested for real with free VST3s (`zam-plugins`, `dragonfly-reverb-vst3`
from the Ubuntu archive), which the Phase 0 spike had to skip.

| Finding | Consequence |
|---|---|
| `load_plugin`, `.parameters` (min, max, units), `.raw_state` and processing all work headless. Output was bit-identical between two fresh instances of each stereo plugin tried. | A preset stores `path`, `parameters` and base64 `state`; reloading reproduces the audio within 1e-6. |
| Mono-only plugins (ZamComp, ZamEQ2, ZamTube) raise `ValueError` on stereo input. | The chain runs them once per channel (dual mono). |
| ZamEQ2 returns NaN at its own default settings. | Every stage's output is checked; NaN or infinity stops the chain with a clear error instead of reaching the limiter. |
| DPF-based plugins expose host values (`buffer_size_frames`, `sample_rate_frames`) as parameters. | They are captured like any other parameter; setting them back was harmless here. |
| Scanning loads each bundle to read its names, and some plugins print to stderr while loading. | Scanning is on demand (button or `masterbench plugins`), never at startup. |
| Pedalboard's compressor with a 1 ms attack raises the crest factor of the 808 loop: the click passes, the body is squashed. | Expected behaviour, noted because the packaged `gentle-glue` uses a 30 ms attack for that reason. |

### Decisions

- The packaged presets are starting points made up for this project, not tuned by ear.
- Preset names are restricted to `[a-z0-9_-]`, so a name can never leave the presets folder.
- The full resolved preset (modules and plugin state) is written into `run.json`, so a run
  stays reproducible after the preset file changes.
- Built-in modules run in float32 inside Pedalboard; the pipeline converts back to float64.

### Known gaps

- Robin's own plugins, Audio Units and Waves WaveShell are untested (no macOS, no licences).
- No slider UI for plugin parameters and no native editor window.
- A hosted plugin's reported latency is left to Pedalboard; only length preservation is
  checked. A plugin with lookahead that Pedalboard does not compensate would shift the audio.

## Phase 4: stems (2026-11-24)

### What was built

`masterbench separate`, `stems import|list`, `rebalance`; the Stems tab; stem solos in the
A/B/C player; iCloud-safe data root (`init --icloud`). 204 tests pass.

### Findings

| Finding | Consequence |
|---|---|
| audio-separator scales any output stem that peaks above its `normalization_threshold` (0.9 by default), each stem on its own. | That silently changes the balance between stems. The wrapper stages the input at a 0.5 peak, sets the threshold to 1.0, and scales the stems back. |
| It always works and writes at 44.1 kHz, whatever the input. | Stems are resampled and trimmed to the source's rate and length. For a 48 kHz bounce the stems then sum to within about -33 dB of the mix, because content above 22 kHz is gone. |
| With `use_soundfile=True` the output keeps the input's subtype. | The staged input is float, so stems come back as float and nothing is clipped. |
| Demucs weights come from `dl.fbaipublicfiles.com`, MDX and Roformer weights from GitHub. | Both hosts must be reachable on first use of a model. |
| On the fixture song, `demucs-4-fast` measured drums at -7.61 dB and bass at -3.69 dB relative to the mix; the true stems give -7.56 and -3.60. | Drum and bass ratios from separation look trustworthy. |
| Neither model treats the synthetic sawtooth "vocal" as a voice (Demucs filed it under "other"). | Vocal-to-beat on separated stems cannot be validated with fixtures. Needs a real song. |
| Separation ran at about real time on 4 CPU cores (14 s of audio in 14 to 16 s). | An M2 with CoreML should be several times faster; unmeasured. |

### Decisions

- **Reference comparison uses separated reference stems**, cached in
  `references/<genre>/.stems/`. When the mix's stems are real (from Logic) and the
  reference's are separated, the comparison mixes exact and estimated numbers; the flag
  threshold is 2 dB to leave room for that.
- **Rebalance never limits.** Gains are capped at ±6 dB per stem; if the new sum would peak
  above -1 dBFS the whole mix is turned down and the amount reported.
- **A rebalanced mix is mastered as a new run of the same track**, so it lines up next to
  the original master in the Compare tab.
- **Role guessing** for Logic stems is by file name (vocal/vox/lead/adlib, drum/kick/snare/hat,
  bass/808/sub, beat/instrumental; everything else is "other") and can be overridden.

### Spotify's article, read directly

`support.spotify.com` is now reachable from the build environment, and the article matches
what Phase 2 implemented. Two additions came out of it: the `loud_for_spotify` flag (masters
louder than -14 LUFS should peak under -2 dBTP), and a note that album playback uses one
gain for the whole album, which the EP mode in Phase 5 should report.

### Known gaps

- Stem diagnostics are stored as `diagnosis.json` beside the stems but are not in the
  SQLite index yet, so the progress tracker cannot trend them. That belongs with habit
  detection in Phase 7.
- `roformer-2` and `demucs-4` were not run (large and slow on CPU).
- Manifests store absolute paths, so a data root shared between two Macs with different
  user names would need those rewritten.

## Phase 5: batch, AI notes, Quick Action (2026-11-24)

### What was built

`masterbench batch`, `masterbench notes`, `master --notify`, a Batch tab, AI mix notes in
the Analyze tab, `quick_action/`. Robin's cycle workflow commands were added under
`.claude/commands/`. 222 tests pass.

### Findings

| Finding | Consequence |
|---|---|
| anthropic SDK 2.13 sends the key as `Authorization: Bearer` under API version `2026-10-21`, not `x-api-key`. | Nothing to do in our code; noted because older examples show the other header. |
| Claude Sonnet 5.5 rejects forced tool choice and disabled thinking. | Notes use structured output (`output_config.format` with a JSON schema) and leave thinking adaptive, steering depth with `effort: medium`. |
| Sonnet 5.5 can decline a request (`stop_reason: "refusal"`). | The request opts into the API's server-side fallback (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`). A refusal that survives it is an error and nothing is saved. Haiku has no server-side fallback, so it is called without one. |
| Matching every track of a set to one reference pulls their tone together. | In a batch, band outliers are rare by construction; the report mostly earns its keep on loudness, peaks, and tracks given different chains or references. |
| A full payload for one run is about 12 kB of JSON. | Roughly 4k input tokens: about 2 to 4 cents per run on Sonnet 5.5, under a tenth of a cent on Haiku 5.5. |

### Decisions

- **What is sent:** track title and genre, targets and chain module names, loudness, band
  levels and deltas, a third-octave spectrum, stereo figures, a short loudness curve, the
  flags, and stem and streaming figures when they exist. Paths, hashes and file names are
  left out.
- **Cost control:** an offline estimate before sending, the real cost from the response's
  token counts, a monthly cap (`notes_monthly_cap_usd`, default $5) checked against the
  notes stored on disk, and reuse of a stored answer when measurements, intent, model and
  prompt version are all unchanged.
- **Prices are a table in the code** (`PRICES`), from the published per-token rates on
  2026-11-24. They need updating by hand if rates change.
- **Batch consistency** compares each master with the set's median: loudness beyond 0.7 LU,
  a band beyond 2.5 dB, or a loudest section beyond 1.4 LU is flagged. "Loudest section" is
  the short-term loudness maximum until real sections arrive in Phase 6.

### Known gaps

- **No live API call has been made.** There is no key in the build environment, and network
  secrets cannot be stored for Anthropic's own hosts. The first real `masterbench notes`
  run is the test.
- The Quick Action script and its Automator steps are untried.
- Album-mode normalization is described in the batch report but not computed, because
  Spotify's article does not say how the album gain is derived.

## Still to verify on the Mac

Re-run `uv run python scripts/verify_stack.py` there and update this file.

1. **Plugin hosting with your plugins:** VST3 hosting is verified on Linux with free
   plugins. Still to check: an Audio Unit, one of your own plugins in a preset round trip
   (`masterbench preset add-plugin NAME /path/to/Plugin.vst3`, then master twice and compare),
   and Waves through its WaveShell (`--plugin-name`).
2. **Homebrew ffmpeg encoders:** `libvorbis`, `libopus`, and whether `aac_at` (AudioToolbox,
   closer to Apple's encoder) is present. The codec check prefers `aac_at` when it exists.
3. **Source builds:** `pyworld` and `diffq` have no macOS arm64 wheels for Python 3.12, so
   `uv sync` compiles them. That needs the Xcode Command Line Tools.
4. **Separation on a real song:** time `masterbench separate song.wav` for each model
   (`mdx-2`, `roformer-2`, `demucs-4-fast`, `demucs-4`), and judge whether the vocal-to-beat
   figure matches what you hear.
5. **watchdog** on the FSEvents backend (inotify was used here).
6. **Notifications** through `osascript` or `terminal-notifier` (only the latter can open
   the run link on click), the Quick Action, and the LaunchAgent.
6a. **AI notes:** run `masterbench notes RUN_ID --payload` to see what would be sent, then
   without `--payload` once a key is set. This is the first live call.
6b. **Quick Action:** follow `quick_action/README.md`.
7. **Phase 2 by ear:** the A/B/C player in Safari or Chrome (switching, loops, blind mode).
8. **Phase 1 on a real Logic bounce:** master it from the UI, listen to the limiter on
   808-heavy material, and drop a bounce into `~/MasterBench/Inbox/<genre>/` while
   `masterbench watch` runs.
