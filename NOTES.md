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

## Phase 6: deep analysis (2026-11-24)

### What was built

Key and tempo, sections, problem detectors, 808 tools, reference quality and suggestions,
genre target curves with curve-match mastering, speaker simulation.
New commands: `master --curve`, `curve build|show`, `ref suggest|refresh`, `sections`, `key`,
`lowend`, `speakers`. The Analyze tab shows key, tempo, sections and the low-end table and
lets them be corrected; the player has jump buttons for sections and flagged moments.
291 tests pass.

### Findings

| Finding | Consequence |
|---|---|
| Comparing a few seconds either side of a point splits a looped beat at every bar where the 808 note changes. | Section boundaries are judged over eight bars either side (from the detected tempo; 12 s without one). A 15-times-repeated loop then reads as one section. |
| Deciding hook versus verse by level alone fails exactly when it matters: a hook 1 LU over its verse looks like the verse. | Sections are grouped by timbre first; the loudest group is the hook. The weak hook is then recognised and flagged. |
| A level-step search alone finds boundaries to within 0.5 s; timbre clustering alone is 1 to 2 s off. | Level steps place the boundary; timbre boundaries are kept only where no step is within 2.5 s. |
| On a full mix the sibilance detector fires on every hi-hat. | Sibilance runs on vocal stems only (in the stem diagnosis). |
| A held note and a resonance are the same thing in a long-term spectrum. | Narrow peaks within 20 cents of a musical pitch are skipped; others must be present 60% of the time their region sounds. A resonance that lands on a note is missed. |
| A kick landing on a held 808 breaks the pitch track in two. | Consecutive notes of the same pitch within 0.3 s are rejoined. |
| Low-passed noise looks pitched to pYIN. | A note needs its spectral peak 32 dB above the low band's median, and a fundamental under about 160 Hz. |
| MP3 at 128 kbps cuts off at 17 kHz on the fixture; at 320 kbps the cutoff is 20.5 kHz. | A cutoff under 19.5 kHz is called a lossy rip, whatever the file extension. High-bitrate lossy files pass on spectrum and are only cautioned for their format. |
| EQ changes loudness, and levels are relative to loudness. | The matching EQ is refined over up to four passes; on pink noise it ends within 0.05 dB of the curve. |
| The loop fixture's planted detuned note is now found everywhere. | Tests that expected "no flags" on the loop now expect exactly that one. |

### Measurements

| Check | Result |
|---|---|
| Detuned 808 note (planted +35 cents) | read as +35.0 to +35.1 cents, from the full mix |
| Section boundaries on the 69 s fixture song | within 0.75 s of the truth, kinds all correct, at both strong and weak contrast |
| Curve match on pink noise, three-reference curve | within 0.05 dB in every band (criterion: 1 dB); bit-identical on repeat |
| Full analysis of a 206 s, 48 kHz mix (4 CPU cores) | 16 s, up from 7 s before this phase |
| Whole test suite | 291 tests, about 10 minutes |

### Known gaps

- **Nothing here has been run on real music.** Thresholds for section grouping, the
  detectors and reference quality came from synthetic fixtures.
- Section kinds are guesses. A song whose hook is quieter than its verse will be labelled
  backwards; hand-set sections fix that per track.
- 808 glides and slides are not modelled; a sliding note is dropped or read at one pitch.
- The kick/808 clash needs drum and bass stems (Logic stems or a four-stem separation).
- Speaker profiles are plausible EQ shapes, not measurements of real devices.
- Section markers are typed, not dragged on a waveform.
- With two or three references a curve's range is narrow, so small differences fall
  outside it. It becomes meaningful with five or more.
- essentia was not evaluated as an alternative key detector.
- Stem diagnostics and low-end results are still not in the SQLite index.

## Phase 7: MCP server, alternate versions, listening sessions, habits (2026-11-24)

### What was built

`release/versions.py` (instrumental and a cappella), habit detection in `progress.py`,
`notes/trends.py` (AI trend summary), `monitoring.py` (calibration noise, sessions, break
reminders, fresh-ears queue) and `mcp_server.py` (ten tools).
New commands: `versions`, `habits`, `trends`, `calibrate`, `mcp`. New tabs: Release and
Settings; Progress gained habits, a song-shape chart and the trend summary; Compare gained
the fresh-ears queue. Migration 004 adds hook contrast and in-tune share to the index.
320 tests, about 13 minutes.

### Findings

| Finding | Consequence |
|---|---|
| The plan's static transfer (master spectrum over mix spectrum, smoothed) reproduces a Matchering master only to -18.7 dB. Matchering's mid and side EQs differ, and smoothing throws away what it did. | The locked process is fitted by least squares on mid and side separately (cross-spectrum over mix spectrum, 8192-tap FIR). That reaches about -35 dB. |
| A curve-match master is EQ plus gain, both stored in `run.json`. | It is replayed exactly on each part; the remaining -38 dB before limiting is the limiter's own work, and with the same limiter on the sum the difference is under -70 dB. |
| Saturation and compression depend on level, so no process applied to the parts separately can add up to the master. | Chains made only of filters and gain stay exact; anything else is fitted and flagged `fitted_processing`. With the `warm` post-chain the fit still reaches -30 dB. |
| Three out of three is weak evidence. | Confidence is the Wilson lower bound: 0.44 for 3 of 3, 0.57 for 5 of 5, 0.68 for 8 of 8. |
| Band levels are relative to integrated loudness, and on the loop the sub carries much of the loudness. An 8 dB sub cut moved the sub delta from -0.7 to -3.9 dB and every band from low-mids to presence from about +2 to about +5 dB. | **A habit list does not say which band is the cause.** The seeded cut is reported as sub under, and also as four bands over plus `mud` and `harsh` on 5 of 5, all at the same confidence. A control run of the same loops has no sub habit. See the known gaps. |
| MCP SDK 2.x turns an unexpected exception into "Error executing tool" plus a traceback. | Engine errors the user can act on are re-raised as `ToolError` with the engine's own message. |
| The fixture loop cannot be pushed to -10 LUFS within the limiter's 6 dB budget; it lands near -10.5. | Nothing changed; the run carries `off_target` and `heavy_limiting`. Noted because a test assumed otherwise. |

### Measurements

| Check | Result |
|---|---|
| Instrumental + a cappella vs master, before limiting, Matchering run | -34.7 dB (set error: -25 dB) |
| The same, curve-match run | -38.1 dB; with a linear pre-chain -46.6 dB; with a saturating post-chain (fitted) -30.3 dB |
| Calibration noise | -20.00 dBFS RMS, over 99% of its power between 500 Hz and 2 kHz |
| `compare_runs` reply | under 3 kB of JSON; `get_analysis` under 12 kB |

### Known gaps

- **No Claude model has called the MCP tools in a conversation.** The SDK client has, in
  process and over stdio, and `claude mcp list` reports the server connected.
- The trend summary, like the mix notes, has only met a stand-in for the API.
- Versions are files beside the project, not runs: not indexed, not in the player, 24-bit
  only, and remade from scratch each time.
- Stems exported from Logic through a processed mix bus will not add up to the bounce. The
  tool reports by how much and carries on; the versions are then only as close as that.
- **One cause can show up as several habits.** A cut in the band that carries the loudness
  reads as that band under and the others over (see the findings). The list is ordered by
  confidence, then count, then name, so the cause is not necessarily first. Worth a rethink
  once there are real tracks: for example judging bands after removing the overall tilt.
- The fixture loops already sit about 2 dB over the fixture reference from low-mids to
  presence, so those four habits appear in the control too. That is the fixtures, not a
  detection fault, but it means the seeded test shows one new habit among existing ones.
- Habits are unconditional. "Over on vocal-heavy tracks" is out of reach until stem
  results are indexed. The yardstick can differ between tracks (curve for some, a single
  reference for others); the statement then says "the comparison".
- Sessions measure how long the UI was open, not how long anyone listened. Closing the
  tab ends the count at the last minute tick.
- The Settings tab shows settings; it does not edit them. `settings.toml` is edited by hand.
- `launchd/` (LaunchAgent for the watch folder) is still not written.

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
6c. **Phase 6 on real tracks:** analyze a few finished songs and check the key, the
   sections (`masterbench sections RUN_ID`), the 808 table (`masterbench lowend RUN_ID`),
   and whether the flags match what you hear. Add five or more references to a genre and
   run `masterbench curve build GENRE`.
6a. **AI notes:** run `masterbench notes RUN_ID --payload` to see what would be sent, then
   without `--payload` once a key is set. This is the first live call.
6b. **Quick Action:** follow `quick_action/README.md`.
6d. **MCP from Claude Code:** `claude mcp add masterbench -- uv run --project "$PWD" masterbench mcp`,
   then ask it to compare two runs and to list your habits. This is the first time a model
   uses the tools.
6e. **Alternate versions from real stems:** `masterbench versions RUN_ID` after
   `masterbench stems import`. Check the reported "stems versus bounce" figure first; if it
   is poor, export the stems with the mix-bus processing off or printed.
6f. **Calibration and breaks:** `masterbench calibrate` with a phone SPL meter, and whether
   the break notification appears after 50 minutes with the UI open.
7. **Phase 2 by ear:** the A/B/C player in Safari or Chrome (switching, loops, blind mode).
8. **Phase 1 on a real Logic bounce:** master it from the UI, listen to the limiter on
   808-heavy material, and drop a bounce into `~/MasterBench/Inbox/<genre>/` while
   `masterbench watch` runs.
