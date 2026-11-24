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

### Still to verify on the Mac

Re-run `uv run python scripts/verify_stack.py` there and update this file.

1. **Plugin hosting:** `load_plugin` on one AU and one VST3, parameter listing, and Waves
   through its WaveShell (`plugin_name`). The check picks the first plugin it can load.
2. **Homebrew ffmpeg encoders:** `libvorbis`, `libopus`, and whether `aac_at` (AudioToolbox,
   closer to Apple's encoder) is present. The codec check prefers `aac_at` when it exists.
3. **Source builds:** `pyworld` and `diffq` have no macOS arm64 wheels for Python 3.12, so
   `uv sync` compiles them. That needs the Xcode Command Line Tools.
4. **Separation speed** with the CoreML provider, and a Roformer model against `htdemucs_ft`.
   `htdemucs` weights come from a different host than the MDX models.
5. **watchdog** on the FSEvents backend (inotify was used here).
6. **Notifications** through `osascript`, the Quick Action, and the LaunchAgent.
