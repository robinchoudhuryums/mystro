"""Phase 0 verification spike: exercise every third-party piece on generated audio.

    uv run --extra stems python scripts/verify_stack.py [--only a,b] [--skip a,b] [--json out.json]

Each check either passes with a one-line finding, skips with the reason (missing optional
dependency, no plugins on this machine, model host unreachable), or fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from masterbench import db, fixtures

DISTRIBUTIONS = (
    "matchering",
    "pedalboard",
    "pyloudnorm",
    "librosa",
    "numpy",
    "scipy",
    "numba",
    "soundfile",
    "pyworld",
    "pyrubberband",
    "audio-separator",
    "torch",
    "onnxruntime",
    "mcp",
    "gradio",
    "typer",
    "watchdog",
    "anthropic",
    "plotly",
)
SEPARATION_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"

MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


class Skip(Exception):
    pass


@dataclass
class Context:
    work: Path
    fx: dict[str, Path]
    truth: dict


@dataclass
class Outcome:
    name: str
    status: str
    detail: str
    seconds: float


def true_peak_dbtp(audio: np.ndarray, oversample: int = 4) -> float:
    up = signal.resample_poly(audio, oversample, 1, axis=0)
    return float(20 * np.log10(np.max(np.abs(up))))


def lufs(path: Path) -> float:
    import pyloudnorm as pyln

    audio, sr = sf.read(path)
    return float(pyln.Meter(sr).integrated_loudness(audio))


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def cents(f: float, ref: float) -> float:
    return float(1200 * np.log2(f / ref))


# --- checks -------------------------------------------------------------------------------


def check_sqlite(ctx: Context) -> str:
    conn = db.connect(ctx.work / "spike.db")
    applied = db.migrate(conn)
    again = db.migrate(conn)
    tables = conn.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
    conn.close()
    assert applied and not again
    return f"{len(applied)} migration(s) applied, idempotent, {tables} tables"


def check_matchering(ctx: Context) -> str:
    import matchering as mg

    limited = ctx.work / "mg_limited_24.wav"
    unlimited = ctx.work / "mg_unlimited_float.wav"
    pcm16 = ctx.work / "mg_16.wav"
    mg.process(
        target=str(ctx.fx["loop"]),
        reference=str(ctx.fx["reference"]),
        results=[
            mg.Result(str(limited), subtype="PCM_24", use_limiter=True, normalize=True),
            mg.Result(str(unlimited), subtype="FLOAT", use_limiter=False, normalize=False),
            mg.pcm16(str(pcm16)),
        ],
    )
    target_frames = sf.info(ctx.fx["loop"]).frames
    assert sf.info(limited).frames == target_frames, "output length differs from target"
    assert sf.info(pcm16).subtype == "PCM_16"
    raw, _ = sf.read(unlimited)
    out, ref = lufs(limited), lufs(ctx.fx["reference"])
    assert abs(out - ref) < 2.0, f"matched loudness too far from reference: {out:.1f} vs {ref:.1f}"
    return (
        f"limited {out:.1f} LUFS vs reference {ref:.1f}; "
        f"use_limiter=False/normalize=False peaks at {20 * np.log10(np.abs(raw).max()):+.1f} dBFS "
        "(needs our own limiter); 16-bit output is not dithered"
    )


def check_pedalboard(ctx: Context) -> str:
    import pedalboard as pb
    from pedalboard.io import AudioFile

    board = pb.Pedalboard(
        [
            pb.HighpassFilter(cutoff_frequency_hz=30),
            pb.LowShelfFilter(cutoff_frequency_hz=100, gain_db=1.5),
            pb.PeakFilter(cutoff_frequency_hz=3000, gain_db=-2.0, q=1.0),
            pb.Compressor(threshold_db=-18, ratio=2.0, attack_ms=10, release_ms=120),
            pb.Gain(gain_db=12.0),
            pb.Limiter(threshold_db=-1.0, release_ms=100),
        ]
    )
    out = ctx.work / "pb_chain.wav"
    with AudioFile(str(ctx.fx["loop"])) as src:
        audio, sr = src.read(src.frames), src.samplerate
    processed = board(audio, sr)
    with AudioFile(str(out), "w", sr, processed.shape[0], bit_depth=24) as dst:
        dst.write(processed)
    assert processed.shape == audio.shape
    sample_peak = 20 * np.log10(np.abs(processed).max())
    tp = true_peak_dbtp(processed.T)
    return (
        f"chain ran, shape kept; Limiter(threshold_db=-1.0) gives sample peak "
        f"{sample_peak:+.2f} dBFS, true peak {tp:+.2f} dBTP (not a true-peak limiter)"
    )


def plugin_dirs() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        bases = [Path("/Library/Audio/Plug-Ins"), home / "Library/Audio/Plug-Ins"]
        return [b / kind for b in bases for kind in ("VST3", "Components")]
    return [home / ".vst3", Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3")]


def check_plugins(ctx: Context) -> str:
    import pedalboard as pb

    found = sorted(
        p
        for d in plugin_dirs()
        if d.is_dir()
        for p in d.iterdir()
        if p.suffix in {".vst3", ".component"}
    )
    if not found:
        raise Skip(f"no VST3/AU plugins in {', '.join(str(d) for d in plugin_dirs())}")
    errors = []
    for path in found:
        try:
            names = (
                pb.VST3Plugin.get_plugin_names_for_file(str(path)) if path.suffix == ".vst3" else []
            )
            plugin = (
                pb.load_plugin(str(path), plugin_name=names[0])
                if len(names) > 1
                else pb.load_plugin(str(path))
            )
        except Exception as exc:  # plugin hosts fail in many ways; try the next one
            errors.append(f"{path.name}: {exc}")
            continue
        audio, sr = sf.read(ctx.fx["loop"], dtype="float32")
        out = plugin(audio.T, sr)
        assert out.shape[-1] == len(audio)
        return f"{len(found)} found; loaded {path.name} with {len(plugin.parameters)} parameters"
    raise AssertionError(f"{len(found)} plugins found, none loaded: {errors[:3]}")


def ffmpeg_ebur128(path: Path) -> tuple[float, float]:
    cmd = ["ffmpeg", "-nostats", "-hide_banner", "-i", str(path), "-af", "ebur128=peak=true"]
    summary = run([*cmd, "-f", "null", "-"]).stderr.split("Summary:")[-1]
    integrated = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    peak = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary)
    assert integrated and peak, summary
    return float(integrated.group(1)), float(peak.group(1))


def check_loudness(ctx: Context) -> str:
    worst_lu, worst_tp = 0.0, 0.0
    for name in ("pink_noise", "loop", "reference", "clipped", "sine_sweep"):
        audio, _ = sf.read(ctx.fx[name])
        ff_lufs, ff_tp = ffmpeg_ebur128(ctx.fx[name])
        worst_lu = max(worst_lu, abs(lufs(ctx.fx[name]) - ff_lufs))
        worst_tp = max(worst_tp, abs(true_peak_dbtp(audio) - ff_tp))
    assert worst_lu <= 0.1, f"pyloudnorm vs ffmpeg ebur128 differ by {worst_lu:.2f} LU"
    assert worst_tp <= 0.3, f"4x true peak vs ffmpeg differ by {worst_tp:.2f} dB"
    return f"max |pyloudnorm - ffmpeg| {worst_lu:.2f} LU; max true-peak diff {worst_tp:.2f} dB"


def estimate_key(chroma: np.ndarray) -> tuple[str, float]:
    scores = []
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for tonic in range(12):
            score = np.corrcoef(chroma, np.roll(profile, tonic))[0, 1]
            scores.append((score, f"{fixtures.NOTE_NAMES[tonic]} {mode}"))
    scores.sort(reverse=True)
    return scores[0][1], float(scores[0][0] - scores[1][0])


def check_librosa(ctx: Context) -> str:
    import librosa

    y, sr = librosa.load(ctx.fx["loop"], sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    truth_tempo = ctx.truth["loop"]["tempo_bpm"]
    ratio = tempo / truth_tempo
    assert min(abs(ratio - r) for r in (0.5, 1.0, 2.0)) < 0.03, f"tempo {tempo:.1f}"

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    key, margin = estimate_key(chroma)
    assert key == ctx.truth["loop"]["key"], f"key {key}"

    # pYIN alone is not enough down here: it lands an octave low on some notes and reads
    # about 5 cents sharp on decaying 808s. A spectral-peak pass on each note fixes both.
    lowpass = signal.butter(4, 150, "lowpass", fs=sr, output="sos")
    decimate = 8
    bass = signal.resample_poly(signal.sosfiltfilt(lowpass, y), 1, decimate)
    bass_sr = sr / decimate
    f0, voiced, _ = librosa.pyin(bass, fmin=25, fmax=150, sr=bass_sr, frame_length=1024)
    times = librosa.times_like(f0, sr=bass_sr, hop_length=256)
    raw_errors, errors = [], []
    for note in ctx.truth["loop"]["notes_808"]:
        start, end = note["start_s"] + 0.25, note["start_s"] + 1.0
        expected = fixtures.midi_to_hz(note["midi"], note["cents"])
        guess = float(np.nanmedian(f0[(times > start) & (times < end) & voiced]))
        refined = refine_pitch(bass[int(start * bass_sr) : int(end * bass_sr)], bass_sr, guess)
        raw_errors.append(cents(guess, expected))
        errors.append(cents(refined, expected))
    worst = max(abs(e) for e in errors)
    assert worst < 2, f"808 pitch errors (cents): {np.round(errors, 1)}"
    octave_slips = sum(abs(e) > 600 for e in raw_errors)
    raw_bias = float(np.median([e for e in raw_errors if abs(e) < 600]))
    return (
        f"tempo {tempo:.1f} BPM (truth {truth_tempo:.0f}); key {key} (margin {margin:.2f}); "
        f"808 notes: raw pYIN {octave_slips} octave slip(s) and {raw_bias:+.1f} cents bias, "
        f"within {worst:.2f} cents after spectral refinement"
    )


def refine_pitch(segment: np.ndarray, sr: float, guess_hz: float) -> float:
    """Zero-padded FFT peak near the guess or its octave, whichever clearly dominates."""
    n = 1 << 18
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    best_level, best_hz = 0.0, guess_hz
    for candidate in (guess_hz, 2 * guess_hz):
        window = np.flatnonzero(
            (freqs >= candidate * 2 ** (-1.5 / 12)) & (freqs <= candidate * 2 ** (1.5 / 12))
        )
        k = window[np.argmax(spectrum[window])]
        a, b, c = np.log(spectrum[k - 1 : k + 2])
        peak_hz = freqs[k] + 0.5 * (a - c) / (a - 2 * b + c) * (freqs[1] - freqs[0])
        if spectrum[k] > 4 * best_level:
            best_level, best_hz = float(spectrum[k]), float(peak_hz)
    return best_hz


def median_f0(y: np.ndarray, sr: int, start_s: float, end_s: float) -> float:
    import librosa

    f0, voiced, _ = librosa.pyin(y, fmin=80, fmax=600, sr=sr)
    times = librosa.times_like(f0, sr=sr)
    inside = (times > start_s) & (times < end_s) & voiced
    return float(np.nanmedian(f0[inside]))


def shift_report(ctx: Context, shifted: np.ndarray, sr: int, shift_cents: float) -> str:
    original, _ = sf.read(ctx.fx["vocal"])
    note = ctx.truth["vocal"]["notes"][0]
    window = (note["start_s"] + 0.05, note["start_s"] + 0.2)  # before vibrato starts
    measured = cents(median_f0(shifted, sr, *window), median_f0(original, sr, *window))
    assert abs(measured - shift_cents) < 5, f"shift measured {measured:.1f} cents"
    extra = len(shifted) - len(original)
    return f"asked {shift_cents:+.0f} cents, measured {measured:+.1f}; length delta {extra} samples"


def check_pyworld(ctx: Context) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pyworld as pw

    y, sr = sf.read(ctx.fx["vocal"])
    f0, t = pw.harvest(y, sr)
    f0 = pw.stonemask(y, f0, t, sr)
    envelope = pw.cheaptrick(y, f0, t, sr)
    aperiodicity = pw.d4c(y, f0, t, sr)
    shifted = pw.synthesize(f0 * 2 ** (-30 / 1200), envelope, aperiodicity, sr)
    return "harvest/cheaptrick/d4c/synthesize: " + shift_report(ctx, shifted, sr, -30)


def check_rubberband(ctx: Context) -> str:
    import pyrubberband as pyrb

    if shutil.which("rubberband") is None:
        raise Skip("rubberband CLI not installed (brew install rubberband)")
    version = (
        run(["rubberband", "--version"]).stderr or run(["rubberband", "--version"]).stdout
    ).strip()
    assert int(version.split(".")[0]) >= 3, f"R3 engine needs rubberband >= 3, found {version}"
    y, sr = sf.read(ctx.fx["vocal"])
    shifted = pyrb.pitch_shift(y, sr, n_steps=-0.3, rbargs={"--fine": "", "--formant": ""})
    return f"CLI {version}, R3 (--fine) + --formant: " + shift_report(ctx, shifted, sr, -30)


def check_separation(ctx: Context) -> str:
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise Skip(f"stems extra not installed ({exc}); uv sync --extra stems") from exc

    models = Path.home() / ".cache" / "masterbench-spike-models"
    separator = Separator(log_level=30, model_file_dir=str(models), output_dir=str(ctx.work))
    try:
        separator.load_model(model_filename=SEPARATION_MODEL)
    except Exception as exc:
        if (models / SEPARATION_MODEL).exists():
            raise
        raise Skip(f"could not download {SEPARATION_MODEL}: {str(exc)[:120]}") from exc
    started = time.perf_counter()
    outputs = separator.separate(str(ctx.fx["loop"]))
    elapsed = time.perf_counter() - started
    assert len(outputs) == 2, outputs
    duration = sf.info(ctx.fx["loop"]).duration
    stems = [re.search(r"\((\w+)\)", Path(o).name) for o in outputs]
    names = ", ".join(m.group(1) for m in stems if m)
    return f"{SEPARATION_MODEL}: stems [{names}], {duration:.0f}s audio in {elapsed:.1f}s on CPU"


MCP_SERVER = """
from mcp.server.mcpserver import MCPServer

server = MCPServer("masterbench-spike")

@server.tool()
def band_delta(mix_db: float, reference_db: float) -> dict[str, float]:
    \"\"\"Difference between two band levels.\"\"\"
    return {"delta_db": mix_db - reference_db}

if __name__ == "__main__":
    server.run("stdio")
"""


def check_mcp(ctx: Context) -> str:
    from mcp import Client, StdioServerParameters

    script = ctx.work / "mcp_spike_server.py"
    script.write_text(textwrap.dedent(MCP_SERVER), encoding="utf-8")

    async def roundtrip() -> tuple[list[str], dict]:
        params = StdioServerParameters(command=sys.executable, args=[str(script)])
        async with Client(params) as client:
            tools = await client.list_tools()
            result = await client.call_tool("band_delta", {"mix_db": -3.0, "reference_db": -1.5})
            return [t.name for t in tools.tools], result.structured_content or {}

    names, payload = asyncio.run(roundtrip())
    assert names == ["band_delta"] and payload.get("delta_db") == -1.5, (names, payload)
    return (
        "MCPServer (FastMCP's 2.x name): tool listed and called over stdio via mcp.Client; "
        "structured output needs a typed return annotation"
    )


def check_codecs(ctx: Context) -> str:
    encoders = run(["ffmpeg", "-hide_banner", "-encoders"]).stdout
    source, sr = sf.read(ctx.fx["reference"])
    source_tp = true_peak_dbtp(source)
    profiles = {
        "vorbis": ("libvorbis", "ogg", ["-q:a", "9"]),
        "aac": ("aac_at" if " aac_at " in encoders else "aac", "m4a", ["-b:a", "256k"]),
        "opus": ("libopus", "opus", ["-b:a", "160k"]),
    }
    findings = []
    for name, (encoder, ext, args) in profiles.items():
        assert f" {encoder} " in encoders, f"ffmpeg build lacks {encoder}"
        encoded, decoded = ctx.work / f"codec.{ext}", ctx.work / f"codec_{name}.wav"
        quiet = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        run([*quiet, "-i", str(ctx.fx["reference"]), "-c:a", encoder, *args, str(encoded)])
        run([*quiet, "-i", str(encoded), "-ar", str(sr), "-c:a", "pcm_f32le", str(decoded)])
        audio, _ = sf.read(decoded)
        findings.append(f"{name} ({encoder}) {true_peak_dbtp(audio) - source_tp:+.2f} dB")
    return f"true-peak change after decode, source {source_tp:+.2f} dBTP: " + ", ".join(findings)


def check_watchdog(ctx: Context) -> str:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    inbox = ctx.work / "inbox"
    inbox.mkdir()
    seen: list[str] = []

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            seen.append(Path(str(event.src_path)).name)

    observer = Observer()
    observer.schedule(Handler(), str(inbox), recursive=True)
    observer.start()
    try:
        shutil.copy(ctx.fx["vocal"], inbox / "bounce.wav")
        deadline = time.monotonic() + 5
        while "bounce.wav" not in seen and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        observer.stop()
        observer.join()
    assert "bounce.wav" in seen
    return f"{type(observer).__name__} saw the new file"


def check_gradio(ctx: Context) -> str:
    import gradio as gr
    import plotly.graph_objects as go

    with gr.Blocks(title="spike") as demo, gr.Tab("Master"):
        gr.Audio(label="Mix", type="filepath")
        gr.Plot(go.Figure(go.Scatter(x=[20, 20000], y=[0, -3])))
        gr.HTML("<div id='ab-player'></div>")
    assert demo.blocks
    return f"Blocks with Tab/Audio/Plot/HTML built ({len(demo.blocks)} components), not launched"


def check_anthropic(ctx: Context) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key="spike-not-a-real-key")
    assert callable(client.messages.create) and callable(client.messages.count_tokens)
    return "client constructed, messages.create present (no request sent)"


CHECKS: dict[str, Callable[[Context], str]] = {
    "sqlite": check_sqlite,
    "matchering": check_matchering,
    "pedalboard": check_pedalboard,
    "plugins": check_plugins,
    "loudness": check_loudness,
    "librosa": check_librosa,
    "pyworld": check_pyworld,
    "rubberband": check_rubberband,
    "separation": check_separation,
    "mcp": check_mcp,
    "codecs": check_codecs,
    "watchdog": check_watchdog,
    "gradio": check_gradio,
    "anthropic": check_anthropic,
}
NETWORK_CHECKS = frozenset({"separation"})


def versions() -> dict[str, str]:
    found = {"python": platform.python_version(), "platform": platform.platform()}
    for dist in DISTRIBUTIONS:
        try:
            found[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            found[dist] = "not installed"
    for tool, pattern in (("ffmpeg", r"ffmpeg version (\S+)"), ("rubberband", r"(\d+\.\d+\.\d+)")):
        if shutil.which(tool) is None:
            found[tool] = "not installed"
            continue
        flag = "-version" if tool == "ffmpeg" else "--version"
        proc = subprocess.run([tool, flag], capture_output=True, text=True)
        match = re.search(pattern, proc.stdout + proc.stderr)
        found[tool] = match.group(1) if match else "unknown"
    return found


def run_checks(names: list[str], work: Path) -> list[Outcome]:
    paths = fixtures.write_all(work / "fixtures")
    ctx = Context(work, paths, json.loads(paths["truth"].read_text(encoding="utf-8")))
    outcomes = []
    for name in names:
        started = time.perf_counter()
        try:
            status, detail = "PASS", CHECKS[name](ctx)
        except Skip as skip:
            status, detail = "SKIP", str(skip)
        except Exception as exc:
            status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
        outcomes.append(Outcome(name, status, detail, round(time.perf_counter() - started, 2)))
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default="", help="comma-separated check names")
    parser.add_argument("--skip", default="", help="comma-separated check names")
    parser.add_argument("--json", type=Path, help="write versions and outcomes to this file")
    args = parser.parse_args()

    names = [n for n in (args.only.split(",") if args.only else CHECKS) if n]
    skipped = [n for n in args.skip.split(",") if n]
    unknown = sorted(set(names + skipped) - set(CHECKS))
    if unknown:
        parser.error(f"unknown checks: {unknown}; choose from {', '.join(CHECKS)}")
    names = [n for n in names if n not in skipped]

    found = versions()
    print("\n".join(f"{name:>16}  {version}" for name, version in found.items()), "\n")
    with tempfile.TemporaryDirectory(prefix="masterbench-spike-") as tmp:
        outcomes = run_checks(names, Path(tmp))
    for o in outcomes:
        print(f"{o.status}  {o.name:<11} {o.seconds:>6.1f}s  {o.detail}")
    if args.json:
        payload = {"versions": found, "outcomes": [asdict(o) for o in outcomes]}
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 1 if any(o.status == "FAIL" for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
