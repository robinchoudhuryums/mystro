"""Streaming previews: what the codecs and the platforms' loudness normalization do to a master.

Spotify's rules follow its "Loudness normalization" support article (read 2026-11-24):
-14 LUFS by default, Loud -11 and Quiet -19 for Premium. Louder masters are turned down.
Quieter ones are turned up only as far as 1 dB of peak headroom allows ("-20 LUFS with a
-5 dBFS true peak is only lifted to -16"), except in Loud mode, which applies the full gain
and a limiter (-1 dB, 5 ms attack, 100 ms decay). Albums played in order get one gain for
the whole album; that is not modelled here. The web player does not normalize at all.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import numpy as np

from masterbench import audio_io, projects
from masterbench.analysis import loudness
from masterbench.analysis.flags import Flag
from masterbench.config import Settings
from masterbench.master.pipeline import RunResult

PREVIEW_DIR = "previews"
SUMMARY_FILE = "streaming.json"
SPOTIFY_HEADROOM_DBFS = -1.0


@dataclass(frozen=True)
class Codec:
    name: str
    label: str
    encoders: tuple[str, ...]
    extension: str
    args: tuple[str, ...]


CODECS = (
    Codec("vorbis", "Ogg Vorbis ~320k (Spotify-like)", ("libvorbis",), "ogg", ("-q:a", "9")),
    Codec("aac", "AAC 256k (Apple-like)", ("aac_at", "aac"), "m4a", ("-b:a", "256k")),
    Codec("opus", "Opus 128k (YouTube-like)", ("libopus",), "opus", ("-b:a", "128k")),
)


@dataclass(frozen=True)
class CodecPreview:
    name: str
    label: str
    encoder: str
    encoded_file: str
    decoded_file: str
    true_peak_dbtp: float
    peak_change_db: float
    lufs_i: float


@dataclass(frozen=True)
class PlatformGain:
    platform: str
    target_lufs: float
    gain_db: float
    result_lufs: float
    message: str


@dataclass(frozen=True)
class StreamingPreview:
    source: str
    source_true_peak_dbtp: float
    codecs: list[CodecPreview]
    skipped: list[str]
    platforms: list[PlatformGain]
    flags: list[Flag]

    @classmethod
    def from_dict(cls, data: dict) -> StreamingPreview:
        return cls(
            source=data["source"],
            source_true_peak_dbtp=data["source_true_peak_dbtp"],
            codecs=[CodecPreview(**c) for c in data["codecs"]],
            skipped=data["skipped"],
            platforms=[PlatformGain(**p) for p in data["platforms"]],
            flags=[Flag(**f) for f in data["flags"]],
        )


class PreviewError(RuntimeError):
    pass


@cache
def available_encoders() -> frozenset[str]:
    try:
        cmd = ["ffmpeg", "-hide_banner", "-encoders"]
        listing = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise PreviewError("ffmpeg is needed for codec previews (brew install ffmpeg)") from exc
    return frozenset(line.split()[1] for line in listing.splitlines() if len(line.split()) > 1)


def pick_encoder(codec: Codec, encoders: frozenset[str]) -> str | None:
    """The first of the codec's encoders this ffmpeg build has (Apple's AAC before ffmpeg's)."""
    return next((encoder for encoder in codec.encoders if encoder in encoders), None)


def _ffmpeg(*args: str) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise PreviewError(f"ffmpeg failed: {done.stderr.strip()[-300:]}")


def _round_trip(source: Path, codec: Codec, encoder: str, out_dir: Path) -> tuple[Path, Path]:
    """Encode, then decode to float WAV at the source rate and length (peaks left unclipped)."""
    audio, info = audio_io.load(source)
    encoded = out_dir / f"{codec.name}.{codec.extension}"
    decoded = out_dir / f"{codec.name}.wav"
    _ffmpeg("-i", str(source), "-c:a", encoder, *codec.args, str(encoded))
    _ffmpeg("-i", str(encoded), "-ar", str(info.sample_rate), "-c:a", "pcm_f32le", str(decoded))
    result, _ = audio_io.load(decoded)
    fitted = np.zeros_like(audio)
    frames = min(len(audio), len(result))
    fitted[:frames] = result[:frames, : audio.shape[1]]
    audio_io.save_wav(decoded, fitted, info.sample_rate, 32)
    return encoded, decoded


def _turn(gain_db: float) -> str:
    if abs(gain_db) < 0.05:
        return "plays this unchanged"
    return f"will turn this {'up' if gain_db > 0 else 'down'} {abs(gain_db):.1f} dB"


def platform_gains(lufs_i: float, true_peak_dbtp: float) -> list[PlatformGain]:
    """Playback gain each platform's normalization would apply to this master."""
    headroom = max(0.0, SPOTIFY_HEADROOM_DBFS - true_peak_dbtp)
    gains = []
    for mode, target in (("Loud", -11.0), ("Normal", -14.0), ("Quiet", -19.0)):
        wanted = target - lufs_i
        gain = wanted if mode == "Loud" or wanted <= 0 else min(wanted, headroom)
        message = f"Spotify ({mode}) {_turn(gain)}."
        if gain < wanted - 0.05:
            message = (
                f"Spotify ({mode}) can only turn this up {gain:.1f} dB before peaks reach "
                f"{SPOTIFY_HEADROOM_DBFS:.0f} dBFS, so it plays at {lufs_i + gain:.1f} LUFS, "
                f"quieter than other tracks."
            )
        elif mode == "Loud" and gain > headroom + 0.05:
            message += " Its limiter will act on the peaks."
        gains.append(PlatformGain(f"Spotify {mode}", target, gain, lufs_i + gain, message))
    down_only = min(0.0, -14.0 - lufs_i)
    message = f"A -14 LUFS turn-down-only platform (YouTube-style) {_turn(down_only)}."
    gains.append(PlatformGain("Generic -14", -14.0, down_only, lufs_i + down_only, message))
    return [
        PlatformGain(
            g.platform, g.target_lufs, round(g.gain_db, 2), round(g.result_lufs, 2), g.message
        )
        for g in gains
    ]


def _codec_flags(codecs: list[CodecPreview], ceiling: float, limit: float) -> list[Flag]:
    over = [c for c in codecs if c.true_peak_dbtp > limit]
    if not over:
        return []
    worst = max(over, key=lambda c: c.true_peak_dbtp)
    overshoot = worst.true_peak_dbtp - limit
    suggested = ceiling - math.ceil(overshoot * 2) / 2
    severity = "problem" if worst.true_peak_dbtp >= 0 else "warn"
    message = (
        f"Encoding pushes the true peak to {worst.true_peak_dbtp:+.2f} dBTP ({worst.label})"
        f"{', which clips on playback' if severity == 'problem' else ''}. "
        f"A ceiling of {suggested:.1f} dBTP would keep the encoded file under {limit:.1f}."
    )
    return [Flag("codec_peak_over", severity, message, worst.true_peak_dbtp)]


def render(run: RunResult, settings: Settings, force: bool = False) -> StreamingPreview:
    """Codec round trips and platform gains for a master run, cached under previews/."""
    if "master_24" not in run.manifest.outputs:
        raise PreviewError("streaming previews need a master; this run only has an analysis")
    out_dir = run.run_dir / PREVIEW_DIR
    cached = load(run)
    if cached is not None and not force:
        return cached

    source = run.output("master_24")
    master = run.reports["master"].loudness
    out_dir.mkdir(exist_ok=True)
    codecs, skipped = [], []
    for codec in CODECS:
        encoder = pick_encoder(codec, available_encoders())
        if encoder is None:
            skipped.append(f"{codec.label}: this ffmpeg has none of {', '.join(codec.encoders)}")
            continue
        encoded, decoded = _round_trip(source, codec, encoder, out_dir)
        audio, info = audio_io.load(decoded)
        peak = loudness.to_db(loudness.true_peak(audio))
        codecs.append(
            CodecPreview(
                name=codec.name,
                label=codec.label,
                encoder=encoder,
                encoded_file=f"{PREVIEW_DIR}/{encoded.name}",
                decoded_file=f"{PREVIEW_DIR}/{decoded.name}",
                true_peak_dbtp=round(peak, 2),
                peak_change_db=round(peak - master.true_peak_dbtp, 2),
                lufs_i=round(loudness.integrated(audio, info.sample_rate), 2),
            )
        )
    ceiling = run.manifest.settings.get("ceiling_dbtp", settings.targets.true_peak_dbtp)
    preview = StreamingPreview(
        source=run.manifest.outputs["master_24"],
        source_true_peak_dbtp=round(master.true_peak_dbtp, 2),
        codecs=codecs,
        skipped=skipped,
        platforms=platform_gains(master.lufs_i, master.true_peak_dbtp),
        flags=_codec_flags(codecs, ceiling, settings.thresholds.codec_peak_dbtp),
    )
    projects.write_json(out_dir / SUMMARY_FILE, asdict(preview))
    return preview


def load(run: RunResult) -> StreamingPreview | None:
    summary = run.run_dir / PREVIEW_DIR / SUMMARY_FILE
    if not summary.exists():
        return None
    preview = StreamingPreview.from_dict(json.loads(summary.read_text(encoding="utf-8")))
    files = [run.run_dir / c.decoded_file for c in preview.codecs]
    return preview if all(f.exists() for f in files) else None
