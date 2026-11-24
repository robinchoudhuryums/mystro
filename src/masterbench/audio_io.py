"""Loading, saving and describing audio files."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

LOSSLESS_EXTENSIONS = frozenset({".wav", ".aif", ".aiff", ".flac"})
LOSSY_EXTENSIONS = frozenset({".mp3", ".m4a", ".aac", ".ogg", ".opus"})
AUDIO_EXTENSIONS = LOSSLESS_EXTENSIONS | LOSSY_EXTENSIONS

_BIT_DEPTHS = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}
_BIT_DEPTHS |= {"FLOAT": 32, "DOUBLE": 64}


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioInfo:
    format: str
    subtype: str
    sample_rate: int
    channels: int
    frames: int
    duration_s: float
    bit_depth: int | None
    lossy: bool


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def is_lossy(path: Path) -> bool:
    return path.suffix.lower() in LOSSY_EXTENSIONS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_with_ffmpeg(path: Path, out: Path) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path)]
    try:
        subprocess.run([*cmd, "-c:a", "pcm_f32le", str(out)], check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise AudioError(f"cannot read {path.name}: ffmpeg is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioError(f"cannot read {path.name}: {exc.stderr.decode().strip()}") from exc


def load(path: Path) -> tuple[np.ndarray, AudioInfo]:
    """Read a file as float64 with shape (frames, channels).

    Formats libsndfile cannot open (AAC/M4A) are decoded through ffmpeg.
    """
    if not path.is_file():
        raise AudioError(f"no such file: {path}")
    try:
        with sf.SoundFile(path) as handle:
            audio = handle.read(dtype="float64", always_2d=True)
            fmt, subtype, sr = handle.format, handle.subtype, handle.samplerate
    except sf.LibsndfileError:
        with tempfile.TemporaryDirectory() as tmp:
            decoded = Path(tmp) / "decoded.wav"
            _decode_with_ffmpeg(path, decoded)
            audio, sr = sf.read(decoded, dtype="float64", always_2d=True)
        fmt, subtype = path.suffix.lstrip(".").upper(), "LOSSY"
    lossy = is_lossy(path)
    info = AudioInfo(
        format=fmt,
        subtype=subtype,
        sample_rate=int(sr),
        channels=audio.shape[1],
        frames=len(audio),
        duration_s=len(audio) / sr,
        bit_depth=None if lossy else _BIT_DEPTHS.get(subtype),
        lossy=lossy,
    )
    return audio, info


def tpdf_dither(audio: np.ndarray, bit_depth: int, seed: int) -> np.ndarray:
    """Quantize to integers at the given depth with triangular (TPDF) dither of ±1 LSB."""
    rng = np.random.default_rng(seed)
    scale = float(2 ** (bit_depth - 1))
    noise = rng.random(audio.shape) - rng.random(audio.shape)
    return np.clip(np.round(audio * scale + noise), -scale, scale - 1)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int, bit_depth: int = 24) -> Path:
    """Write a WAV. 16-bit output is TPDF-dithered, seeded from the audio so it is repeatable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if bit_depth == 16:
        seed = int.from_bytes(hashlib.sha256(audio.tobytes()).digest()[:8], "big")
        sf.write(path, tpdf_dither(audio, 16, seed).astype(np.int16), sample_rate, subtype="PCM_16")
    elif bit_depth == 24:
        sf.write(path, audio, sample_rate, subtype="PCM_24")
    elif bit_depth == 32:
        sf.write(path, audio, sample_rate, subtype="FLOAT")
    else:
        raise ValueError(f"unsupported bit depth: {bit_depth}")
    return path
