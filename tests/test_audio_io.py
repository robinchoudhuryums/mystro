from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from masterbench import audio_io


def encode(source: Path, dest: Path) -> Path:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), str(dest)]
    subprocess.run(cmd, check=True)
    return dest


def test_load_is_always_2d_float(fx: dict[str, Path]) -> None:
    stereo, stereo_info = audio_io.load(fx["loop"])
    mono, mono_info = audio_io.load(fx["vocal"])
    assert stereo.dtype == np.float64 and stereo.shape == (stereo_info.frames, 2)
    assert mono.shape == (mono_info.frames, 1) and mono_info.channels == 1
    assert (stereo_info.sample_rate, stereo_info.bit_depth, stereo_info.lossy) == (44100, 24, False)
    assert stereo_info.duration_s == pytest.approx(stereo_info.frames / 44100)


@pytest.mark.parametrize("extension", [".mp3", ".m4a"])
def test_lossy_files_load_and_are_marked(
    fx: dict[str, Path], tmp_path: Path, extension: str
) -> None:
    lossy = encode(fx["loop"], tmp_path / f"loop{extension}")
    audio, info = audio_io.load(lossy)
    assert info.lossy and info.bit_depth is None and info.channels == 2
    assert abs(info.duration_s - sf.info(fx["loop"]).duration) < 0.1
    assert np.abs(audio).max() > 0.1


def test_missing_and_unreadable_files_raise(tmp_path: Path) -> None:
    with pytest.raises(audio_io.AudioError, match="no such file"):
        audio_io.load(tmp_path / "nope.wav")
    garbage = tmp_path / "garbage.m4a"
    garbage.write_bytes(b"not audio at all")
    with pytest.raises(audio_io.AudioError, match="cannot read"):
        audio_io.load(garbage)


def test_save_24_bit_round_trip(tmp_path: Path) -> None:
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, (1000, 2))
    path = audio_io.save_wav(tmp_path / "out" / "a.wav", audio, 48000, 24)
    loaded, info = audio_io.load(path)
    assert (info.subtype, info.sample_rate) == ("PCM_24", 48000)
    assert np.abs(loaded - audio).max() <= 1 / 2**23


def test_16_bit_export_is_tpdf_dithered_and_repeatable(tmp_path: Path) -> None:
    # A signal far below one 16-bit step: plain rounding would erase it, dither keeps it.
    t = np.arange(48000) / 48000
    quiet = np.column_stack([np.sin(2 * np.pi * 1000 * t)] * 2) * (0.3 / 32768)
    first = audio_io.save_wav(tmp_path / "a.wav", quiet, 48000, 16)
    second = audio_io.save_wav(tmp_path / "b.wav", quiet, 48000, 16)
    assert first.read_bytes() == second.read_bytes()

    stored, _ = sf.read(first, dtype="int16")
    assert sf.info(first).subtype == "PCM_16"
    assert set(np.unique(stored)) <= {-1, 0, 1} and np.any(stored != 0)
    correlation = np.corrcoef(stored[:, 0], quiet[:, 0])[0, 1]
    assert correlation > 0.2, "the sub-LSB tone should survive in the dithered output"
    error = stored / 32768 - quiet
    assert abs(error.mean()) < 0.05 / 32768


def test_sha256_tracks_content(tmp_path: Path) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"x" * 3_000_000)
    b.write_bytes(b"x" * 3_000_000)
    assert audio_io.file_sha256(a) == audio_io.file_sha256(b)
    b.write_bytes(b"y")
    assert audio_io.file_sha256(a) != audio_io.file_sha256(b)
