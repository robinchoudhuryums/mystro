from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from masterbench import audio_io, fixtures, validate
from masterbench.analysis.report import analyze_file
from masterbench.config import Settings


def codes(path: Path, settings: Settings) -> set[str]:
    return {flag.code for flag in analyze_file(path, "mix", settings).flags}


def test_clean_mix_has_no_flags(fx: dict[str, Path], settings: Settings) -> None:
    assert codes(fx["loop"], settings) == {"bass_detuned"}, "only the planted detuned 808 note"


def test_clipped_mix_is_flagged_with_the_logic_hint(
    fx: dict[str, Path], settings: Settings
) -> None:
    report = analyze_file(fx["clipped"], "mix", settings)
    by_code = {flag.code: flag for flag in report.flags}
    assert {"clipping", "already_mastered"} <= set(by_code)
    assert by_code["clipping"].severity == "problem" and (by_code["clipping"].value or 0) > 10
    assert "master bus bypassed" in by_code["already_mastered"].message


def test_loud_reference_as_a_mix_looks_already_mastered(fx: dict[str, Path], settings) -> None:
    flags = {f.code: f for f in analyze_file(fx["reference"], "mix", settings).flags}
    assert "already_mastered" in flags and "-8.7 LUFS" in flags["already_mastered"].message


def test_flags_only_apply_to_mixes(fx: dict[str, Path], settings: Settings) -> None:
    assert analyze_file(fx["clipped"], "reference", settings).flags == []


def test_lossy_target_and_mono_are_called_out(
    fx: dict[str, Path], tmp_path: Path, settings
) -> None:
    mp3 = tmp_path / "vocal.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(fx["vocal"]), str(mp3)], check=True
    )
    found = codes(mp3, settings)
    assert {"lossy_target", "mono"} <= found


def test_dc_offset_and_edge_silence(tmp_path: Path, settings: Settings) -> None:
    sr = fixtures.SR
    noise = fixtures.pink_noise(4.0)
    padded = np.concatenate([np.zeros((3 * sr, 2)), noise + 0.01, np.zeros((5 * sr, 2))])
    path = audio_io.save_wav(tmp_path / "padded.wav", padded, sr)
    flags = {f.code: f for f in analyze_file(path, "mix", settings).flags}
    assert flags["dc_offset"].value is not None and flags["dc_offset"].value > 0.001
    # The offset makes the padding non-silent, so measure silence on a clean copy.
    clean = np.concatenate([np.zeros((3 * sr, 2)), noise, np.zeros((5 * sr, 2))])
    head, tail = validate.edge_silence(clean, sr)
    assert (round(head), round(tail)) == (3, 5)
    clean_path = audio_io.save_wav(tmp_path / "clean.wav", clean, sr)
    assert {"silence_head", "silence_tail"} <= codes(clean_path, settings)


def test_clipped_runs_counts_runs_not_samples() -> None:
    channel = np.zeros(100)
    channel[10:15] = 1.0
    channel[40:42] = -1.0
    channel[60:70] = -1.0
    audio = np.column_stack([channel, np.zeros(100)])
    assert validate.clipped_runs(audio, min_run=3) == 2
    assert validate.clipped_runs(audio, min_run=2) == 3
