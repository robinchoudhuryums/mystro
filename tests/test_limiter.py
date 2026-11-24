from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from masterbench import audio_io
from masterbench.analysis import loudness
from masterbench.master import limiter


@pytest.fixture(scope="module")
def loop(fx: dict[str, Path]) -> tuple[np.ndarray, int]:
    audio, info = audio_io.load(fx["loop"])
    return audio, info.sample_rate


def tp_db(audio: np.ndarray) -> float:
    return loudness.to_db(loudness.true_peak(audio))


@pytest.mark.parametrize("push_db", [6.0, 9.0, 18.0])
@pytest.mark.parametrize("ceiling", [-1.0, -2.0])
def test_ceiling_is_never_exceeded(loop, push_db: float, ceiling: float) -> None:
    audio, sr = loop
    result = limiter.true_peak_limit(audio * 10 ** (push_db / 20), sr, ceiling)
    assert result.audio.shape == audio.shape
    assert tp_db(result.audio) <= ceiling + 1e-6
    assert result.true_peak_dbtp == pytest.approx(tp_db(result.audio), abs=1e-6)
    assert result.max_reduction_db > 0


def test_audio_under_the_ceiling_is_untouched(loop) -> None:
    audio, sr = loop
    result = limiter.true_peak_limit(audio, sr, -1.0)
    assert result.audio is audio and result.max_reduction_db == 0.0 and result.trim_db == 0.0


def test_gain_is_linked_across_channels(loop) -> None:
    audio, sr = loop
    wide = np.column_stack([audio[:, 0], 0.25 * audio[:, 0]]) * 10 ** (12 / 20)
    out = limiter.true_peak_limit(wide, sr, -1.0).audio
    assert np.allclose(out[:, 1], 0.25 * out[:, 0], atol=1e-9), "stereo image must not shift"


def test_limiting_only_ever_turns_down(loop) -> None:
    audio, sr = loop
    loud = audio * 10 ** (12 / 20)
    out = limiter.true_peak_limit(loud, sr, -1.0).audio
    assert np.all(np.abs(out) <= np.abs(loud) + 1e-12)
    quiet_part = np.abs(loud).max(axis=1) < 0.05
    assert np.abs(out[quiet_part]).sum() > 0.5 * np.abs(loud[quiet_part]).sum()


@pytest.mark.parametrize("target", [-16.0, -14.0, -11.0])
def test_finalize_lands_on_target_under_the_ceiling(loop, target: float) -> None:
    audio, sr = loop
    result = limiter.finalize(audio, sr, target, -1.0)
    assert loudness.integrated(result.audio, sr) == pytest.approx(target, abs=0.1)
    assert result.lufs_i == pytest.approx(loudness.integrated(result.audio, sr))
    assert tp_db(result.audio) <= -1.0 + 1e-6


def test_finalize_refuses_to_crush_for_an_unreachable_target(loop) -> None:
    audio, sr = loop
    result = limiter.finalize(audio, sr, -4.0, -1.0, max_reduction_db=6.0)
    assert result.max_reduction_db <= 6.5
    assert result.lufs_i < -6.0, "comes out quieter than asked instead of over-limiting"
    assert tp_db(result.audio) <= -1.0 + 1e-6


def test_finalize_without_a_target_only_limits(loop) -> None:
    audio, sr = loop
    result = limiter.finalize(audio * 10 ** (10 / 20), sr, None, -1.0)
    assert result.gain_db == 0.0 and result.iterations == 1
    assert tp_db(result.audio) <= -1.0 + 1e-6
