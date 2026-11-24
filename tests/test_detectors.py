from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from masterbench import fixtures
from masterbench.analysis import detectors
from masterbench.config import Thresholds

SR = fixtures.SR
LIMITS = Thresholds()


@pytest.fixture(scope="module")
def song() -> np.ndarray:
    return fixtures.arranged_song()[0]


def test_planted_sibilance_is_found_at_its_times() -> None:
    vocal, times = fixtures.sibilant_vocal()
    flags = detectors.sibilance(vocal, SR, LIMITS, is_vocal=True)
    summary, events = flags[0], flags[1:]
    assert summary.code == "sibilance" and summary.value == 3 and "de-esser" in summary.message
    assert [e.start_s for e in events] == pytest.approx(times, abs=0.05)
    assert all(e.code == "sibilance_at" and e.end_s and e.end_s > (e.start_s or 0) for e in events)


def test_clean_or_gently_sibilant_vocals_are_left_alone() -> None:
    assert detectors.sibilance(fixtures.synthetic_vocal()[0], SR, LIMITS, True) == []
    quiet, _ = fixtures.sibilant_vocal(burst_db=-20.0)
    assert detectors.sibilance(quiet, SR, LIMITS, True) == []


@pytest.mark.parametrize("name", ["loop", "song", "pink", "reference"])
def test_clean_mixes_raise_nothing(name: str, song: np.ndarray) -> None:
    audio = {
        "loop": fixtures.drum_808_loop()[0],
        "song": song,
        "pink": fixtures.pink_noise(),
        "reference": fixtures.reference_master(),
    }[name]
    assert detectors.run(audio, SR, LIMITS) == []


@pytest.mark.parametrize(("freq", "q"), [(2400.0, 20.0), (5200.0, 30.0)])
def test_planted_resonance_is_found(song: np.ndarray, freq: float, q: float) -> None:
    b, a = signal.iirpeak(freq, q, fs=SR)
    ringing = song + 3.0 * np.asarray(signal.lfilter(b, a, song, axis=0))
    (flag,) = detectors.resonances(ringing.mean(axis=1), SR, LIMITS)
    assert flag.code == "resonance" and flag.value == pytest.approx(freq, rel=0.02)
    assert "narrow cut" in flag.message and flag.start_s is None


def test_held_notes_are_not_called_resonances() -> None:
    t = np.arange(SR * 8) / SR
    drone = fixtures.pink_noise(8.0)[:, 0] * 0.2 + 0.2 * np.sin(2 * np.pi * 440.0 * t)
    assert detectors.resonances(drone, SR, LIMITS) == []


def test_harsh_burst_is_found_with_its_time() -> None:
    loop, _ = fixtures.drum_808_loop()
    rng = np.random.default_rng(0)
    burst = fixtures._butter(rng.standard_normal(int(0.4 * SR)), 4, [2200, 4500], "bandpass", SR)
    harsh = loop.copy()
    harsh[5 * SR : 5 * SR + len(burst)] += 0.5 * burst[:, None]
    (flag,) = detectors.harshness(harsh.mean(axis=1), SR, LIMITS)
    assert flag.code == "harsh_at" and flag.start_s == pytest.approx(5.0, abs=0.05)
    assert flag.end_s == pytest.approx(5.4, abs=0.05) and "0:05" in flag.message


def test_out_of_phase_stretch_is_found(song: np.ndarray) -> None:
    flipped = song.copy()
    flipped[30 * SR : 36 * SR, 1] *= -1
    (flag,) = detectors.phase_events(flipped, SR, LIMITS)
    assert flag.code == "phase_at" and flag.value is not None and flag.value < -0.9
    assert flag.start_s == pytest.approx(30.0, abs=1.0) and flag.end_s == pytest.approx(
        36.0, abs=1.0
    )
    assert detectors.phase_events(song.mean(axis=1, keepdims=True), SR, LIMITS) == []


def test_uneven_vocal_phrases_are_found() -> None:
    stems, mix = fixtures.song_stems()
    beat = mix - stems["vocals"]
    assert detectors.vocal_consistency(stems["vocals"], beat, SR, LIMITS) == []
    vocal = stems["vocals"].copy()
    vocal[4 * SR : int(6.5 * SR)] *= 10 ** (-8 / 20)
    vocal[9 * SR : 11 * SR] *= 10 ** (6 / 20)
    flags = detectors.vocal_consistency(vocal, beat, SR, LIMITS)
    quiet = next(f for f in flags if f.code == "vocal_phrase_quiet")
    loud = next(f for f in flags if f.code == "vocal_phrase_loud")
    assert quiet.start_s == pytest.approx(4.0, abs=0.6) and (quiet.value or 0) < -6
    assert loud.start_s == pytest.approx(9.0, abs=0.6) and (loud.value or 0) > 4
    assert "rider" in quiet.message


def test_clock() -> None:
    assert [detectors.clock(t) for t in (0, 5.9, 61.2, 600)] == ["0:00", "0:05", "1:01", "10:00"]
