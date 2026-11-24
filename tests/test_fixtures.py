from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import cents, peak_hz
from masterbench import fixtures
from masterbench.fixtures import SR


def rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(x**2)))


def longest_full_scale_run(x: np.ndarray) -> int:
    longest = current = 0
    for hit in np.abs(x) >= 1.0:
        current = current + 1 if hit else 0
        longest = max(longest, current)
    return longest


def test_generation_is_deterministic() -> None:
    assert np.array_equal(fixtures.pink_noise(2.0), fixtures.pink_noise(2.0))
    assert np.array_equal(fixtures.drum_808_loop()[0], fixtures.drum_808_loop()[0])
    assert not np.array_equal(fixtures.pink_noise(2.0, seed=0), fixtures.pink_noise(2.0, seed=1))


def test_pink_noise_level_and_slope() -> None:
    noise = fixtures.pink_noise()
    assert noise.shape == (10 * SR, 2)
    assert rms_db(noise) == pytest.approx(-20.0, abs=0.01)
    spectrum = np.abs(np.fft.rfft(noise.mean(axis=1))) ** 2
    freqs = np.fft.rfftfreq(len(noise), 1 / SR)
    octaves = [(100, 200), (1000, 2000), (5000, 10000)]
    energy = [10 * np.log10(spectrum[(freqs >= lo) & (freqs < hi)].sum()) for lo, hi in octaves]
    assert max(energy) - min(energy) < 1.0, "pink noise should carry equal energy per octave"


def test_sweep_peak() -> None:
    sweep = fixtures.sine_sweep()
    assert 20 * np.log10(np.abs(sweep).max()) == pytest.approx(-6.0, abs=0.05)
    assert np.array_equal(sweep[:, 0], sweep[:, 1])


def test_loop_truth_describes_the_audio() -> None:
    audio, truth = fixtures.drum_808_loop()
    beat = 60 / truth.tempo_bpm
    assert truth.key == "F minor" and truth.bars == 8
    assert len(audio) == round(truth.bars * 4 * beat * SR)
    assert [n.cents for n in truth.notes_808].count(fixtures.DETUNED_CENTS) == 1
    assert truth.notes_808[fixtures.DETUNED_BAR].cents == fixtures.DETUNED_CENTS
    assert truth.notes_808[0].name == "F1"

    mono = audio.mean(axis=1)
    for note in truth.notes_808:
        segment = mono[int((note.start_s + 0.25) * SR) : int((note.start_s + 1.0) * SR)]
        nominal = fixtures.midi_to_hz(note.midi)
        measured = peak_hz(segment, SR, nominal * 0.94, nominal * 1.06)
        assert cents(measured, nominal) == pytest.approx(note.cents, abs=1.0)


def test_only_the_clipped_mix_clips() -> None:
    loop, _ = fixtures.drum_808_loop()
    clipped = fixtures.clipped_mix()
    assert longest_full_scale_run(loop[:, 0]) == 0
    assert longest_full_scale_run(clipped[:, 0]) >= 3
    assert np.abs(clipped).max() == 1.0


def test_reference_is_much_louder_than_the_mix() -> None:
    loop, _ = fixtures.drum_808_loop()
    assert rms_db(fixtures.reference_master()) - rms_db(loop) > 6.0


def test_vocal_off_notes_are_planted() -> None:
    audio, truth = fixtures.synthetic_vocal(vibrato_cents=0.0)
    assert audio.ndim == 1
    off = [i for i, note in enumerate(truth.notes) if note.cents]
    assert off == [2, 5] and all(truth.notes[i].cents == 30.0 for i in off)
    for note in truth.notes:
        segment = audio[int((note.start_s + 0.05) * SR) : int((note.start_s + 0.75) * SR)]
        nominal = fixtures.midi_to_hz(note.midi)
        measured = peak_hz(segment, SR, nominal * 0.96, nominal * 1.04)
        assert cents(measured, nominal) == pytest.approx(note.cents, abs=1.0)


def test_write_all(fx: dict[str, Path], truth: dict) -> None:
    assert set(fx) == {"pink_noise", "sine_sweep", "loop", "reference", "clipped", "vocal", "truth"}
    for name, path in fx.items():
        if name == "truth":
            continue
        info = sf.info(path)
        assert (info.samplerate, info.subtype) == (SR, "PCM_24")
        assert info.channels == (1 if name == "vocal" else 2)
    assert truth["loop"]["tempo_bpm"] == 140.0
    assert len(truth["vocal"]["notes"]) == 8
