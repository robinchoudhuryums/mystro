from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from masterbench import fixtures
from masterbench.analysis import lowend
from masterbench.analysis.lowend import LowEnd
from masterbench.config import Thresholds

SR = fixtures.SR
LIMITS = Thresholds()


@pytest.fixture(scope="module")
def loop() -> LowEnd:
    return lowend.analyze(fixtures.drum_808_loop()[0], SR, "F minor", LIMITS)


@pytest.fixture(scope="module")
def stems() -> tuple[dict[str, np.ndarray], np.ndarray]:
    return fixtures.song_stems()


def test_808_notes_are_read_from_the_full_mix(loop: LowEnd) -> None:
    _, truth = fixtures.drum_808_loop()
    # Two bars of the same note in a row read as one held note.
    expected = [
        n for i, n in enumerate(truth.notes_808) if i == 0 or n.midi != truth.notes_808[i - 1].midi
    ]
    assert [n.note for n in loop.notes] == [n.name for n in expected]
    for found, planted in zip(loop.notes, expected, strict=True):
        assert found.cents == pytest.approx(planted.cents, abs=1.0)
        assert found.freq_hz == pytest.approx(planted.freq_hz, rel=0.001)
        assert found.start_s == pytest.approx(planted.start_s, abs=0.35)
    assert loop.source == "mix" and loop.clash is None


def test_the_detuned_note_is_found_within_ten_cents(loop: LowEnd) -> None:
    (flag,) = lowend.flags(loop, LIMITS)
    assert flag.code == "bass_detuned"
    assert flag.value == pytest.approx(fixtures.DETUNED_CENTS, abs=10.0)
    assert flag.value == pytest.approx(fixtures.DETUNED_CENTS, abs=1.0), "in fact within a cent"
    assert "35 cents sharp of C2" in flag.message and flag.start_s == pytest.approx(8.57, abs=0.2)
    assert loop.in_tune_share == pytest.approx(0.87, abs=0.04)
    clean = lowend.analyze(fixtures.drum_808_loop(detune=False)[0], SR, "F minor", LIMITS)
    assert lowend.flags(clean, LIMITS) == [] and clean.in_tune_share == 1.0


def test_notes_outside_the_key_are_named() -> None:
    found = lowend.analyze(fixtures.drum_808_loop(detune=False)[0], SR, "G major", LIMITS)
    flags = lowend.flags(found, LIMITS)
    assert flags and all(f.code == "bass_out_of_key" for f in flags)
    assert "which is not in G major" in flags[0].message
    unknown = lowend.analyze(fixtures.drum_808_loop(detune=False)[0], SR, None, LIMITS)
    assert all(n.in_key is None for n in unknown.notes) and lowend.flags(unknown, LIMITS) == []


def test_rumble_voices_and_sweeps_are_not_bass_notes() -> None:
    for audio in (
        fixtures.pink_noise(),
        fixtures.synthetic_vocal()[0][:, None],
        fixtures.sine_sweep(),
    ):
        assert lowend.analyze(audio, SR, None, LIMITS).notes == []
    silent = lowend.analyze(np.zeros((SR * 2, 2)), SR, None, LIMITS)
    assert silent.notes == [] and silent.in_tune_share is None and silent.harmonics_db is None


def test_stems_give_pitches_harmonics_and_the_kick_clash(stems) -> None:
    parts, mix = stems
    found = lowend.analyze(mix, SR, "F minor", LIMITS, bass=parts["bass"], drums=parts["drums"])
    assert found.source == "bass stem" and len(found.notes) == 7, "seven different notes"
    assert found.harmonic_hz is not None and found.harmonic_hz >= LIMITS.small_speaker_hz
    assert found.harmonics_db is not None and -30 < found.harmonics_db < -5
    clash = found.clash
    assert clash is not None and clash.hits >= 14 and clash.share > 0.9
    assert clash.low_hz >= 40 and clash.high_hz <= 120
    codes = {f.code: f for f in lowend.flags(found, LIMITS)}
    assert "Sidechain the 808 to the kick" in codes["kick_bass_clash"].message
    assert LowEnd.from_dict(asdict(found)) == found


def test_a_sidechained_808_does_not_clash(stems) -> None:
    parts, mix = stems
    _, truth = fixtures.drum_808_loop()
    duck = np.ones(len(mix))
    for hit in truth.kick_times_s:
        duck[max(0, int(hit * SR) - 200) : int((hit + 0.15) * SR)] = 0.05
    found = lowend.analyze(
        mix, SR, "F minor", LIMITS, bass=parts["bass"] * duck[:, None], drums=parts["drums"]
    )
    assert found.clash is not None and found.clash.share == 0.0
    assert "kick_bass_clash" not in {f.code for f in lowend.flags(found, LIMITS)}


def test_a_pure_sub_is_flagged_for_small_speakers() -> None:
    t = np.arange(SR * 4) / SR
    sub = 0.5 * np.sin(2 * np.pi * fixtures.midi_to_hz(29) * t)
    found = lowend.analyze(np.column_stack([sub, sub]), SR, "F minor", LIMITS)
    assert [n.note for n in found.notes] == ["F1"]
    assert found.harmonics_db is not None and found.harmonics_db < LIMITS.bass_harmonics_db
    (flag,) = lowend.flags(found, LIMITS)
    assert flag.code == "bass_small_speakers" and "Saturation" in flag.message


def test_kick_hits_are_located(stems) -> None:
    parts, _ = stems
    _, truth = fixtures.drum_808_loop()
    found = lowend.kick_times(parts["drums"].mean(axis=1), SR)
    assert len(found) >= len(truth.kick_times_s) - 1
    assert all(min(abs(hit - k) for k in truth.kick_times_s) < 0.03 for hit in found)
