from __future__ import annotations

import numpy as np
import pytest

from masterbench import fixtures
from masterbench.vocal import pitch_report, pitch_track, scales
from masterbench.vocal.scales import ScaleError

SR = fixtures.SR


@pytest.fixture(scope="module")
def vocal():
    audio, truth = fixtures.synthetic_vocal()
    pitch = pitch_track.track(audio, SR)
    return audio, truth, pitch, pitch_track.segment(pitch)


@pytest.fixture(scope="module")
def legato():
    audio, truth = fixtures.legato_vocal()
    pitch = pitch_track.track(audio, SR)
    return audio, truth, pitch, pitch_track.segment(pitch)


def test_scales_and_note_names() -> None:
    minor = scales.scale("F minor")
    assert minor.label == "F minor" and minor.note_names() == [
        "C",
        "C#",
        "D#",
        "F",
        "G",
        "G#",
        "A#",
    ]
    assert scales.scale("F minor", "dorian").note_names() == ["C", "D", "D#", "F", "G", "G#", "A#"]
    assert scales.scale("f minor", "Minor_Pentatonic").pitch_classes == (0, 3, 5, 8, 10)
    assert scales.scale("A major", "chromatic").label == "chromatic"
    assert scales.scale("Bb major").label == "A# major"
    custom = scales.custom("F, Ab Bb c")
    assert custom.label == "custom: C F G# A#" and custom.pitch_classes == (0, 5, 8, 10)
    assert scales.custom(["F", "F", "G#"]).pitch_classes == (5, 8)
    assert (scales.parse_note("G#3"), scales.parse_note("C-1"), scales.parse_note("Bb4")) == (
        56,
        0,
        70,
    )
    assert [scales.note_name(m) for m in (56, 60.4, 59.6, 0)] == ["G#3", "C4", "C4", "C-1"]
    for bad in ("H minor", "F lydianish", ""):
        with pytest.raises(ScaleError):
            scales.scale(bad)
    for bad in ("G#", "X3", ""):
        with pytest.raises(ScaleError):
            scales.parse_note(bad)
    with pytest.raises(ScaleError, match="at least one note"):
        scales.custom("")


def test_nearest_scale_note() -> None:
    minor = scales.scale("F minor")
    sung = [53.2, 53.9, 54.4, 54.6, 59.2, 66.0, 47.3, 100.6]
    assert minor.nearest(sung).tolist() == [53, 53, 55, 55, 60, 65, 48, 101]
    assert minor.nearest(54.0)[0] == 53, "exactly between two scale notes, the lower wins"
    assert scales.scale("C major", "chromatic").nearest([60.4, 60.6]).tolist() == [60, 61]
    assert minor.contains(53.3) and not minor.contains(54.0)
    assert scales.custom("C").nearest([55.0, 67.0]).tolist() == [60, 72]


def test_pitch_is_tracked_to_a_cent(vocal) -> None:
    _, truth, pitch, _ = vocal
    assert pitch.frame_s == 0.005 and pitch.sample_rate == SR
    assert 0.8 < pitch.voiced_share < 0.9, "the fixture is voiced for 6.4 of its 7.95 seconds"
    for note in truth.notes:
        middle = (pitch.times_s > note.start_s + 0.4) & (pitch.times_s < note.start_s + 0.75)
        measured = float(np.mean(pitch.midi[middle]))
        # Over two vibrato cycles the vibrato averages out; what is left is the note.
        assert measured == pytest.approx(note.midi + note.cents / 100, abs=0.02)


def test_notes_are_found_where_they_were_sung(vocal) -> None:
    _, truth, pitch, notes = vocal
    assert len(notes) == len(truth.notes) == 8
    for found, sung in zip(notes, truth.notes, strict=True):
        assert found.start_s() == pytest.approx(sung.start_s, abs=0.02)
        assert found.end_s() == pytest.approx(sung.start_s + sung.duration_s, abs=0.04)
        assert found.center == pytest.approx(sung.midi + sung.cents / 100, abs=0.01), (
            "within a cent"
        )
        assert 10 < found.spread_cents < 25, "the 20-cent vibrato"
    assert [n.index for n in notes] == list(range(8))
    assert [n.name for n in notes] == ["F3", "G#3", "C4", "A#3", "G#3", "G3", "F3", "C4"]


def test_slides_between_notes_belong_to_no_note(legato) -> None:
    _, truth, pitch, notes = legato
    assert len(pitch_track.voiced_runs(pitch.voiced)) == 1, "one unbroken phrase"
    assert [n.name for n in notes] == ["F3", "G#3", "C4", "A#3"]
    for found, sung in zip(notes, truth.notes, strict=True):
        assert found.center == pytest.approx(sung.midi + sung.cents / 100, abs=0.03)
        assert found.start_s() >= sung.start_s - 0.03 and found.end_s() <= sung.start_s + 0.75
    gaps = [b.start - a.end for a, b in zip(notes, notes[1:], strict=False)]
    assert all(10 <= gap <= 30 for gap in gaps), "the 120 ms glides, at 5 ms a frame"


def test_silence_and_noise_give_no_notes() -> None:
    quiet = pitch_track.track(np.zeros((SR, 2)), SR)
    assert not quiet.voiced.any() and pitch_track.segment(quiet) == [] and quiet.voiced_share == 0.0
    noise = pitch_track.track(fixtures.pink_noise(2.0), SR)
    assert len(pitch_track.segment(noise)) <= 1


def test_sustained_pitch_ignores_vibrato_and_follows_drift() -> None:
    t = np.arange(400) * 0.005
    wobble = 0.2 * np.sin(2 * np.pi * 5.5 * t)
    level = pitch_track.sustained(0.3 + wobble, 70)
    assert np.abs(level - 0.3).max() < 0.03, "a 20-cent vibrato leaves under 3 cents"
    drift = pitch_track.sustained(0.2 * t + wobble, 70)
    assert np.abs(drift - 0.2 * t)[40:-40].max() < 0.03
    assert pitch_track.sustained(np.array([1.0, 3.0]), 70).tolist() == [2.0, 2.0]


def test_key_is_read_from_the_sung_notes(vocal, legato) -> None:
    _, truth, _, notes = vocal
    key, confidence, others = scales.detect_key(
        [n.center for n in notes], [float(n.end - n.start) for n in notes]
    )
    assert key == truth.key == "F minor" and confidence > 0.5 and len(others) == 2
    assert scales.detect_key([60.0, 62.0], [1.0, 1.0]) == ("C major", 0.0, []), (
        "two notes: no guess"
    )
    assert scales.detect_key([], []) == ("C major", 0.0, [])


def test_pitch_report_measures_against_the_scale(vocal) -> None:
    audio, truth, pitch, _ = vocal
    report = pitch_report.analyze(audio, SR, "F minor", pitch=pitch)
    assert (
        report.key == "F minor" and report.key_source == "given" and report.key_confidence is None
    )
    assert [round(n.cents) for n in report.notes] == [0, 0, 30, 0, 0, 30, 0, 0]
    assert [n.target_name for n in report.notes] == [
        "F3",
        "G#3",
        "C4",
        "A#3",
        "G#3",
        "G3",
        "F3",
        "C4",
    ]
    assert report.in_tune_share == pytest.approx(0.75, abs=0.01), "two of eight equal notes are out"
    assert report.mean_cents == pytest.approx(
        7.5, abs=0.3
    ) and report.mean_abs_cents == pytest.approx(7.6, abs=0.3)
    assert report.out_of_key == [] and all(n.in_key for n in report.notes)
    assert report.flags == [], "7 cents sharp on average is not worth a flag"
    assert report.duration_s == pytest.approx(7.95, abs=0.01) and report.curve_step_s == 0.01
    assert len(report.curve) == pytest.approx(795, abs=2) and report.curve[0] is None
    assert report.headline() == (
        "8 notes in F minor, 75% of the time within ±20 cents, 8 cents off on average"
    )
    assert pitch_report.PitchReport.from_dict(report.to_dict()) == report


def test_each_gap_starts_a_phrase_and_the_worst_are_listed(vocal) -> None:
    audio, _, pitch, _ = vocal
    report = pitch_report.analyze(audio, SR, "F minor", pitch=pitch)
    # The fixture's notes are 150 ms apart, closer than a breath: it is one phrase.
    assert report.worst_phrases == [], "one phrase, 8 cents off on average: not worth listing"
    sharp = pitch_report.analyze(audio, SR, "F minor", "minor pentatonic", pitch=pitch)
    assert sharp.scale == "F minor pentatonic" and sharp.out_of_key == [5], "G is not in it"
    (phrase,) = sharp.worst_phrases
    assert phrase.notes == 8 and phrase.worst_note == 5 and phrase.mean_abs_cents > 10
    assert sharp.notes[5].target_name in ("F3", "G#3") and abs(sharp.notes[5].cents) > 60
    flagged = {f.code: f for f in sharp.flags}
    assert (
        "out_of_key_notes" in flagged
        and "first: G3 at 5.0 s" in flagged["out_of_key_notes"].message
    )


def test_report_guesses_the_key_and_says_so(vocal) -> None:
    audio, _, pitch, _ = vocal
    report = pitch_report.analyze(audio, SR, pitch=pitch)
    assert (report.key, report.key_source) == ("F minor", "detected")
    assert report.key_confidence is not None and report.key_confidence > 0.5
    custom = pitch_report.analyze(audio, SR, "F minor", custom_notes=("F", "G#", "C"), pitch=pitch)
    assert custom.scale == "custom: C F G#" and custom.out_of_key == [3, 5]


def test_report_flags_what_it_cannot_trust(vocal) -> None:
    audio, _, pitch, _ = vocal
    separated = pitch_report.analyze(audio, SR, "F minor", separated=True, pitch=pitch)
    assert [f.code for f in separated.flags] == ["separated_vocal"]
    nothing = pitch_report.analyze(np.zeros((SR * 2, 1)), SR, "F minor")
    assert nothing.notes == [] and nothing.in_tune_share is None
    assert [f.code for f in nothing.flags] == ["vocal_no_pitch"]
    assert nothing.headline() == "no sung notes found"
    flat = pitch_report.analyze(fixtures.synthetic_vocal(off_cents=-45.0)[0], SR, "F minor")
    assert flat.mean_cents == pytest.approx(-11.3, abs=0.5)
    assert [f.code for f in flat.flags] == ["vocal_flat"]
