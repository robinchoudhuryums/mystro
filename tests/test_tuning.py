from __future__ import annotations

import shutil
from dataclasses import replace

import numpy as np
import pytest
from scipy import ndimage, signal

from masterbench import fixtures
from masterbench.analysis import loudness
from masterbench.vocal import engines, pitch_track, tuning
from masterbench.vocal.engines import EngineError
from masterbench.vocal.pitch_track import PitchTrack
from masterbench.vocal.tuning import NoteEdit, TuningError, TuningSettings

SR = fixtures.SR
NATURAL = TuningSettings(key="F minor", retune_ms=100, humanize=1.0, amount=0.9)
HARD = TuningSettings(key="F minor", retune_ms=0, humanize=0.0, amount=1.0, engine="world")
SCALE = np.array([53, 55, 56, 58, 60, 61, 63, 65], dtype=float)
ENGINES = [
    "world",
    pytest.param(
        "rubberband",
        marks=pytest.mark.skipif(shutil.which("rubberband") is None, reason="no rubberband"),
    ),
]


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


@pytest.fixture(scope="module", params=ENGINES)
def natural(request, vocal):
    """The stepwise vocal under Natural, with each engine, and its pitch tracked again."""
    audio, _, pitch, _ = vocal
    result = tuning.tune(audio, SR, replace(NATURAL, engine=request.param), pitch=pitch)
    return result, pitch_track.track(result.audio, SR)


def held(pitch: PitchTrack, note, after_s: float = 0.15) -> np.ndarray:
    """A note's pitch once the retune has arrived, to a little before it ends."""
    frames = pitch.midi[note.start + int(after_s / 0.005) : note.end - 6]
    return frames[np.isfinite(frames)]


def level(frames: np.ndarray) -> float:
    """Where a note sits, with its vibrato taken out."""
    return float(np.mean(pitch_track.sustained(frames, 70)))


def movement(frames: np.ndarray) -> float:
    """RMS of the pitch around its own sustained level, in cents: the vibrato."""
    return float(np.std(frames - pitch_track.sustained(frames, 70)) * 100)


def test_natural_brings_the_sharp_notes_within_five_cents(natural, vocal) -> None:
    """Phase 8 criterion: +30-cent notes within ±5 cents under Natural, vibrato kept."""
    _, truth, pitch, notes = vocal
    result, after = natural
    for note, sung in zip(notes, truth.notes, strict=True):
        was = level(held(pitch, note)) - sung.midi
        now = level(held(after, note)) - sung.midi
        if sung.cents:
            assert was * 100 == pytest.approx(30.0, abs=1.0)
            assert abs(now * 100) < 5.0, "the criterion"
            assert now * 100 == pytest.approx(3.0, abs=1.2), "90% of 30 cents leaves 3"
        else:
            assert abs(now * 100) < 1.0, "notes that were in tune stay where they were"
        kept = movement(held(after, note)) / movement(held(pitch, note))
        assert kept == pytest.approx(1.0, abs=0.12), "the vibrato is still there, at its depth"
    assert result.plan.targets == [float(n.midi) for n in truth.notes]


def test_output_has_the_same_length_start_and_level(natural, vocal) -> None:
    """Phase 8 criterion: length and alignment. Only corrected, voiced audio is replaced."""
    audio, truth, _, _ = vocal
    result, _ = natural
    tuned = result.audio
    assert tuned.shape == audio.shape and tuned.dtype == audio.dtype
    changed = tuned != audio
    for index, sung in enumerate(truth.notes):
        a, b = int((sung.start_s - 0.05) * SR), int((sung.start_s + sung.duration_s + 0.05) * SR)
        if not sung.cents:
            continue
        assert changed[a:b].mean() > 0.9, f"note {index + 1} was resynthesized"
    untouched = np.ones(len(audio), dtype=bool)
    hop, room = SR * 0.005, int(0.03 * SR)
    for first, last in pitch_track.voiced_runs(result.pitch.voiced):
        untouched[max(0, int(first * hop) - room) : int(last * hop) + room] = False
    assert untouched.mean() > 0.08, "the gaps between notes, and the silence at both ends"
    assert not changed[untouched].any(), "outside the sung stretches every sample is the original's"

    envelope = [np.sqrt(np.maximum(ndimage.uniform_filter1d(x**2, 64), 0)) for x in (audio, tuned)]
    for sung in truth.notes:
        if not sung.cents:
            continue
        a, b = int((sung.start_s - 0.04) * SR), int((sung.start_s + sung.duration_s + 0.04) * SR)
        starts = [a + int(np.argmax(e[a:b] > 0.1 * e[a:b].max())) for e in envelope]
        assert abs(starts[1] - starts[0]) < 0.003 * SR, "a corrected note starts within 3 ms"
    levels = [loudness.integrated(x[:, None], SR) for x in (audio, tuned)]
    assert levels[1] == pytest.approx(levels[0], abs=0.1), "tuning does not move the level"


@pytest.mark.parametrize("engine", ENGINES)
def test_hard_tune_flattens_notes_and_steps_between_them(engine: str, legato) -> None:
    """Phase 8 criterion: under Hard-tune notes snap and the slides become steps."""
    audio, truth, pitch, notes = legato
    result = tuning.tune(audio, SR, replace(HARD, engine=engine), pitch=pitch)
    after = pitch_track.track(result.audio, SR)
    for note, sung in zip(notes, truth.notes, strict=True):
        frames = held(after, note, 0.05)
        assert float(np.mean(frames)) == pytest.approx(sung.midi, abs=0.01), "on the note"
        assert movement(frames) < 0.15 * movement(held(pitch, note, 0.05)), "no vibrato left"

    def on_scale(tracked: PitchTrack) -> float:
        """Share of the first slide (F3 to G#3) spent within 15 cents of a scale note."""
        first = int((truth.notes[0].start_s + 0.68) / 0.005)
        last = int((truth.notes[1].start_s + 0.02) / 0.005)
        midi = tracked.midi[first:last]
        midi = midi[np.isfinite(midi)]
        return float(np.mean(np.min(np.abs(midi[:, None] - SCALE[None, :]), axis=1) < 0.15))

    assert on_scale(pitch) < 0.4, "sung, the slide spends most of its time between notes"
    assert on_scale(after) > 0.75, "tuned, it sits on scale notes and jumps between them"


def test_note_hold_takes_one_step_where_the_staircase_takes_two(legato) -> None:
    audio, truth, pitch, notes = legato
    slide = slice(notes[0].end, notes[1].start)
    stairs = tuning.plan(pitch, notes, HARD)
    held_plan = tuning.plan(pitch, notes, replace(HARD, note_hold=True))
    landed = np.round(pitch.midi[slide] + stairs.correction[slide], 3)
    assert sorted(set(landed)) == [53.0, 55.0, 56.0], "F3, through G3, to G#3"
    landed = np.round(pitch.midi[slide] + held_plan.correction[slide], 3)
    assert sorted(set(landed)) == [53.0, 56.0], "straight from F3 to G#3"
    assert np.count_nonzero(np.diff(landed)) == 1


def test_plan_follows_amount_humanize_and_retune_speed(vocal) -> None:
    _, _, pitch, notes = vocal
    sharp = notes[2]
    middle = slice(sharp.start + 60, sharp.end - 20)

    def correction(**changes) -> np.ndarray:
        return tuning.plan(pitch, notes, replace(NATURAL, **changes)).correction * 100

    assert not correction(amount=0.0).any()
    assert np.median(correction(amount=1.0)[middle]) == pytest.approx(-30.0, abs=0.7)
    assert np.median(correction(amount=0.5)[middle]) == pytest.approx(-15.0, abs=0.4)
    assert np.ptp(correction()[middle]) < 2.0, "humanize at 100%: the whole note moves as one"
    flat = correction(humanize=0.0, retune_ms=0, amount=1.0)[middle]
    assert np.ptp(flat) > 30, "humanize at 0: the correction mirrors the vibrato"
    assert np.allclose((pitch.midi[middle] * 100 + flat) / 100, 60.0, atol=0.001)

    slow, fast, instant = (correction(retune_ms=ms)[sharp.frames] for ms in (200, 50, 0))
    target = -27.0
    assert abs(instant[2]) > 0.8 * abs(target), "at 0 ms the correction is there at once"
    assert fast[10] / target == pytest.approx(0.9, abs=0.08), "90% of the way after 50 ms"
    assert slow[40] / target == pytest.approx(0.9, abs=0.08), "90% of the way after 200 ms"
    assert abs(slow[10]) < abs(fast[10]) < abs(instant[10])
    assert not correction()[~pitch.voiced].any(), "nothing is asked of unvoiced frames"


def test_notes_can_be_left_alone_or_sent_elsewhere(vocal) -> None:
    audio, truth, pitch, notes = vocal
    edits = {2: NoteEdit(bypass=True), 5: NoteEdit(target=56.0)}
    planned = tuning.plan(pitch, notes, NATURAL, edits)
    assert planned.targets[2] is None and planned.targets[5] == 56.0
    assert not planned.correction[notes[2].frames].any()
    moved = np.median(planned.correction[notes[5].start + 60 : notes[5].end])
    assert moved == pytest.approx(0.9 * (56.0 - 55.3), abs=0.01), "G3 sharp, sent up to G#3"

    result = tuning.tune(audio, SR, replace(NATURAL, engine="world"), edits, pitch)
    sung = truth.notes[2]
    a, b = int((sung.start_s - 0.05) * SR), int((sung.start_s + sung.duration_s + 0.05) * SR)
    assert np.array_equal(result.audio[a:b], audio[a:b]), "a bypassed note is the original"
    after = pitch_track.track(result.audio, SR)
    assert float(np.mean(held(after, notes[5]))) == pytest.approx(55.93, abs=0.02)

    with pytest.raises(TuningError, match="no note number 12; this vocal has 8 notes"):
        tuning.plan(pitch, notes, NATURAL, {11: NoteEdit(bypass=True)})


def test_nothing_to_correct_returns_the_input(vocal) -> None:
    audio, _, pitch, notes = vocal
    for settings in (replace(NATURAL, amount=0.0), replace(NATURAL, engine="world", amount=0.0)):
        assert np.array_equal(tuning.tune(audio, SR, settings, pitch=pitch).audio, audio)
    everything = {note.index: NoteEdit(bypass=True) for note in notes}
    assert np.array_equal(tuning.tune(audio, SR, NATURAL, everything, pitch).audio, audio)
    silence = np.zeros((SR, 2))
    assert np.array_equal(tuning.tune(silence, SR, NATURAL).audio, silence)


def test_tracking_slips_are_not_chased(vocal) -> None:
    _, _, pitch, notes = vocal
    slipped = pitch.f0_hz.copy()
    frame = notes[0].start + 80
    slipped[frame] *= 2.0
    wrong = PitchTrack(SR, slipped)
    found = pitch_track.segment(wrong)
    assert len(found) == len(notes), "one frame an octave up does not split the note"
    clean = tuning.plan(pitch, notes, HARD).correction
    planned = tuning.plan(wrong, found, HARD).correction
    assert abs(planned[frame]) < 0.1, "the slipped frame is not dragged down an octave"
    nearby = np.r_[frame - 40 : frame - 2, frame + 3 : frame + 40]
    assert np.abs(clean[nearby]).max() > 0.15, "around it the vibrato is being flattened"
    assert np.allclose(planned[nearby], clean[nearby], atol=0.03), "and still is, the same way"


def test_formant_shift_moves_the_tone_and_needs_world(vocal) -> None:
    audio, _, pitch, _ = vocal

    def centroid(x: np.ndarray) -> float:
        freqs, power = signal.welch(x, SR, nperseg=4096)
        band = (freqs > 300) & (freqs < 5000)
        return float(np.sum(freqs[band] * power[band]) / np.sum(power[band]))

    plain = tuning.tune(audio, SR, HARD, pitch=pitch).audio
    raised = tuning.tune(audio, SR, replace(HARD, formant_shift=4.0), pitch=pitch).audio
    assert centroid(raised) > 1.15 * centroid(plain), "formants a major third up: brighter"
    after = pitch_track.track(raised, SR)
    assert float(np.nanmedian(after.midi[60:200])) == pytest.approx(53.0, abs=0.05), "same pitch"
    if shutil.which("rubberband"):
        with pytest.raises(EngineError, match="formant shift needs the WORLD engine"):
            tuning.tune(
                audio, SR, replace(HARD, engine="rubberband", formant_shift=4.0), pitch=pitch
            )


@pytest.mark.parametrize("engine", ENGINES)
def test_formants_stay_put_unless_told_to_travel(engine: str, vocal) -> None:
    """Formant preservation is required: four semitones up must still be the same vowel."""
    audio, truth, pitch, notes = vocal
    up = {note.index: NoteEdit(target=round(note.center) + 4.0) for note in notes}
    wanted = replace(HARD, engine=engine)
    kept = tuning.tune(audio, SR, wanted, up, pitch).audio
    carried = tuning.tune(audio, SR, replace(wanted, preserve_formants=False), up, pitch).audio

    def third_formant_db(x: np.ndarray) -> float:
        """Level around the fixture's 2600 Hz formant against where a third up would put it."""
        freqs, power = signal.welch(x, SR, nperseg=4096)
        home = power[(freqs > 2350) & (freqs < 2850)].mean()
        moved = power[(freqs > 3000) & (freqs < 3600)].mean()
        return float(10 * np.log10(home / moved))

    assert third_formant_db(audio) == pytest.approx(7.3, abs=0.5)
    assert third_formant_db(kept) > 5, "kept: still at 2600 Hz (WORLD reads 8.5, Rubber Band 6.1)"
    assert third_formant_db(carried) < -4, "carried: now around 3300 Hz (-6.1 and -5.2)"
    for version in (kept, carried):
        after = pitch_track.track(version, SR)
        sung = float(np.mean(after.midi[notes[0].start + 40 : notes[0].end - 10]))
        assert sung == pytest.approx(truth.notes[0].midi + 4.0, abs=0.05), "both are a third up"


def test_stereo_vocals_keep_their_channels(vocal) -> None:
    audio, _, pitch, _ = vocal
    wide = np.stack([audio, 0.5 * audio], axis=1)
    tuned = tuning.tune(wide, SR, replace(NATURAL, engine="world"), pitch=pitch).audio
    assert tuned.shape == wide.shape
    changed = np.any(tuned != wide, axis=1)
    ratio = np.sqrt(np.mean(tuned[changed, 1] ** 2) / np.mean(tuned[changed, 0] ** 2))
    assert ratio == pytest.approx(0.5, abs=0.02), "each channel keeps its own level"


@pytest.mark.skipif(shutil.which("rubberband") is None, reason="no rubberband")
def test_rubberband_latency_is_measured_not_assumed() -> None:
    measured = engines.rubberband_latency(SR)
    assert 1000 < measured < 10000, "about 3860 samples with Rubber Band 3.3 at 44.1 kHz"
    assert engines.rubberband_latency(SR) == measured


def test_missing_rubberband_is_explained(vocal, monkeypatch: pytest.MonkeyPatch) -> None:
    audio, _, pitch, _ = vocal
    engines.rubberband_latency.cache_clear()
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(EngineError, match="brew install rubberband"):
        tuning.tune(audio, SR, NATURAL, pitch=pitch)


def test_settings_are_checked_and_hash_the_same_however_written() -> None:
    for bad in (
        {"amount": 1.2},
        {"humanize": -0.1},
        {"retune_ms": 5000},
        {"engine": "melodyne"},
        {"formant_shift": 24},
        {"key": "H minor"},
        {"scale": "wrong"},
    ):
        with pytest.raises(ValueError):
            TuningSettings(**{"key": "F minor", **bad})
    whole = TuningSettings(key="F minor", retune_ms=100, amount=1, humanize=1)
    assert tuning.settings_hash(whole) == tuning.settings_hash(replace(whole, retune_ms=100.0))
    assert tuning.settings_hash(whole) != tuning.settings_hash(replace(whole, amount=0.9))
    assert tuning.settings_hash(whole) != tuning.settings_hash(whole, {3: NoteEdit(bypass=True)})
    assert tuning.settings_hash(whole, {}) == tuning.settings_hash(whole, None)
    assert TuningSettings.from_dict(whole.to_dict()) == whole
    custom = TuningSettings(key="F minor", custom_notes=("F", "Ab"))
    assert custom.resolve_scale().label == "custom: F G#"
    assert TuningSettings.from_dict(custom.to_dict()) == custom
    assert TuningSettings(key="F minor", scale="dorian").resolve_scale().label == "F dorian"
