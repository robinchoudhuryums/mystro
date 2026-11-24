from __future__ import annotations

import numpy as np
import pytest

from masterbench import fixtures
from masterbench.analysis import musical

SR = fixtures.SR


@pytest.fixture(scope="module")
def loop() -> musical.Musical:
    found = musical.analyze(fixtures.drum_808_loop()[0], SR)
    assert found is not None
    return found


def test_key_is_right_on_the_fixtures(loop: musical.Musical) -> None:
    assert loop.key == "F minor" and loop.key_confidence > 0.8
    song = musical.analyze(fixtures.arranged_song()[0], SR)
    reference = musical.analyze(fixtures.reference_master(), SR)
    vocal = musical.analyze(fixtures.synthetic_vocal()[0][:, None], SR)
    assert song and reference and vocal
    assert song.key == reference.key == vocal.key == "F minor"
    assert abs(loop.tuning_cents) < 10


def test_half_time_tempo_is_folded_into_the_usual_range(loop: musical.Musical) -> None:
    assert loop.tempo_detected_bpm == pytest.approx(70.0, abs=0.5), "the tracker locks to half time"
    assert loop.tempo_bpm == pytest.approx(140.0, abs=0.5)
    slow = musical.analyze(fixtures.drum_808_loop()[0], SR, tempo_range=(60.0, 120.0))
    assert slow is not None and slow.tempo_bpm == pytest.approx(70.0, abs=0.5)
    assert 0 <= loop.beat_offset_s < loop.beat_s


@pytest.mark.parametrize(
    ("bpm", "expected"),
    [(70.0, 140.0), (140.0, 140.0), (280.0, 140.0), (87.0, 87.0), (174.0, 87.0)],
)
def test_fold_tempo(bpm: float, expected: float) -> None:
    assert musical.fold_tempo(bpm, 80.0, 160.0) == expected


def test_sample_rate_does_not_change_the_answer(loop: musical.Musical) -> None:
    at_48k = musical.analyze(fixtures.drum_808_loop(sr=48000)[0], 48000)
    assert at_48k is not None and at_48k.key == loop.key
    assert at_48k.tempo_bpm == pytest.approx(loop.tempo_bpm, abs=0.5)


def test_noise_and_silence_do_not_get_a_confident_key() -> None:
    noise = musical.analyze(fixtures.pink_noise(), SR)
    assert noise is not None and noise.key_confidence < 0.6
    assert musical.analyze(np.zeros((SR * 3, 2)), SR) is None
    assert musical.analyze(np.zeros((100, 2)), SR) is None


def test_keys_parse_and_give_their_scale() -> None:
    assert musical.parse_key("F minor") == (5, "minor")
    assert musical.scale_pitch_classes("C major") == frozenset({0, 2, 4, 5, 7, 9, 11})
    assert 8 in musical.scale_pitch_classes("F minor") and 9 not in musical.scale_pitch_classes(
        "F minor"
    )
    for bad in ("H minor", "F", "f minor", "F dorian"):
        with pytest.raises(ValueError, match="not a key"):
            musical.parse_key(bad)
