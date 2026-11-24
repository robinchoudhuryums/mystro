from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest

from masterbench import fixtures
from masterbench.analysis import sections
from masterbench.analysis.sections import Sections
from masterbench.config import Thresholds

SR = fixtures.SR
KINDS = ["intro", "verse", "hook", "verse", "hook", "outro"]


@pytest.fixture(scope="module")
def strong() -> Sections:
    """Verses 6 dB under the hooks."""
    return sections.detect(fixtures.arranged_song(verse_db=-6.0)[0], SR, 140.0)


@pytest.fixture(scope="module")
def weak() -> Sections:
    """Verses only 1 dB under the hooks: the hook barely lifts."""
    return sections.detect(fixtures.arranged_song(verse_db=-1.0)[0], SR, 140.0)


@pytest.mark.parametrize("which", ["strong", "weak"])
def test_sections_are_found_where_they_are(which: str, request: pytest.FixtureRequest) -> None:
    found: Sections = request.getfixturevalue(which)
    _, truth = fixtures.arranged_song()
    assert [item.kind for item in found.items] == KINDS
    assert [item.label for item in found.items][1:5] == ["verse 1", "hook 1", "verse 2", "hook 2"]
    for item, expected in zip(found.items, truth, strict=True):
        assert item.start_s == pytest.approx(expected.start_s, abs=0.75)
        assert item.end_s == pytest.approx(expected.end_s, abs=0.75)
    assert found.source == "auto"


def test_section_loudness_and_contrast(strong: Sections, weak: Sections) -> None:
    assert strong.contrast_lu == pytest.approx(6.3, abs=0.6)
    assert weak.contrast_lu == pytest.approx(1.3, abs=0.4)
    hook, verse, intro = strong.items[2], strong.items[1], strong.items[0]
    assert hook.lufs > verse.lufs > intro.lufs
    assert hook.bands is not None and hook.regions is not None and hook.correlation is not None
    assert set(hook.regions) == {"mud", "harsh"}


def test_loudest_thirty_seconds(strong: Sections) -> None:
    assert strong.loudest_end_s - strong.loudest_start_s == pytest.approx(30.0, abs=0.2)
    assert 20 <= strong.loudest_start_s <= 32, "it spans a hook, the verse between, and a hook"
    short = sections.detect(fixtures.drum_808_loop()[0], SR, 140.0)
    assert (short.loudest_start_s, short.loudest_end_s) == (0.0, pytest.approx(13.71, abs=0.01))


def test_a_loop_is_one_section_however_long() -> None:
    loop = fixtures.drum_808_loop()[0]
    for audio in (loop, np.tile(loop, (6, 1)), fixtures.pink_noise(40.0)):
        found = sections.detect(audio, SR, 140.0)
        assert [item.kind for item in found.items] == ["section"] and found.contrast_lu is None
    without_tempo = sections.detect(np.tile(loop, (6, 1)), SR)
    assert len(without_tempo.items) == 1


def test_weak_hook_is_flagged_and_a_strong_one_is_not(strong: Sections, weak: Sections) -> None:
    limits = Thresholds()
    assert sections.flags(strong, None, limits) == []
    (alone,) = sections.flags(weak, None, limits)
    assert alone.code == "weak_hook_contrast" and "It may not lift" in alone.message
    assert alone.start_s == weak.items[2].start_s

    (versus,) = sections.flags(sections.compare(weak, strong), strong, limits)
    assert "is only 1." in versus.message and "(reference: 6." in versus.message
    (house,) = sections.flags(weak, None, limits, genre="house")
    assert house.message.startswith("The drop is only") and "louder than the break" in house.message
    assert sections.flags(sections.compare(strong, strong), strong, limits) == []


def test_hook_balance_is_compared_with_the_reference_hook(strong: Sections) -> None:
    bassy = replace(
        strong,
        items=[
            replace(item, bands={**item.bands, "sub": item.bands["sub"] - 4.0})
            if item.bands
            else item
            for item in strong.items
        ],
    )
    compared = sections.compare(bassy, strong)
    assert compared.items[2].band_deltas is not None
    assert compared.items[2].band_deltas["sub"] == pytest.approx(-4.0, abs=0.1)
    codes = {f.code: f for f in sections.flags(compared, strong, Thresholds())}
    assert (
        "The hook has 4.0 dB less sub than the reference hook." in codes["hook_sub_under"].message
    )
    assert sections.compare(strong, None) is strong


def test_hand_set_sections_are_used_as_given() -> None:
    song, truth = fixtures.arranged_song()
    starts = [t.start_s for t in truth]
    found = sections.build(song, SR, starts, KINDS, source="manual")
    assert found.source == "manual" and [i.kind for i in found.items] == KINDS
    assert [i.start_s for i in found.items] == [round(s, 2) for s in starts]
    assert found.contrast_lu == pytest.approx(6.3, abs=0.6)
    assert Sections.from_dict(asdict(found)) == found


def test_markers_follow_a_bounce_that_starts_earlier_or_later() -> None:
    song, _ = fixtures.arranged_song()
    saved = list(sections.envelope(song, SR))
    later = np.concatenate([np.zeros((int(2.3 * SR), 2)), song])
    assert sections.align_offset(saved, sections.envelope(later, SR)) == pytest.approx(2.3, abs=0.1)
    earlier = song[int(1.5 * SR) :]
    assert sections.align_offset(saved, sections.envelope(earlier, SR)) == pytest.approx(
        -1.5, abs=0.1
    )
    assert sections.align_offset(saved, sections.envelope(song, SR)) == 0.0
    assert sections.align_offset([], sections.envelope(song, SR)) == 0.0
