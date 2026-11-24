from __future__ import annotations

from pathlib import Path

import pytest

from conftest import FakeAPI
from masterbench import blind, fixtures, index, progress
from masterbench.config import Paths, Settings, Thresholds
from masterbench.master import chain, pipeline
from masterbench.master.chain import ChainPreset, Module
from masterbench.notes import mix_notes, trends
from masterbench.notes.mix_notes import NotesError
from masterbench.progress import TrackPoint
from masterbench.references import library

SR = fixtures.SR
THIN = ChainPreset("thin", "", (Module("low_shelf", {"freq_hz": 70.0, "gain_db": -8.0}),))
TREND_REPLY = {
    "summary": "Your sub sits low on every recent track.",
    "habits": [
        {
            "title": "Thin sub",
            "evidence": "Sub is under the reference on 5 of the last 5 tracks.",
            "why_it_matters": "The 808 loses weight on full-range systems.",
            "exercise": "Match the sub level of a reference with Channel EQ on one track.",
            "how_to_tell_it_worked": "The sub delta is within 1.5 dB on the next analysis.",
        }
    ],
    "caveats": ["Measured against a single reference."],
}


def point(title: str, flags: list[str] | None = None, **deltas: float) -> TrackPoint:
    return TrackPoint(
        slug=title.lower(),
        title=title,
        genre="hip-hop",
        created_at="2026-11-01T00:00:00.000Z",
        run_id="r",
        lufs_i=-18.0,
        true_peak_dbtp=-3.0,
        lra=5.0,
        crest_db=12.0,
        correlation=0.9,
        band_deltas={"sub": 0.0, "bass": 0.0, "mids": 0.0, **deltas},
        flags=flags or [],
        delta_vs="target_curve",
    )


def analyze_five(
    paths: Paths, settings: Settings, folder: Path, preset: ChainPreset | None
) -> None:
    """Five different loops, each put through the same preset before it is analyzed."""
    for number in range(5):
        loop, _ = fixtures.drum_808_loop(
            seed=20 + number, hat_gain=0.10 + 0.02 * number, detune=number < 2
        )
        audio = chain.run(preset, loop, SR) if preset else loop
        file = fixtures.write_wav(folder / f"Track {number}.wav", audio)
        pipeline.analyze(file, paths, settings, genre="hip-hop")


@pytest.fixture
def seeded(paths: Paths, settings: Settings, tmp_path: Path, fx: dict[str, Path]) -> Paths:
    """Five different tracks that all share one deliberate fault: a thin sub."""
    library.add(paths, fx["reference"], "hip-hop", settings)
    analyze_five(paths, settings, tmp_path, THIN)
    return paths


def test_without_the_seeded_fault_there_is_no_sub_habit(
    paths: Paths, settings: Settings, tmp_path: Path, fx: dict[str, Path]
) -> None:
    """The control for the test below: the same five loops, left alone."""
    library.add(paths, fx["reference"], "hip-hop", settings)
    analyze_five(paths, settings, tmp_path, None)
    with index.open_index(paths) as conn:
        points = progress.track_points(conn)
    assert all(abs(p.band_deltas["sub"]) < 1.0 for p in points)
    found = {h.code for h in progress.habits(points, settings.thresholds)}
    assert not any(code.startswith("band_sub") for code in found)
    assert "bass_detuned" not in found, "two of five"


def test_a_bias_seeded_across_tracks_is_found(seeded: Paths, settings: Settings) -> None:
    """Phase 7 criterion: habit detection finds a deliberate bias seeded across fixture tracks."""
    with index.open_index(seeded) as conn:
        points = progress.track_points(conn)
    assert len(points) == 5 and all(p.delta_vs == "reference" for p in points)
    assert all(p.band_deltas["sub"] < -3 for p in points), "the seeded fault is in every track"

    found = {h.code: h for h in progress.habits(points, settings.thresholds)}
    thin = found["band_sub_under"]
    assert (
        (thin.tracks, thin.considered) == (5, 5)
        and thin.mean_value is not None
        and thin.mean_value < -3
    )
    assert thin.statement.startswith(
        "Sub (20 Hz to 60 Hz) is under the reference on 5 of the last 5 tracks, by "
    )
    assert thin.confidence == pytest.approx(0.57, abs=0.01) and len(thin.examples) == 3
    assert "bass_detuned" not in found, "only two of the five were detuned: not a habit"
    # Levels are relative to loudness, and the sub carries much of this loop's loudness, so
    # taking it away also lifts every band above it by about 3 dB against the reference.
    assert found["band_mids_over"].mean_value == pytest.approx(5.2, abs=0.5)
    assert points[0].bass_in_tune_share is not None and points[0].bass_in_tune_share < 1.0
    assert points[4].bass_in_tune_share == 1.0


def test_confidence_grows_with_the_number_of_tracks() -> None:
    assert progress.wilson_lower(3, 3) == pytest.approx(0.44, abs=0.01)
    assert progress.wilson_lower(8, 8) == pytest.approx(0.68, abs=0.01)
    assert progress.wilson_lower(7, 8) == pytest.approx(0.53, abs=0.01)
    assert progress.wilson_lower(30, 30) > 0.88 and progress.wilson_lower(0, 0) == 0.0
    assert progress.wilson_lower(8, 8) > progress.wilson_lower(3, 3) > progress.wilson_lower(2, 3)


def test_habits_need_enough_tracks_and_consistency() -> None:
    limits = Thresholds()
    two = [point("A", sub=-4.0), point("B", sub=-4.0)]
    assert progress.habits(two, limits) == []
    mixed = [point(n, sub=d) for n, d in zip("ABCDE", (-4.0, 4.0, -4.0, 0.0, 0.5), strict=True)]
    assert progress.habits(mixed, limits) == [], "two of five is not a habit"
    most = [
        point(n, sub=d, mids=2.0)
        for n, d in zip("ABCDE", (-4.0, -3.0, -2.0, 0.0, -5.0), strict=True)
    ]
    found = {h.code: h for h in progress.habits(most, limits)}
    assert found["band_sub_under"].tracks == 4 and found["band_mids_over"].tracks == 5
    assert found["band_mids_over"].statement.startswith(
        "Mids (500 Hz to 2 kHz) is over your genre curve"
    )
    assert list(found) == ["band_mids_over", "band_sub_under"], "most certain first"


def test_recurring_flags_become_habits_and_only_recent_tracks_count() -> None:
    limits = Thresholds()
    old = [point(f"Old {n}", ["clipping"]) for n in range(6)]
    recent = [point(f"New {n}", ["weak_hook_contrast", "bass_detuned"]) for n in range(8)]
    found = {h.code: h for h in progress.habits([*old, *recent], limits)}
    assert set(found) == {"weak_hook_contrast", "bass_detuned"}, "clipping is in the past"
    assert found["weak_hook_contrast"].statement == (
        "On 8 of the last 8 tracks, the hook barely lifts over the verse."
    )
    assert "clipping" in {h.code for h in progress.habits(old, limits)}


def test_blind_test_wins_are_tallied(seeded: Paths) -> None:
    candidates = [{"id": "mix", "label": "Mix"}, {"id": "m", "label": "Master -14 LUFS (11-24)"}]
    for ranking in (["B", "A"], ["B", "A"], ["A", "B"]):
        blind.record(seeded, "track-0", candidates, ranking, {"A": "mix", "B": "m"})
    with index.open_index(seeded) as conn:
        index.sync_blind_tests(conn, seeded)
        results = progress.blind_results(conn)
    assert progress.blind_summary(results) == {"Master": 2, "Mix": 1}
    assert progress.blind_summary([]) == {}


def test_trend_summary_sends_numbers_and_stores_the_answer(
    seeded: Paths, settings: Settings
) -> None:
    with index.open_index(seeded) as conn:
        payload = trends.gather(conn, settings)
    text = mix_notes.payload_json(payload)
    assert str(seeded.root) not in text and ".wav" not in text
    assert payload["tracks_considered"] == 5 and len(payload["tracks"]) == 5
    assert "band_sub_under" in [h["code"] for h in payload["habits"]]
    assert payload["tracks"][0]["compared_with"] == "reference" and "blind_tests" not in payload

    api = FakeAPI(TREND_REPLY)
    summary = trends.generate(payload, seeded, settings, client=api.client())
    body = api.body()
    assert "top three habits" in body["system"] and "confidence" in body["system"]
    assert body["output_config"]["format"]["schema"] == trends.TRENDS_SCHEMA
    assert summary.body == TREND_REPLY and summary.cost_usd > 0 and not summary.cached
    assert mix_notes.spent_this_month(seeded) == summary.cost_usd, "trends count toward the cap"
    assert trends.latest(seeded) == summary
    rendered = trends.render(summary)
    assert "### 1. Thin sub" in rendered and "**You will know it worked when:**" in rendered

    again = trends.generate(payload, seeded, settings, client=api.client())
    assert again.cached and len(api.requests) == 1


def test_trend_summary_respects_the_cap_and_needs_tracks(
    seeded: Paths, settings: Settings, paths: Paths
) -> None:
    from dataclasses import replace

    with index.open_index(seeded) as conn:
        payload = trends.gather(conn, settings)
    api = FakeAPI(TREND_REPLY)
    with pytest.raises(NotesError, match="would pass the monthly cap"):
        trends.generate(
            payload, seeded, replace(settings, notes_monthly_cap_usd=0.001), client=api.client()
        )
    with pytest.raises(NotesError, match="no analyzed tracks"):
        trends.generate({**payload, "tracks": []}, seeded, settings, client=api.client())
    assert api.requests == [] and trends.latest(seeded) is None
