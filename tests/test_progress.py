from __future__ import annotations

from pathlib import Path

import pytest

from masterbench import blind, index, progress
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.pipeline import MasterOptions
from masterbench.references.library import Reference

CANDIDATES = [
    {"id": "mix", "label": "Mix"},
    {"id": "master", "label": "Master"},
    {"id": "reference", "label": "Reference"},
]
REVEALED = {"A": "reference", "B": "mix", "C": "master"}


@pytest.fixture
def three_tracks(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> Paths:
    options = MasterOptions.from_settings(settings, "hip-hop/default")
    pipeline.analyze(fx["loop"], paths, settings, genre="hip-hop")
    pipeline.master(fx["loop"], options, paths, settings)
    pipeline.analyze(fx["pink_noise"], paths, settings, genre="hip-hop")
    pipeline.analyze(fx["sine_sweep"], paths, settings, genre="hip-hop")
    return paths


def test_one_point_per_track_from_its_latest_run(three_tracks: Paths) -> None:
    with index.open_index(three_tracks) as conn:
        points = progress.track_points(conn)
        latest_loop = index.recent_runs(conn, kind="master")[0]["id"]
    assert [p.slug for p in points] == ["loop", "pink-noise", "sine-sweep"]
    loop = points[0]
    assert loop.run_id == latest_loop and loop.lufs_i == pytest.approx(-19.7, abs=0.1)
    assert set(loop.band_deltas) == {
        "sub",
        "bass",
        "low_mids",
        "mids",
        "upper_mids",
        "presence",
        "air",
    }
    assert loop.correlation is not None and loop.correlation > 0.9
    assert points[1].band_deltas["sub"] < -5 and "band_sub_under" in points[1].flags


def test_flag_counts_rank_recurring_problems(three_tracks: Paths) -> None:
    with index.open_index(three_tracks) as conn:
        points = progress.track_points(conn)
    counts = {c.code: c for c in progress.flag_counts(points)}
    assert counts["band_sub_under"].tracks == 2
    assert counts["band_sub_under"].share == pytest.approx(2 / 3)
    ordered = [c.tracks for c in progress.flag_counts(points)]
    assert ordered == sorted(ordered, reverse=True)
    assert progress.flag_counts([]) == []


def test_blind_tests_are_stored_indexed_and_survive_reindex(three_tracks: Paths) -> None:
    paths = three_tracks
    test = blind.record(paths, "loop", CANDIDATES, ["C", "A", "B"], REVEALED)
    assert test.winner == "master"
    assert (paths.projects / "loop" / "blind_tests" / f"{test.id}.json").is_file()
    with index.open_index(paths) as conn:
        assert index.sync_blind_tests(conn, paths) == 1
        (result,) = progress.blind_results(conn)
    assert result.title == "loop" and result.winner == "Master"
    assert result.ranking == ["Master", "Reference", "Mix"]

    paths.db.unlink()
    with index.open_index(paths) as conn:
        assert index.reindex(conn, paths)["blind tests"] == 1
        assert progress.blind_results(conn) == [result]


@pytest.mark.parametrize(
    ("ranking", "revealed"),
    [
        (["A", "B"], REVEALED),
        (["A", "B", "C"], {"A": "mix", "B": "mix", "C": "master"}),
        (["A", "B", "D"], REVEALED),
    ],
)
def test_incomplete_blind_tests_are_rejected(three_tracks: Paths, ranking, revealed) -> None:
    with pytest.raises(ValueError):
        blind.record(three_tracks, "loop", CANDIDATES, ranking, revealed)
    assert blind.load_all(three_tracks) == []


def test_blind_test_needs_a_real_track(three_tracks: Paths) -> None:
    with pytest.raises(LookupError):
        blind.record(three_tracks, "../escape", CANDIDATES, ["A", "B", "C"], REVEALED)
