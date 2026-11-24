from __future__ import annotations

from pathlib import Path

from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.pipeline import MasterOptions
from masterbench.references import library
from masterbench.references.library import Reference
from masterbench.ui import app


def test_app_builds_with_an_empty_library(paths: Paths, settings: Settings) -> None:
    assert app.build(paths, settings).blocks
    assert app.reference_choices(paths, "hip-hop") == ([], None)
    assert app.run_choices(paths) == [] and app.library_rows(paths) == []


def test_genre_default_reference_is_preselected(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    library.add(paths, fx["pink_noise"], "hip-hop", settings)
    genres, selected = app.genre_choices(paths, settings)
    assert selected == "hip-hop" and "house" in genres
    choices, value = app.reference_choices(paths, selected)
    assert choices == [
        ("reference (default)", "hip-hop/reference"),
        ("pink_noise", "hip-hop/pink_noise"),
    ]
    assert value == "hip-hop/reference"
    assert app.library_rows(paths)[1][:3] == ["hip-hop", "reference", "yes"]


def test_run_view_points_players_at_real_files(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    options = MasterOptions.from_settings(settings, "hip-hop/default")
    result = pipeline.master(fx["loop"], options, paths, settings)
    markdown, flags, mix, master, ref, downloads = app.run_view(result)
    assert "| Master | -14.0 |" in markdown and flags == "No flags."
    assert Path(mix).is_file() and Path(master).name == "master_24.wav"
    assert Path(ref) == reference.path
    assert [Path(p).name for p in downloads] == ["master_24.wav", "master_16.wav"]
    assert all(Path(p).is_relative_to(paths.root) for p in (mix, master, ref, *downloads))

    analyzed = pipeline.analyze(fx["clipped"], paths, settings)
    _, flags, _, master, ref, downloads = app.run_view(analyzed)
    assert "Clipping" in flags and (master, ref, downloads) == (None, None, None)
    labels = [label for label, _ in app.run_choices(paths)]
    assert any("clipped · analyzed" in label for label in labels)
    assert any("loop · mastered" in label for label in labels)
