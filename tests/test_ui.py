from __future__ import annotations

import json
from pathlib import Path

import pytest

from masterbench import blind, index
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.pipeline import MasterOptions
from masterbench.preview import streaming
from masterbench.references import library
from masterbench.references.library import Reference
from masterbench.ui import app, player


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


def test_player_sources_carry_what_level_matching_needs(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    options = MasterOptions.from_settings(settings, "hip-hop/default")
    first = pipeline.master(fx["loop"], options, paths, settings)
    louder = pipeline.master(
        fx["loop"], MasterOptions("hip-hop/default", -10.0, -1.0), paths, settings
    )

    config = json.loads(player.config(first))
    assert config["track"] == "loop" and config["selected"] == "master"
    assert [s["id"] for s in config["sources"]] == ["mix", "master", "reference"]
    for source in config["sources"]:
        assert source["url"].startswith("/gradio_api/file=")
        assert Path(source["url"].split("=", 1)[1]).is_file()
    quietest = min(s["lufs"] for s in config["sources"])
    assert quietest == pytest.approx(-19.7, abs=0.1), "matching plays everything at the mix level"

    preview = streaming.render(first, settings)
    full = json.loads(player.config(first, preview, [louder]))
    ids = [s["id"] for s in full["sources"]]
    assert ids[3:6] == ["codec_vorbis", "codec_aac", "codec_opus"]
    assert ids[6] == f"master_{louder.manifest.id}" and "-10 LUFS" in full["sources"][6]["label"]
    assert app.other_masters(paths, first) == [
        (app.other_masters(paths, first)[0][0], louder.manifest.id)
    ]

    text = app.streaming_markdown(preview)
    assert "| Ogg Vorbis ~320k (Spotify-like) | libvorbis |" in text
    assert "Spotify (Normal) plays this unchanged." in text
    assert "not been rendered" in app.streaming_markdown(None)

    empty = json.loads(player.config(None))
    assert empty["sources"] == [] and "Choose a run" in empty["message"]
    analyzed = pipeline.analyze(fx["clipped"], paths, settings)
    assert json.loads(player.config(analyzed))["selected"] == "mix"


def test_analysis_and_progress_views_build(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    options = MasterOptions.from_settings(settings, "hip-hop/default")
    result = pipeline.master(fx["loop"], options, paths, settings)
    spectrum_plot, band_plot, loudness_plot, stereo_text, flags = app.analysis_view(
        result, settings
    )
    assert [t.name for t in spectrum_plot.data] == ["Mix", "Master", "Reference"]
    assert [t.name for t in band_plot.data] == ["Mix", "Master"]
    assert [t.name for t in loudness_plot.data] == ["Mix", "Master"]
    assert "| Master | +0.9" in stereo_text and flags == "No flags."

    blind.record(
        paths,
        "loop",
        [{"id": "mix", "label": "Mix"}, {"id": "master", "label": "Master"}],
        ["B", "A"],
        {"A": "mix", "B": "master"},
    )
    with index.open_index(paths) as conn:
        index.sync_blind_tests(conn, paths)
    trends, loudness_trends, flag_plot, rows = app.progress_view(paths)
    assert len(trends.data) == 7 and len(loudness_trends.data) == 3
    assert rows[0][1:] == ["loop", "Master", "Master > Mix"]
    assert flag_plot.layout.annotations[0].text == "No flags raised so far."


def test_empty_progress_view_explains_itself(paths: Paths) -> None:
    trends, loudness_trends, flag_plot, rows = app.progress_view(paths)
    assert rows == [] and "No tracks" in trends.layout.annotations[0].text
