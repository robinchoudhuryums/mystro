from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from masterbench import blind, index
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.pipeline import MasterOptions
from masterbench.preview import streaming
from masterbench.references import library
from masterbench.references.library import Reference
from masterbench.ui import app, player, vocal_tab
from masterbench.vocal import presets as tuning_presets
from masterbench.vocal import workflow as vocal_workflow
from masterbench.vocal.tuning import NoteEdit


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
        ("Closest to this mix (picked automatically)", "hip-hop/auto"),
    ]
    assert value == "hip-hop/reference"
    closest_first = replace(settings, default_reference="auto")
    assert app.reference_choices(paths, selected, closest_first) == (choices, "hip-hop/auto")
    assert app.reference_choices(paths, "house", closest_first) == ([], None)
    assert app.library_rows(paths)[1][:3] == ["hip-hop", "reference", "yes"]


def test_run_view_points_players_at_real_files(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    options = MasterOptions.from_settings(settings, "hip-hop/default")
    result = pipeline.master(fx["loop"], options, paths, settings)
    markdown, flags, mix, master, ref, downloads = app.run_view(result)
    assert "| Master | -14.0 |" in markdown and "35 cents sharp of C2" in flags
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
    view = app.analysis_view(result, settings, paths)
    spectrum_plot, band_plot, loudness_plot, stereo_text, flags, song, low, markers, key = view
    assert [t.name for t in spectrum_plot.data] == ["Mix", "Master", "Reference"]
    assert [t.name for t in band_plot.data] == ["Mix", "Master"]
    assert [t.name for t in loudness_plot.data] == ["Mix", "Master"]
    assert "| Master | +0.9" in stereo_text and "35 cents sharp of C2" in flags
    assert (
        song.startswith("Key F minor")
        and "C2" in low
        and markers == "0 section"
        and key == "F minor"
    )

    blind.record(
        paths,
        "loop",
        [{"id": "mix", "label": "Mix"}, {"id": "master", "label": "Master"}],
        ["B", "A"],
        {"A": "mix", "B": "master"},
    )
    with index.open_index(paths) as conn:
        index.sync_blind_tests(conn, paths)
    habits, trends, loudness_trends, shape, flag_plot, rows = app.progress_view(paths, settings)
    assert habits.startswith("### Habits") and "preferred version: Master 1" in habits
    assert len(trends.data) == 7 and len(loudness_trends.data) == 3
    assert rows[0][1:] == ["loop", "Master", "Master > Mix"]
    assert list(flag_plot.data[0].y) == ["bass detuned"], "the loop's planted detuned note"


def test_empty_progress_view_explains_itself(paths: Paths, settings: Settings) -> None:
    habits, trends, _, _, _, rows = app.progress_view(paths, settings)
    assert rows == [] and "No tracks" in trends.layout.annotations[0].text
    assert "need at least 3 analyzed tracks; there are 0" in habits


def test_stems_reach_the_player_and_the_rebalance_table(
    paths: Paths, settings: Settings, tmp_path: Path
) -> None:
    from masterbench import fixtures
    from masterbench.stems import workflow

    stems, _ = fixtures.song_stems()
    for role, audio in stems.items():
        fixtures.write_wav(tmp_path / "stems" / f"{role}.wav", audio)
    project, stem_set = workflow.import_logic_stems(paths, tmp_path / "stems", "Song")
    result = pipeline.analyze(project.input_path, paths, settings, title="Song")
    diagnosis = workflow.diagnose(stem_set, result.mix_path, settings)

    rows = app.gain_rows(stem_set, diagnosis)
    assert [row[0] for row in rows] == ["bass", "drums", "other", "vocals"]
    assert all(row[3] == 0.0 and row[2] < 0 for row in rows)
    assert "| Vocal vs beat |" in app.summary.stems_markdown(diagnosis)

    config = json.loads(player.config(result, stems=workflow.sets_for_run(result)[0]))
    labels = [s["label"] for s in config["sources"]]
    assert labels == ["Mix", *[f"Stem: {n} (Logic)" for n in ("bass", "drums", "other", "vocals")]]
    assert all(Path(s["url"].split("=", 1)[1]).is_file() for s in config["sources"])


def test_phase_six_views(paths: Paths, settings: Settings, tmp_path: Path) -> None:
    from masterbench import fixtures
    from masterbench.preview import speakers
    from masterbench.references import target_curve
    from masterbench.ui import charts

    for name, verse_db in (("a", -7.0), ("b", -5.0)):
        song = fixtures.arranged_song(verse_db=verse_db, seed=ord(name))[0]
        library.add(paths, fixtures.write_wav(tmp_path / f"{name}.wav", song), "hip-hop", settings)
    curve = target_curve.build(paths, "hip-hop")
    assert app.library_rows(paths)[0][:4] == ["hip-hop", "a", "yes", "good"]

    weak = fixtures.write_wav(tmp_path / "weak.wav", fixtures.arranged_song(verse_db=-1.0)[0])
    options = MasterOptions.from_settings(settings, "", mode="curve", genre="hip-hop")
    result = pipeline.master(weak, options, paths, settings)

    view = app.analysis_view(result, settings, paths)
    spectrum_plot, band_plot, _, _, flags, song_md, low_md, markers, key = view
    names = [trace.name for trace in spectrum_plot.data if trace.name]
    assert "hip-hop curve (median)" in names and "hip-hop references (middle half)" in names
    assert "genre curve" in band_plot.layout.title.text
    assert "may not lift" in flags and "| hook 1 |" in song_md and "Hook minus verse" in song_md
    assert "C2" in low_md and markers.startswith("0 intro") and key == "F minor"
    assert charts.spectrum(result.reports, curve).data

    sims = speakers.render(result, paths)
    config = json.loads(player.config(result, sims=sims))
    assert [s["label"] for s in config["sections"]][:3] == ["intro", "verse 1", "hook 1"]
    assert config["sections"][2]["start"] == pytest.approx(20.6, abs=0.75)
    assert {e["label"] for e in config["events"]} >= {"bass detuned", "weak hook contrast"}
    assert [e["start"] for e in config["events"]] == sorted(e["start"] for e in config["events"])
    labels = [s["label"] for s in config["sources"]]
    assert labels[:3] == ["Mix", "Master", "Reference"] and "Sim: Phone speaker" in labels
    assert json.loads(player.config(pipeline.analyze(weak, paths, settings)))["sections"]


def test_vocal_tab_feeds_the_editor_and_the_player(
    paths: Paths, settings: Settings, fx: dict[str, Path]
) -> None:
    assert vocal_tab.source_choices(paths) == []
    assert json.loads(vocal_tab.editor_value(None, None)) == {"notes": []}
    assert json.loads(vocal_tab.player_config(None))["sources"] == []

    source = vocal_workflow.open_vocal(paths, fx["vocal"], track="Night Drive")
    report = vocal_workflow.report(source, paths)
    vocal_workflow.write_edits(source, {5: NoteEdit(bypass=True)})
    editor = json.loads(vocal_tab.editor_value(source, report))
    assert editor["source"] == source.id and len(editor["notes"]) == 8
    assert editor["scale"] == [0, 1, 3, 5, 7, 8, 10] and editor["step_s"] == 0.01
    assert editor["notes"][2] == {
        "index": 2,
        "start_s": report.notes[2].start_s,
        "end_s": report.notes[2].end_s,
        "center": report.notes[2].center,
        "target": 60.0,
        "name": "C4",
        "cents": report.notes[2].cents,
    }
    assert editor["edits"] == {"5": {"bypass": True, "target": None}}
    assert len(editor["curve"]) * editor["step_s"] == pytest.approx(editor["duration_s"], abs=0.02)
    assert "from the sung notes" in vocal_tab.key_note(report)

    only = json.loads(vocal_tab.player_config(source))
    assert [s["label"] for s in only["sources"]] == ["Original"] and only["selected"] == "original"
    wanted = tuning_presets.get(paths, "natural").settings("F minor", engine="world")
    version = vocal_workflow.tune(source, wanted, label="Natural")
    both = json.loads(vocal_tab.player_config(source))
    assert both["track"] == "night-drive" and both["selected"] == f"tuned_{version.id}"
    assert [s["label"] for s in both["sources"]] == ["Original", "Natural (world, 90%, 1 edit)"]
    for listed in both["sources"]:
        assert Path(listed["url"].removeprefix("/gradio_api/file=")).is_file()
        assert listed["lufs"] == pytest.approx(-17.1, abs=0.2)
    assert [e["label"] for e in both["events"]] == ["note 3 (+30 c)", "note 6 (+30 c)"]
    assert json.loads(vocal_tab.player_config(source, "original"))["selected"] == "original"
    assert vocal_tab.source_choices(paths) == [("Night Drive: vocal", source.id)]
    assert app.build(paths, settings).blocks
