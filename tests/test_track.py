from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from masterbench import fixtures, projects
from masterbench import track as track_tools
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.pipeline import RunResult
from masterbench.stems import workflow
from masterbench.track import TrackError

SR = fixtures.SR
MARKERS = "0 intro, 6.9 verse, 0:20.6 chorus, 34.3 verse, 48 drop, 1:01.7 outro"
KINDS = ["intro", "verse", "hook", "verse", "hook", "outro"]


@pytest.fixture
def song_run(paths: Paths, settings: Settings, tmp_path: Path) -> RunResult:
    song = fixtures.write_wav(tmp_path / "Night Drive.wav", fixtures.arranged_song()[0])
    return pipeline.analyze(song, paths, settings, genre="hip-hop")


def test_markers_parse_seconds_minutes_and_everyday_words() -> None:
    markers = track_tools.parse_markers(MARKERS)
    assert [m["kind"] for m in markers] == KINDS
    assert [m["start_s"] for m in markers] == [0.0, 6.9, 20.6, 34.3, 48.0, 61.7]
    assert track_tools.parse_markers("30 hook\n0 verse") == [
        {"start_s": 0.0, "kind": "verse"},
        {"start_s": 30.0, "kind": "hook"},
    ]
    for bad, message in (
        ("5 solo", "unknown section kind"),
        ("hook", "cannot read"),
        (" , ", "no markers"),
    ):
        with pytest.raises(TrackError, match=message):
            track_tools.parse_markers(bad)


def test_the_first_analysis_detects_key_tempo_and_sections(song_run: RunResult) -> None:
    mix = song_run.reports["mix"]
    assert mix.musical is not None and mix.musical.key == "F minor"
    assert mix.musical.tempo_bpm == pytest.approx(140.0, abs=0.5)
    assert mix.sections is not None and mix.sections.source == "auto"
    assert [i.kind for i in mix.sections.items] == KINDS
    assert mix.lowend is not None and len(mix.lowend.notes) > 10
    assert [f.code for f in mix.flags].count("bass_detuned") == 4, "one per pass of the loop"
    assert track_tools.markers_text(mix.sections).startswith("0 intro, 6.")


def test_hand_set_sections_apply_to_later_analyses(
    song_run: RunResult, paths: Paths, settings: Settings
) -> None:
    saved = track_tools.set_sections(song_run, MARKERS)
    assert saved.source == "manual" and [i.kind for i in saved.items] == KINDS
    stored = projects.read_overrides(track_tools.project_dir(song_run))
    assert len(stored["markers"]) == 6 and len(stored["envelope"]) > 600

    again = pipeline.analyze(song_run.mix_path, paths, settings, title="Night Drive")
    parts = again.reports["mix"].sections
    assert parts is not None and parts.source == "manual"
    assert [i.start_s for i in parts.items] == [0.0, 6.9, 20.6, 34.3, 48.0, 61.7]

    with pytest.raises(TrackError, match="only 68.6 s long"):
        track_tools.set_sections(song_run, "0 intro, 90 hook")
    track_tools.clear_sections(song_run)
    assert "markers" not in projects.read_overrides(track_tools.project_dir(song_run))


def test_markers_move_with_a_re_bounce_that_starts_later(
    song_run: RunResult, paths: Paths, settings: Settings, tmp_path: Path
) -> None:
    track_tools.set_sections(song_run, MARKERS)
    song, _ = fixtures.arranged_song()
    late = np.concatenate([np.zeros((2 * SR, 2)), song])
    rebounce = fixtures.write_wav(tmp_path / "Night Drive v2.wav", late)
    parts = pipeline.analyze(rebounce, paths, settings).reports["mix"].sections
    assert parts is not None and parts.source == "manual"
    assert [i.kind for i in parts.items] == ["intro", *KINDS], "the new silence leads in"
    assert [i.start_s for i in parts.items][1:] == pytest.approx(
        [2.0, 8.9, 22.6, 36.3, 50.0, 63.7], abs=0.15
    )


def test_a_hand_set_key_wins_over_detection(
    song_run: RunResult, paths: Paths, settings: Settings
) -> None:
    assert track_tools.set_key(paths, "night-drive", "g# major") == "G# major"
    again = pipeline.analyze(song_run.mix_path, paths, settings, title="Night Drive")
    music = again.reports["mix"].musical
    assert music is not None and music.key == "G# major" and music.key_confidence == 1.0
    assert "bass_out_of_key" not in {f.code for f in again.flags}, "F minor's notes are in G# major"

    track_tools.set_key(paths, "night-drive", "G major")
    wrong = pipeline.analyze(song_run.mix_path, paths, settings, title="Night Drive")
    assert "bass_out_of_key" in {f.code for f in wrong.flags}

    assert track_tools.set_key(paths, "night-drive", None) is None
    with pytest.raises(TrackError, match="not a key"):
        track_tools.set_key(paths, "night-drive", "H minor")
    with pytest.raises(TrackError, match="no track 'nope'"):
        track_tools.set_key(paths, "nope", "F minor")


def test_master_is_measured_over_the_mix_sections(
    paths: Paths, settings: Settings, tmp_path: Path, fx: dict[str, Path]
) -> None:
    from masterbench.master.pipeline import MasterOptions
    from masterbench.references import library

    library.add(paths, fx["reference"], "hip-hop", settings)
    weak = fixtures.write_wav(tmp_path / "weak.wav", fixtures.arranged_song(verse_db=-1.0)[0])
    result = pipeline.master(
        weak, MasterOptions.from_settings(settings, "hip-hop/default"), paths, settings
    )
    mix, master = result.reports["mix"], result.reports["master"]
    assert mix.sections is not None and master.sections is not None
    assert [(i.label, i.start_s) for i in master.sections.items] == [
        (i.label, i.start_s) for i in mix.sections.items
    ]
    assert master.musical == mix.musical and master.lowend is None
    assert master.sections.items[2].lufs > mix.sections.items[2].lufs + 2, "louder after mastering"
    codes = [f.code for f in result.flags]
    assert codes.count("weak_hook_contrast") == 1, "said once, on the mix"
    weak_flag = next(f for f in result.flags if f.code == "weak_hook_contrast")
    assert weak_flag.start_s == mix.sections.items[2].start_s


def test_low_end_uses_stems_when_the_track_has_them(
    paths: Paths, settings: Settings, tmp_path: Path
) -> None:
    stems, _ = fixtures.song_stems()
    for role, audio in stems.items():
        fixtures.write_wav(tmp_path / "stems" / f"{role}.wav", audio)
    project, _ = workflow.import_logic_stems(paths, tmp_path / "stems", "Song", "hip-hop")
    run = pipeline.analyze(project.input_path, paths, settings, title="Song")
    assert run.reports["mix"].lowend is not None and run.reports["mix"].lowend.source == "mix"
    found = track_tools.low_end(run, paths, settings)
    assert found.source == "bass stem" and found.clash is not None and found.clash.share > 0.9
    assert (run.run_dir / "lowend.json").is_file()
