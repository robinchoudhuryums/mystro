from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from masterbench import audio_io, projects
from masterbench.config import Paths


@pytest.mark.parametrize(
    ("filename", "title", "slug"),
    [
        ("Night Drive v3.wav", "Night Drive", "night-drive"),
        ("Night Drive_mix2.wav", "Night Drive", "night-drive"),
        ("night drive bounce 12.aif", "night drive", "night-drive"),
        ("Track 7.wav", "Track 7", "track-7"),
        ("808s & Heartbreak.wav", "808s & Heartbreak", "808s-heartbreak"),
        ("v2.wav", "v2", "v2"),
    ],
)
def test_takes_of_a_song_share_a_track(filename: str, title: str, slug: str) -> None:
    assert projects.title_from_filename(Path(filename)) == title
    assert projects.slugify(title) == slug


def test_run_ids_sort_by_time_and_are_unique() -> None:
    ids = {projects.new_run_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(run_id.split("-")) == 3 for run_id in ids)


def test_open_project_stores_the_bounce_once(paths: Paths, fx: dict[str, Path]) -> None:
    first = projects.open_project(paths, fx["loop"], genre="hip-hop")
    second = projects.open_project(paths, fx["loop"])
    assert first.input_path == second.input_path == paths.projects / "loop" / "input" / "loop.wav"
    assert first.input_sha256 == audio_io.file_sha256(fx["loop"])
    assert second.track.genre == "hip-hop", "an existing genre is kept when none is given"
    assert second.track.created_at == first.track.created_at


def test_a_new_bounce_with_the_same_name_never_overwrites(
    paths: Paths, fx: dict[str, Path], tmp_path: Path
) -> None:
    original = projects.open_project(paths, fx["loop"])
    before = audio_io.file_sha256(original.input_path)
    revised = tmp_path / "loop.wav"
    shutil.copy(fx["pink_noise"], revised)
    newer = projects.open_project(paths, revised)
    assert newer.input_path != original.input_path
    assert newer.input_path.parent == original.input_path.parent
    assert audio_io.file_sha256(original.input_path) == before
    assert audio_io.file_sha256(newer.input_path) == audio_io.file_sha256(fx["pink_noise"])


def test_title_override_groups_differently_named_files(paths: Paths, fx: dict[str, Path]) -> None:
    a = projects.open_project(paths, fx["loop"], title="My Song")
    b = projects.open_project(paths, fx["pink_noise"], title="My Song")
    assert a.dir == b.dir and a.track.slug == "my-song"
    assert {p.name for p in (a.dir / "input").iterdir()} == {"loop.wav", "pink_noise.wav"}
