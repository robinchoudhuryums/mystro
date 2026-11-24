from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from masterbench.config import Paths, Settings
from masterbench.references import library
from masterbench.references.library import Reference, ReferenceError


def test_first_reference_becomes_the_genre_default(paths: Paths, reference: Reference) -> None:
    assert reference.is_default and reference.spec == "hip-hop/reference"
    assert reference.path == paths.references / "hip-hop" / "reference.wav"
    assert reference.report.role == "reference"
    assert reference.report.loudness.lufs_i == pytest.approx(-8.7, abs=0.1)
    assert (paths.references / "hip-hop" / "reference.wav.analysis.json").is_file()


def test_listing_reads_cached_analyses(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    library.add(paths, fx["pink_noise"], "house", settings, notes="flat")
    library.add(paths, fx["sine_sweep"], "hip-hop", settings)
    listed = library.list_references(paths)
    assert [(r.genre, r.name, r.is_default) for r in listed] == [
        ("hip-hop", "reference", True),
        ("hip-hop", "sine_sweep", False),
        ("house", "pink_noise", True),
    ]
    assert listed[2].notes == "flat"
    assert [r.name for r in library.list_references(paths, "house")] == ["pink_noise"]


@pytest.mark.parametrize("spec", ["hip-hop/default", "hiphop/default", "Hip Hop/reference"])
def test_resolve_accepts_loose_genre_spelling(
    paths: Paths, settings: Settings, reference: Reference, spec: str
) -> None:
    assert library.resolve(paths, spec, settings).path == reference.path


def test_resolve_accepts_a_file_outside_the_library(
    paths: Paths, settings: Settings, fx: dict[str, Path]
) -> None:
    adhoc = library.resolve(paths, str(fx["reference"]), settings)
    assert adhoc.genre == "" and adhoc.spec == str(fx["reference"])
    assert library.list_references(paths) == []


@pytest.mark.parametrize("spec", ["hip-hop/missing", "nope/default", "just-a-name"])
def test_resolve_explains_what_is_missing(
    paths: Paths, settings: Settings, reference: Reference, spec: str
) -> None:
    with pytest.raises(ReferenceError):
        library.resolve(paths, spec, settings)


def test_default_can_be_changed_and_survives_removal(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    library.add(paths, fx["pink_noise"], "hip-hop", settings)
    library.set_default(paths, "hiphop", "pink_noise")
    assert library.get(paths, "hip-hop", "default").name == "pink_noise"
    library.remove(paths, "hip-hop", "pink_noise")
    assert library.get(paths, "hip-hop", "default").name == "reference"
    with pytest.raises(ReferenceError):
        library.set_default(paths, "hip-hop", "pink_noise")


def test_add_refuses_a_different_file_with_the_same_name(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path], tmp_path: Path
) -> None:
    impostor = tmp_path / "reference.wav"
    shutil.copy(fx["pink_noise"], impostor)
    with pytest.raises(ReferenceError, match="already exists"):
        library.add(paths, impostor, "hip-hop", settings)
    again = library.add(paths, fx["reference"], "hip-hop", settings, notes="updated")
    assert again.notes == "updated" and len(library.list_references(paths)) == 1


def test_new_genres_are_created_on_demand(paths: Paths, settings: Settings, fx) -> None:
    ref = library.add(paths, fx["reference"], "UK Garage", settings)
    assert ref.genre == "uk-garage" and "uk-garage" in library.genres(paths)
    with pytest.raises(ReferenceError, match="not an audio file"):
        library.add(paths, fx["truth"], "house", settings)
