from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from masterbench import fixtures
from masterbench.config import Paths, Settings
from masterbench.references import library, suggest
from masterbench.references.quality import Quality

SR = fixtures.SR


def transcode(source: Path, dest: Path, *args: str) -> Path:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), *args, str(dest)]
    subprocess.run(cmd, check=True)
    return dest


def quality(path: Path, settings: Settings) -> Quality:
    return library.analyze(path, settings)[1]


@pytest.fixture(scope="module")
def song_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("quality")
    return fixtures.write_wav(folder / "song.wav", fixtures.arranged_song()[0])


def test_a_full_length_lossless_track_is_good(song_file: Path, settings: Settings) -> None:
    found = quality(song_file, settings)
    assert found.badge == "good" and found.issues == [] and found.cutoff_hz is None
    assert found.usable_for_curves


def test_a_lossy_rip_is_flagged_even_as_a_wav(song_file: Path, settings: Settings) -> None:
    mp3 = transcode(song_file, song_file.with_name("rip.mp3"), "-b:a", "128k")
    resaved = transcode(mp3, song_file.with_name("rip.wav"))
    for path in (mp3, resaved):
        found = quality(path, settings)
        assert found.badge == "poor" and not found.usable_for_curves
        assert found.cutoff_hz is not None and 15000 <= found.cutoff_hz <= 17500
        (issue,) = found.issues
        assert issue.code == "lossy_rip" and "looks like a lossy rip" in issue.message


def test_a_high_bitrate_file_is_only_cautioned_for_its_format(song_file, settings) -> None:
    mp3 = transcode(song_file, song_file.with_name("hq.mp3"), "-b:a", "320k")
    assert [f.code for f in quality(mp3, settings).issues] == ["lossy_format"]
    assert quality(mp3, settings).badge == "caution"
    assert quality(transcode(mp3, song_file.with_name("hq.wav")), settings).badge == "good"


def test_clipped_mono_and_short_references_are_flagged(tmp_path: Path, settings: Settings) -> None:
    song, _ = fixtures.arranged_song()
    cases = {
        "clipped": (np.clip(song * 6, -1, 1), {"clipped"}, "poor"),
        "mono": (np.repeat(song.mean(axis=1)[:, None], 2, axis=1), {"mono"}, "poor"),
        "panned": (np.column_stack([song[:, 0], 0.5 * song[:, 0]]), {"fake_stereo"}, "caution"),
        "short": (fixtures.drum_808_loop()[0], {"short"}, "caution"),
        "squashed": (np.tanh(song * 40) * 0.5, {"brickwalled"}, "caution"),
    }
    for name, (audio, codes, badge) in cases.items():
        found = quality(fixtures.write_wav(tmp_path / f"{name}.wav", audio), settings)
        assert {f.code for f in found.issues} == codes, name
        assert found.badge == badge, name


def test_quality_is_stored_with_the_reference(paths: Paths, settings: Settings, song_file) -> None:
    ref = library.add(paths, song_file, "hip-hop", settings)
    assert ref.quality is not None and ref.quality.badge == "good"
    assert ref.report.musical is not None and ref.report.musical.key == "F minor"
    assert ref.report.sections is not None and len(ref.report.sections.items) == 6
    (listed,) = library.list_references(paths)
    assert listed.quality == ref.quality and library.refresh(paths, settings) == 0


def test_suggestions_rank_the_nearest_reference_first(
    paths: Paths, settings: Settings, song_file: Path, tmp_path: Path
) -> None:
    library.add(paths, song_file, "hip-hop", settings)
    library.add(
        paths,
        fixtures.write_wav(tmp_path / "noise.wav", fixtures.pink_noise(20.0)),
        "hip-hop",
        settings,
    )
    t = np.arange(SR * 20) / SR
    bright = np.column_stack([np.sin(2 * np.pi * 6000 * t), np.sin(2 * np.pi * 6100 * t)]) * 0.2
    library.add(paths, fixtures.write_wav(tmp_path / "bright.wav", bright), "hip-hop", settings)

    loop = fixtures.write_wav(tmp_path / "loop.wav", fixtures.drum_808_loop()[0])
    mix, _ = library.analyze(loop, settings)
    found = suggest.suggest(mix, library.list_references(paths, "hip-hop"))
    assert [s.reference.name for s in found][0] == "song"
    assert found[0].similarity > 0.8 > found[-1].similarity
    assert found[0].why.startswith("closest ") and len(found[0].reasons) == 2
    assert len(suggest.suggest(mix, library.list_references(paths, "hip-hop"), count=1)) == 1
    assert suggest.suggest(mix, []) == []

    itself, _ = library.analyze(song_file, settings)
    names = [s.reference.name for s in suggest.suggest(itself, library.list_references(paths))]
    assert "song" not in names, "a track is never suggested as its own reference"
