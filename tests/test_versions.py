from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from masterbench import audio_io, fixtures
from masterbench.analysis import loudness
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.pipeline import MasterOptions, RunResult
from masterbench.references import library, target_curve
from masterbench.release import versions
from masterbench.release.versions import VersionError
from masterbench.stems import separate, workflow

SR = fixtures.SR
NAMES = {"drums": "Drums", "bass": "808 Bass", "other": "Pads", "vocals": "Lead Vox"}


@pytest.fixture
def song(paths: Paths, settings: Settings, tmp_path: Path, fx: dict[str, Path]):
    """A track with stems from 'Logic', a hip-hop reference, and a two-reference curve."""
    stems, _ = fixtures.song_stems()
    for role, audio in stems.items():
        fixtures.write_wav(tmp_path / "stems" / f"{NAMES[role]}.wav", audio)
    library.add(paths, fx["reference"], "hip-hop", settings)
    second = fixtures.write_wav(tmp_path / "ref2.wav", fixtures.arranged_song(seed=4)[0])
    library.add(paths, second, "hip-hop", settings)
    target_curve.build(paths, "hip-hop")
    project, stem_set = workflow.import_logic_stems(
        paths, tmp_path / "stems", "Night Drive", "hip-hop"
    )
    return project, stem_set


def master(song, paths: Paths, settings: Settings, **options) -> RunResult:
    project, _ = song
    chosen = MasterOptions.from_settings(
        settings, options.pop("reference", "hip-hop/default"), title="Night Drive", **options
    )
    return pipeline.master(project.input_path, chosen, paths, settings)


def parts(run: RunResult) -> dict[str, np.ndarray]:
    folder = versions.versions_dir(run)
    return {
        name: audio_io.load(folder / f"{name}_24.wav")[0] for name in ("instrumental", "acapella")
    }


def test_matchering_master_is_reproduced_on_the_parts(
    song, paths: Paths, settings: Settings
) -> None:
    """Phase 7 criterion: instrumental + a cappella, before limiting, is the master."""
    run = master(song, paths, settings)
    made = versions.make(run, paths, settings)
    assert made.method == "fitted" and not made.separated
    assert made.sum_error_db < versions.GOOD_SUM_DB, "within the set error of -25 dB"
    assert made.sum_error_db < -30, "in fact about -35 dB on the fixture"
    assert made.stems_vs_mix_db < -90, "Logic stems add up to the mix"
    assert [f.code for f in made.flags] == ["fitted_processing"]

    main, _ = audio_io.load(run.output("master_24"))
    written = parts(run)
    assert written["instrumental"].shape == written["acapella"].shape == main.shape
    assert versions.error_db(written["instrumental"] + written["acapella"], main) < -25
    for version in made.versions:
        assert version.true_peak_dbtp <= -1.0 + 0.05
        measured = loudness.integrated(written[version.name], SR)
        assert version.lufs_i == pytest.approx(measured, abs=0.05)
    by_name = {v.name: v for v in made.versions}
    assert by_name["instrumental"].lufs_i > by_name["acapella"].lufs_i
    assert by_name["acapella"].limiter_reduction_db == 0.0, "an a cappella is not limited hard"


def test_curve_master_is_reproduced_exactly(song, paths: Paths, settings: Settings) -> None:
    run = master(song, paths, settings, reference="", mode="curve", genre="hip-hop")
    made = versions.make(run, paths, settings)
    assert made.method == "exact" and made.flags == []
    assert made.sum_error_db < -30, "what remains is the limiter's own work on the master"

    # With the limiter put back, the summed parts are the master to within the file's precision.
    _, stem_set = song
    eq_gain, _ = versions.locked_process(
        run, audio_io.load(run.mix_path)[0], audio_io.load(run.output("master_24"))[0], SR
    )
    from masterbench.master import limiter

    total = eq_gain(separate.mixdown(stem_set))
    limited = limiter.true_peak_limit(total, SR, -1.0).audio
    assert versions.error_db(limited, audio_io.load(run.output("master_24"))[0]) < -70


def test_locked_processing_is_linear(song, paths: Paths, settings: Settings) -> None:
    _, stem_set = song
    for options in ({}, {"reference": "", "mode": "curve", "genre": "hip-hop"}):
        run = master(song, paths, settings, **options)
        mix, master_audio = (
            audio_io.load(run.mix_path)[0],
            audio_io.load(run.output("master_24"))[0],
        )
        process, _ = versions.locked_process(run, mix, master_audio, SR)
        vocal = stem_set.load("lead-vox")
        rest = separate.mixdown(stem_set) - vocal
        assert versions.error_db(process(vocal) + process(rest), process(vocal + rest)) < -100


def test_chains_decide_between_exact_and_fitted(song, paths: Paths, settings: Settings) -> None:
    curve = {"reference": "", "mode": "curve", "genre": "hip-hop"}
    filtered = master(song, paths, settings, **curve, pre_chain="streaming-safe")
    made = versions.make(filtered, paths, settings)
    assert made.method == "exact" and made.sum_error_db < -30, "a high-pass is linear"

    saturated = master(song, paths, settings, **curve, post_chain="warm")
    made = versions.make(saturated, paths, settings)
    assert made.method == "fitted", "saturation depends on the level, so it cannot be split"
    assert made.sum_error_db < versions.GOOD_SUM_DB


def test_versions_are_stored_beside_the_project(song, paths: Paths, settings: Settings) -> None:
    run = master(song, paths, settings)
    assert versions.load(run) is None
    made = versions.make(run, paths, settings)
    folder = paths.projects / "night-drive" / "versions" / run.manifest.id
    assert versions.versions_dir(run) == folder
    assert {f.name for f in folder.iterdir()} == {
        "instrumental_24.wav",
        "acapella_24.wav",
        "versions.json",
    }
    assert json.loads((folder / "versions.json").read_text())["run_id"] == run.manifest.id
    assert versions.load(run) == made
    (folder / "acapella_24.wav").unlink()
    assert versions.load(run) is None


def test_separated_stems_are_flagged(song, paths: Paths, settings: Settings) -> None:
    project, real = song
    run = master(song, paths, settings)

    def fake(source: Path, model_file: str, out_dir: Path) -> list[Path]:
        audio, info = audio_io.load(source)
        return [
            audio_io.save_wav(out_dir / f"x_({stem})_m.wav", audio * share, info.sample_rate, 32)
            for stem, share in (("Vocals", 0.3), ("Instrumental", 0.7))
        ]

    separated = workflow.separate_mix(project, "mdx-2", fake)
    made = versions.make(run, paths, settings, stem_set_name=separated.name)
    assert made.separated and made.stem_set == separated.name
    assert "separated_stems" in {f.code for f in made.flags}
    assert versions.make(run, paths, settings).stem_set == real.name, "Logic stems are preferred"


def test_stems_that_miss_part_of_the_mix_are_called_out(
    song, paths: Paths, settings: Settings
) -> None:
    _, stem_set = song
    run = master(song, paths, settings)
    fixtures.write_wav(stem_set.path("pads"), np.zeros((stem_set.frames, 2)), SR, "FLOAT")
    made = versions.make(run, paths, settings)
    flagged = {f.code: f for f in made.flags}
    assert (
        "stems_differ_from_mix" in flagged
        and "missing from the stems" in flagged["stems_differ_from_mix"].message
    )
    assert made.stems_vs_mix_db > versions.GOOD_SUM_DB


def test_versions_need_a_master_and_usable_stems(
    paths: Paths, settings: Settings, reference, fx: dict[str, Path], tmp_path: Path
) -> None:
    analyzed = pipeline.analyze(fx["loop"], paths, settings)
    with pytest.raises(VersionError, match="made from a master"):
        versions.make(analyzed, paths, settings)
    mastered = pipeline.master(
        fx["loop"], MasterOptions.from_settings(settings, "hip-hop/default"), paths, settings
    )
    with pytest.raises(VersionError, match="need stems with a vocal"):
        versions.make(mastered, paths, settings)
