from __future__ import annotations

import json
from pathlib import Path

import pytest

from masterbench import batch, fixtures, summary
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.pipeline import MasterOptions
from masterbench.references.library import Reference


@pytest.fixture
def ep(tmp_path: Path) -> list[Path]:
    """Four 'songs': three takes on the loop and one that is spectrally unlike them."""
    tracks = {
        "01 Intro.wav": fixtures.drum_808_loop(seed=1)[0],
        "02 Night Drive.wav": fixtures.drum_808_loop(seed=2, hat_gain=0.14)[0],
        "03 Low End.wav": fixtures.drum_808_loop(seed=3, bass_gain=0.9)[0],
        "04 Interlude.wav": fixtures.pink_noise(12.0, rms_dbfs=-24.0),
    }
    return [fixtures.write_wav(tmp_path / "ep" / name, audio) for name, audio in tracks.items()]


def options(settings: Settings) -> MasterOptions:
    return MasterOptions.from_settings(settings, "hip-hop/default", genre="hip-hop")


def test_four_track_ep_runs_unattended_with_a_consistency_report(
    paths: Paths, settings: Settings, reference: Reference, ep: list[Path]
) -> None:
    stages: list[float] = []
    result = batch.run_batch(
        ep, options(settings), paths, settings, "Test EP", progress=lambda s, f: stages.append(f)
    )
    assert len(result.runs) == 4 and result.failures == [] and result.report is not None
    assert stages == sorted(stages) and stages[-1] == 1.0

    report = result.report
    assert [t.title for t in report.tracks] == [
        "01 Intro",
        "02 Night Drive",
        "03 Low End",
        "04 Interlude",
    ]
    assert all(t.lufs_i == pytest.approx(-14.0, abs=0.2) for t in report.tracks)
    assert report.loudness_spread_lu < 0.4 and report.max_true_peak_dbtp <= -1.0
    assert not [f for f in report.flags if f.code == "ep_loudness_outlier"]
    assert all(set(t.band_offsets_db) == set(batch.BANDS) for t in report.tracks)
    # Matching every track to one reference pulls even the odd one out toward the others.
    assert max(abs(v) for t in report.tracks for v in t.band_offsets_db.values()) < 6

    stored = json.loads((result.dir / "batch.json").read_text())
    assert stored["name"] == "Test EP" and len(stored["run_ids"]) == 4
    assert stored["report"]["loudness_spread_lu"] == report.loudness_spread_lu
    assert result.dir.parent == paths.root / "batches"

    text = summary.batch_text(result)
    assert "04 Interlude" in text and "Spotify applies one gain to the whole set" in text
    assert "| 01 Intro | -14.0 |" in summary.batch_markdown(result)


def test_a_bad_file_does_not_stop_the_batch(
    paths: Paths, settings: Settings, reference: Reference, ep: list[Path]
) -> None:
    broken = ep[0].with_name("00 Broken.wav")
    broken.write_bytes(b"RIFF not audio")
    result = batch.run_batch([broken, *ep[:2]], options(settings), paths, settings)
    assert len(result.runs) == 2 and result.report is not None
    (failure,) = result.failures
    assert failure.source == str(broken) and "AudioError" in failure.error
    assert "✖ 00 Broken.wav failed" in summary.batch_text(result)

    alone = batch.run_batch([ep[0]], options(settings), paths, settings)
    assert alone.report is None and "needs at least two" in summary.batch_text(alone)


def test_per_track_references_override_the_shared_one(
    paths: Paths, settings: Settings, reference: Reference, ep: list[Path], fx: dict[str, Path]
) -> None:
    own = {"02 Night Drive.wav": str(fx["pink_noise"])}
    result = batch.run_batch(ep[:2], options(settings), paths, settings, references=own)
    specs = [run.manifest.settings["reference"]["spec"] for run in result.runs]
    assert specs == ["hip-hop/reference", str(fx["pink_noise"])]


def test_a_track_mastered_louder_stands_out(
    paths: Paths, settings: Settings, reference: Reference, ep: list[Path]
) -> None:
    runs = [pipeline.master(source, options(settings), paths, settings) for source in ep[:2]]
    loud = pipeline.master(ep[2], MasterOptions("hip-hop/default", -11.0, -1.0), paths, settings)
    report = batch.consistency([*runs, loud], settings.thresholds)
    assert report.loudness_spread_lu == pytest.approx(3.0, abs=0.3)
    (outlier,) = [f for f in report.flags if f.code == "ep_loudness_outlier"]
    assert "03 Low End is 3.0 LU louder than the set's median." in outlier.message
    assert report.tracks[2].loudness_offset_lu == pytest.approx(3.0, abs=0.2)


def test_a_track_with_a_different_tone_stands_out(
    paths: Paths, settings: Settings, reference: Reference, ep: list[Path]
) -> None:
    from masterbench.master import presets
    from masterbench.master.chain import ChainPreset, Module

    presets.save(paths, ChainPreset("bright", "", (Module("high_shelf", {"gain_db": 9.0}),)))
    runs = [pipeline.master(source, options(settings), paths, settings) for source in ep[:2]]
    bright = MasterOptions.from_settings(settings, "hip-hop/default", post_chain="bright")
    runs.append(pipeline.master(ep[2], bright, paths, settings))
    report = batch.consistency(runs, settings.thresholds)
    flagged = {f.code: f for f in report.flags}
    assert "ep_band_air_outlier" in flagged
    assert "03 Low End has" in flagged["ep_band_air_outlier"].message
    assert (
        "more Air (10 kHz to 20 kHz) than the set's median"
        in flagged["ep_band_air_outlier"].message
    )
