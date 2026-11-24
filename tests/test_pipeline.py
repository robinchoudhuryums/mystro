from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import soundfile as sf

from masterbench import audio_io, fixtures, index
from masterbench.analysis import loudness
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.matchering_engine import MasteringError
from masterbench.master.pipeline import MasterOptions
from masterbench.references.library import Reference


def options(settings: Settings, **overrides) -> MasterOptions:
    return MasterOptions.from_settings(settings, "hip-hop/default", **overrides)


def dump(paths: Paths) -> dict[str, list[tuple]]:
    """Index contents without surrogate ids, for before/after comparison."""
    queries = {
        "tracks": "SELECT slug, title, genre, created_at FROM tracks ORDER BY slug",
        "runs": "SELECT r.id, t.slug, r.kind, r.settings_json, r.path, r.lib_versions_json,"
        " r.created_at FROM runs r JOIN tracks t ON t.id = r.track_id ORDER BY r.id",
        "analyses": "SELECT run_id, source_path, role, sha256, sample_rate, lufs_i, true_peak_dbtp,"
        " lra, crest_db, correlation, low_mono_corr FROM analyses"
        " ORDER BY source_path, role, run_id",
        "band_deltas": "SELECT a.source_path, a.role, a.run_id, d.band, d.delta_db, d.vs, d.section"
        " FROM band_deltas d JOIN analyses a ON a.id = d.analysis_id"
        " ORDER BY a.source_path, a.role, a.run_id, d.band",
        "flags": "SELECT a.source_path, f.code, f.severity, f.message, f.value FROM flags f"
        " JOIN analyses a ON a.id = f.analysis_id ORDER BY a.source_path, f.code",
        "references": "SELECT genre, name, path, is_default, notes FROM reference_tracks"
        " ORDER BY path",
    }
    with index.open_index(paths) as conn:
        return {name: [tuple(row) for row in conn.execute(sql)] for name, sql in queries.items()}


def test_master_hits_the_streaming_target(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    source_hash = audio_io.file_sha256(fx["loop"])
    stages: list[tuple[str, float]] = []
    result = pipeline.master(
        fx["loop"], options(settings), paths, settings, lambda s, f: stages.append((s, f))
    )

    master_24, master_16 = result.output("master_24"), result.output("master_16")
    audio, info = audio_io.load(master_24)
    measured = loudness.measure(audio, info.sample_rate)
    assert measured.lufs_i == pytest.approx(-14.0, abs=0.2)
    assert measured.true_peak_dbtp <= -1.0
    assert (info.subtype, info.sample_rate, info.channels) == ("PCM_24", 44100, 2)
    assert info.frames == sf.info(fx["loop"]).frames
    assert sf.info(master_16).subtype == "PCM_16"

    assert audio_io.file_sha256(fx["loop"]) == source_hash, "the input file is never modified"
    assert audio_io.file_sha256(result.mix_path) == source_hash
    assert result.flags == []
    assert [f for _, f in stages] == sorted(f for _, f in stages) and stages[-1] == ("Done", 1.0)


def test_run_manifest_records_what_is_needed_to_reproduce(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    result = pipeline.master(fx["loop"], options(settings), paths, settings)
    manifest = json.loads((result.run_dir / "run.json").read_text())
    assert manifest["kind"] == "master" and manifest["id"] == result.run_dir.name
    assert manifest["track"] == {**manifest["track"], "slug": "loop", "genre": "hip-hop"}
    assert manifest["input_file"] == "input/loop.wav"
    assert manifest["input_sha256"] == audio_io.file_sha256(fx["loop"])
    assert manifest["settings"] == {
        "mode": "matchering",
        "reference": {
            "spec": "hip-hop/reference",
            "path": str(reference.path),
            "sha256": reference.report.sha256,
        },
        "target_lufs": -14.0,
        "ceiling_dbtp": -1.0,
        "pre_chain": None,
        "post_chain": None,
    }
    assert {"masterbench", "matchering", "numpy", "scipy"} <= set(manifest["lib_versions"])
    analysis = json.loads((result.run_dir / "analysis.json").read_text())
    assert set(analysis) == {"mix", "master", "reference"}
    loaded = pipeline.load_run(paths, result.manifest.id)
    assert loaded.manifest == result.manifest and loaded.reports == result.reports


def test_same_input_and_settings_give_the_same_master(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    first = pipeline.master(fx["loop"], options(settings), paths, settings)
    second = pipeline.master(fx["loop"], options(settings), paths, settings)
    assert first.run_dir != second.run_dir, "every master is a new run"
    for name in ("master_24", "master_16"):
        assert first.output(name).read_bytes() == second.output(name).read_bytes()


def test_sample_rate_and_mono_input_are_handled(
    paths: Paths, settings: Settings, reference: Reference, tmp_path: Path
) -> None:
    loop, _ = fixtures.drum_808_loop(sr=48000)
    mono = fixtures.write_wav(tmp_path / "mono48.wav", loop.mean(axis=1), 48000)
    result = pipeline.master(mono, options(settings, genre="house"), paths, settings)
    info = sf.info(result.output("master_24"))
    assert (info.samplerate, info.channels, info.frames) == (48000, 2, len(loop))
    assert result.manifest.track.genre == "house"
    assert "mono" in {flag.code for flag in result.reports["mix"].flags}
    assert result.reports["master"].loudness.lufs_i == pytest.approx(-14.0, abs=0.2)


def test_louder_target_is_reported_honestly(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    loud = MasterOptions("hip-hop/default", -8.0, -1.0)
    result = pipeline.master(fx["loop"], loud, paths, settings)
    master = result.reports["master"]
    assert master.loudness.true_peak_dbtp <= -1.0
    assert "heavy_limiting" in {flag.code for flag in master.flags}
    assert result.manifest.stats["max_reduction_db"] > pipeline.HEAVY_LIMITING_DB


def test_reference_loudness_can_be_kept(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    result = pipeline.master(
        fx["loop"], MasterOptions("hip-hop/default", None, -1.0), paths, settings
    )
    assert result.manifest.stats["gain_db"] == 0.0
    assert result.reports["master"].loudness.true_peak_dbtp <= -1.0
    assert result.reports["master"].loudness.lufs_i > -12.0


def test_engine_errors_are_clear(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    with pytest.raises(MasteringError, match="same file"):
        pipeline.master(fx["reference"], options(settings), paths, settings)
    with pytest.raises(LookupError):
        pipeline.load_run(paths, "20260101-000000-abcdef")
    with pytest.raises(LookupError):
        pipeline.load_run(paths, "*")


def test_analyze_only_run_carries_validation_flags(
    paths: Paths, settings: Settings, fx: dict[str, Path]
) -> None:
    result = pipeline.analyze(fx["clipped"], paths, settings, genre="hip-hop")
    assert result.manifest.kind == "analysis_only" and result.manifest.outputs == {}
    assert set(result.reports) == {"mix"}
    assert {"clipping", "already_mastered"} <= {flag.code for flag in result.flags}
    assert not list(result.run_dir.glob("*.wav"))


def test_runs_are_indexed(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    mastered = pipeline.master(fx["loop"], options(settings), paths, settings)
    pipeline.analyze(fx["clipped"], paths, settings)
    with index.open_index(paths) as conn:
        runs = index.recent_runs(conn)
        assert {(r["slug"], r["kind"]) for r in runs} == {
            ("loop", "master"),
            ("clipped", "analysis_only"),
        }
        assert [r["id"] for r in index.recent_runs(conn, kind="master")] == [mastered.manifest.id]
        row = conn.execute(
            "SELECT lufs_i, true_peak_dbtp FROM analyses WHERE run_id = ? AND role = 'master'",
            (mastered.manifest.id,),
        ).fetchone()
        assert row["lufs_i"] == pytest.approx(-14.0, abs=0.2) and row["true_peak_dbtp"] <= -1.0
        flagged = conn.execute("SELECT DISTINCT code FROM flags").fetchall()
        assert "clipping" in {r["code"] for r in flagged}
        assert index.input_already_run(conn, audio_io.file_sha256(fx["clipped"]), "analysis_only")
        assert not index.input_already_run(conn, audio_io.file_sha256(fx["loop"]), "analysis_only")


def test_deleting_the_database_and_reindexing_restores_it(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    pipeline.master(fx["loop"], options(settings), paths, settings)
    pipeline.master(fx["loop"], MasterOptions("hip-hop/default", -11.0, -1.5), paths, settings)
    pipeline.analyze(fx["clipped"], paths, settings, genre="hip-hop")
    with index.open_index(paths) as conn:
        index.sync_references(conn, paths)
    before = dump(paths)
    assert len(before["runs"]) == 3 and len(before["references"]) == 1 and before["flags"]
    assert len(before["band_deltas"]) == 7 * 5, "mix and master of two runs, mix of the third"

    for leftover in paths.root.glob("masterbench.db*"):
        leftover.unlink()
    with index.open_index(paths) as conn:
        counts = index.reindex(conn, paths)
    assert counts == {"tracks": 2, "runs": 3, "references": 1, "blind tests": 0}
    assert dump(paths) == before

    with index.open_index(paths) as conn:
        assert index.reindex(conn, paths) == counts, "reindex is repeatable"
    assert dump(paths) == before


def test_reindex_drops_tracks_whose_folder_is_gone(
    paths: Paths, settings: Settings, fx: dict[str, Path]
) -> None:
    pipeline.analyze(fx["loop"], paths, settings)
    doomed = pipeline.analyze(fx["clipped"], paths, settings)
    shutil.rmtree(doomed.run_dir.parent.parent)
    with index.open_index(paths) as conn:
        counts = index.reindex(conn, paths)
        assert counts == {"tracks": 1, "runs": 1, "references": 0, "blind tests": 0}
        assert [r["slug"] for r in conn.execute("SELECT slug FROM tracks")] == ["loop"]


def test_reports_compare_frequency_balance_with_the_reference(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    result = pipeline.master(fx["loop"], options(settings), paths, settings)
    mix, master = result.reports["mix"], result.reports["master"]
    assert mix.band_deltas is not None and master.band_deltas is not None
    assert set(mix.band_deltas) == {
        "sub",
        "bass",
        "low_mids",
        "mids",
        "upper_mids",
        "presence",
        "air",
    }
    worst = lambda deltas: max(abs(v) for v in deltas.values())  # noqa: E731
    assert worst(master.band_deltas) < worst(mix.band_deltas), "matching moves toward the reference"
    assert result.reports["reference"].band_deltas is None
    assert mix.spectrum is not None and mix.stereo is not None and mix.short_term_lufs

    analyzed = pipeline.analyze(fx["pink_noise"], paths, settings, genre="hip-hop")
    assert set(analyzed.reports) == {"mix", "reference"}, "compared with the genre default"
    assert analyzed.manifest.settings["reference"]["spec"] == "hip-hop/reference"
    codes = {flag.code for flag in analyzed.flags}
    assert {"band_sub_under", "band_mids_over"} <= codes
    assert pipeline.analyze(fx["pink_noise"], paths, settings, genre="house").reports.keys() == {
        "mix"
    }
