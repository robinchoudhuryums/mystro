from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from masterbench import audio_io, fixtures, index, progress
from masterbench.config import Paths, Settings, Thresholds
from masterbench.master import pipeline
from masterbench.progress import TrackPoint
from masterbench.stems import workflow as stem_workflow
from masterbench.vocal import pitch_track, presets, tuning, workflow
from masterbench.vocal.tuning import NoteEdit, TuningError
from masterbench.vocal.workflow import VocalError

SR = fixtures.SR
needs_rubberband = pytest.mark.skipif(shutil.which("rubberband") is None, reason="no rubberband")


@pytest.fixture
def source(paths: Paths, fx: dict[str, Path], tmp_path: Path):
    """The fixture vocal, brought in as the dry vocal of 'Night Drive'."""
    file = tmp_path / "Lead Vox.wav"
    shutil.copy(fx["vocal"], file)
    return workflow.open_vocal(paths, file, track="Night Drive")


def world(paths: Paths, name: str = "natural", **changes):
    return presets.get(paths, name).settings("F minor", engine="world", **changes)


def test_a_vocal_is_copied_in_and_the_original_left_alone(
    paths: Paths, fx: dict[str, Path], tmp_path: Path
) -> None:
    file = tmp_path / "Lead Vox.wav"
    shutil.copy(fx["vocal"], file)
    before = audio_io.file_sha256(file)
    source = workflow.open_vocal(paths, file, track="Night Drive")
    folder = paths.projects / "night-drive" / "vocal" / before[:8]
    assert source.folder == folder and source.path == folder / "Lead Vox.wav"
    assert (source.id, source.kind, source.label) == (
        f"night-drive/{before[:8]}",
        "dry",
        "Lead Vox",
    )
    assert source.track.title == "Night Drive" and source.mix is None and not source.separated
    assert audio_io.file_sha256(file) == before == audio_io.file_sha256(source.path)

    again = workflow.open_vocal(paths, file, track="night-drive")
    assert again == source, "the same file for the same track is the same vocal"
    assert workflow.load_source(paths, source.id) == source
    assert workflow.list_sources(paths) == [source] == workflow.list_sources(paths, "night-drive")
    alone = workflow.open_vocal(paths, file)
    assert alone.track.slug == "lead-vox", "without a track, the file names one"
    for bad in ("night-drive/nope", "../x/y", "nope"):
        with pytest.raises(VocalError, match="no vocal"):
            workflow.load_source(paths, bad)
    with pytest.raises(VocalError, match="no such file"):
        workflow.open_vocal(paths, tmp_path / "missing.wav")


def test_report_is_stored_indexed_and_uses_the_track_key(
    paths: Paths, settings: Settings, fx: dict[str, Path], source
) -> None:
    guessed = workflow.report(source, paths)
    assert (guessed.key, guessed.key_source) == ("F minor", "detected")
    assert workflow.stored_report(source) == guessed
    assert (source.folder / "pitch.npy").is_file(), "the pitch track is kept for the next step"

    pipeline.analyze(fx["loop"], paths, settings, title="Night Drive")
    from_track = workflow.report(source, paths)
    assert (from_track.key, from_track.key_source) == ("F minor", "track")
    assert from_track.key_confidence is not None and from_track.notes == guessed.notes
    given = workflow.report(source, paths, "G# major")
    assert (given.key, given.key_source, given.scale) == ("G# major", "given", "G# major")
    (paths.projects / "night-drive" / "overrides.json").write_text('{"key": "C minor"}')
    assert workflow.report(source, paths).key == "C minor", "a key set by hand comes first"

    with index.open_index(paths) as conn:
        row = dict(conn.execute("SELECT * FROM vocal_reports").fetchone())
        assert row["id"] == source.id and row["kind"] == "dry" and row["notes"] == 8
        assert row["key"] == "C minor" and row["source_sha256"] == source.sha256
        conn.execute("DELETE FROM vocal_reports")
        assert index.reindex(conn, paths)["vocal reports"] == 1
        point = next(p for p in progress.track_points(conn) if p.slug == "night-drive")
    assert point.vocal_in_tune_share == row["in_tune_share"] is not None
    assert point.vocal_mean_cents == row["mean_cents"]


def test_tuned_file_drops_into_the_session(paths: Paths, source) -> None:
    version = workflow.tune(source, world(paths), label="Natural")
    file = source.folder / version.file
    assert version.file == f"tuned_{version.id}.wav" and len(version.id) == 10
    assert version.id == tuning.settings_hash(world(paths), {})
    original, tuned = sf.info(source.path), sf.info(file)
    assert (tuned.frames, tuned.samplerate, tuned.channels, tuned.subtype) == (
        original.frames,
        original.samplerate,
        original.channels,
        "PCM_24",
    )
    before, after = audio_io.load(source.path)[0], audio_io.load(file)[0]
    assert np.array_equal(before[: int(0.2 * SR)], after[: int(0.2 * SR)]), "same start"
    assert 0.05 < np.mean(before == after) < 0.5, "silence and gaps identical, notes resynthesized"

    assert (version.notes, version.corrected) == (8, 2) and 25 < version.largest_cents < 30
    assert version.before["in_tune_share"] == pytest.approx(0.75, abs=0.01)
    assert version.after["in_tune_share"] == 1.0 and (version.after["mean_abs_cents"] or 9) < 2.5
    assert version.label == "Natural (world, 90%)" and version.flags == [] and not version.cached
    assert abs(version.lufs - workflow.candidates(source)[0]["lufs"]) < 0.1
    assert version.preview_file is None

    again = workflow.tune(source, world(paths), label="Something else")
    assert again.cached and replace(again, cached=False) == version, "same settings: same file"
    assert workflow.tuned_versions(source) == [version]
    file.unlink()
    assert workflow.tuned_versions(source) == [], "a version whose audio is gone is not listed"


def test_edits_are_kept_with_the_vocal_and_used(paths: Paths, source) -> None:
    assert workflow.read_edits(source) == {}
    kept = workflow.write_edits(
        source, {5: NoteEdit(bypass=True), 2: NoteEdit(target=61.0), 0: NoteEdit()}
    )
    assert kept == {2: NoteEdit(target=61.0), 5: NoteEdit(bypass=True)}, "an empty edit is dropped"
    assert workflow.read_edits(source) == kept
    assert json.loads((source.folder / "edits.json").read_text()) == {
        "2": {"bypass": False, "target": 61.0},
        "5": {"bypass": True, "target": None},
    }
    edited = workflow.tune(source, world(paths), label="Natural")
    assert edited.edits == kept and edited.label == "Natural (world, 90%, 2 edits)"
    assert 60 < edited.largest_cents < 70, "C4, 30 cents sharp, sent up to C#4"
    plain = workflow.tune(source, world(paths), edits={}, label="Natural")
    assert plain.id != edited.id and plain.edits == {}
    workflow.write_edits(source, {40: NoteEdit(bypass=True)})
    with pytest.raises(TuningError, match="no note number 41"):
        workflow.tune(source, world(paths))


def test_vocal_stem_of_a_track_is_used_in_place(paths: Paths, tmp_path: Path) -> None:
    stems, _ = fixtures.song_stems()
    names = {"drums": "Drums", "bass": "808 Bass", "other": "Pads", "vocals": "Lead Vox"}
    for role, audio in stems.items():
        fixtures.write_wav(tmp_path / "stems" / f"{names[role]}.wav", audio)
    _, stem_set = stem_workflow.import_logic_stems(paths, tmp_path / "stems", "Night Drive")
    source = workflow.from_stems(paths, "night-drive")
    assert source.kind == "stem" and source.path == stem_set.path("lead-vox")
    assert source.label == "lead-vox (Logic stem)" and source.mix is None
    assert not list(source.folder.glob("*.wav")), "a stem is read where it is, not copied"
    found = workflow.report(source, paths, "F minor")
    assert len(found.notes) >= 8 and found.flags == []
    with pytest.raises(VocalError, match="no vocal stem"):
        workflow.from_stems(paths, "night-drive", stem="pads-2")
    with pytest.raises(stem_workflow.StemError):
        workflow.from_stems(paths, "nope")


def test_full_mix_path_swaps_the_vocal_and_calls_it_a_preview(
    paths: Paths, settings: Settings, tmp_path: Path
) -> None:
    stems, mix = fixtures.song_stems()
    vocal = stems["vocals"]
    bounce = fixtures.write_wav(tmp_path / "Night Drive.wav", mix)

    def separator(staged: Path, model_file: str, out_dir: Path) -> list[Path]:
        """A perfect separation: the wrapper hands over the mix at its own working level."""
        audio, info = audio_io.load(staged)
        ours = vocal * (np.abs(audio).max() / np.abs(mix).max())
        parts = (("Vocals", ours), ("Instrumental", audio - ours))
        return [
            audio_io.save_wav(out_dir / f"x_({name})_m.wav", part, info.sample_rate, 32)
            for name, part in parts
        ]

    wanted = world(paths)
    source, version = workflow.tune_mix(paths, bounce, wanted, runner=separator)
    assert source.kind == "separated" and source.separated and source.track.slug == "night-drive"
    assert source.mix is not None and source.mix.parent == paths.projects / "night-drive" / "input"
    assert [f.code for f in version.flags][0] == "preview_quality"
    assert "export the dry vocal from Logic" in version.flags[0].message
    assert version.preview_file == f"preview_{version.id}.wav" and version.corrected >= 2

    full = audio_io.load(source.mix)[0]
    assert version.preview_file is not None
    preview, info = audio_io.load(source.folder / version.preview_file)
    tuned = audio_io.load(source.folder / version.file)[0]
    separated = audio_io.load(source.path)[0]
    assert preview.shape == full.shape and info.subtype == "FLOAT"
    assert np.allclose(preview - full, tuned - separated, atol=1e-6), "only the vocal changed"
    assert np.mean(np.abs(preview - full) < 1e-7) > 0.05, "where nothing was sung it is the mix"
    listed = [c["label"] for c in workflow.mix_candidates(source)]
    assert listed == ["Mix", "Mix, vocal tuned (world, preview)"]

    found = workflow.report(source, paths, "F minor")
    assert [f.code for f in found.flags] == ["separated_vocal"]
    assert workflow.load_source(paths, source.id) == source, "the mix is remembered"
    assert workflow.separate_vocal(paths, bounce, runner=separator) == source, "separated once"


def test_outside_versions_must_line_up(paths: Paths, source, tmp_path: Path) -> None:
    vocal, _ = fixtures.synthetic_vocal()
    waves = fixtures.write_wav(tmp_path / "bounce 3.wav", vocal * 0.5)
    stored = workflow.add_external(source, waves, "Waves Tune")
    assert stored == source.folder / "external" / "waves-tune.wav" and stored.is_file()
    workflow.add_external(source, waves, "Waves Tune")
    (other,) = workflow.externals(source)
    assert other["label"] == "Waves Tune" and other["lufs"] == pytest.approx(-23.2, abs=0.3)
    listed = workflow.candidates(source)
    assert [c["label"] for c in listed] == ["Original", "Waves Tune"]
    assert [c["id"] for c in listed] == ["original", "external_0"] and listed[1]["path"] == stored

    short = fixtures.write_wav(tmp_path / "short.wav", vocal[: 3 * SR])
    with pytest.raises(VocalError, match="export the same region"):
        workflow.add_external(source, short, "Flex Pitch")
    other_rate = fixtures.write_wav(tmp_path / "48k.wav", vocal, 48000)
    with pytest.raises(VocalError, match="export it at the same rate"):
        workflow.add_external(source, other_rate, "Flex Pitch")


@needs_rubberband
def test_benchmark_measures_both_engines(paths: Paths, source) -> None:
    rows = workflow.benchmark(source, paths, "F minor")
    assert [(r.preset, r.engine) for r in rows] == [
        ("natural", "rubberband"),
        ("natural", "world"),
        ("hard-tune", "rubberband"),
        ("hard-tune", "world"),
    ]
    for row in rows:
        assert row.in_tune_share is not None and row.in_tune_share > 0.97
        assert (row.mean_abs_cents or 9) < 3 and row.seconds > 0 and row.realtime_factor > 0.5
        assert row.timbre_change_db < 2.0
        natural = row.preset == "natural"
        assert row.vibrato_kept is not None
        assert (row.vibrato_kept > 0.9) if natural else (row.vibrato_kept < 0.2)
    by = {(r.preset, r.engine): r for r in rows}
    flattened = [by["hard-tune", engine].vibrato_kept or 0.0 for engine in ("world", "rubberband")]
    assert flattened[0] <= flattened[1], "WORLD follows a frame-by-frame target more exactly"
    assert by["natural", "rubberband"].timbre_change_db < by["natural", "world"].timbre_change_db
    stored = json.loads((source.folder / "benchmark.json").read_text())
    assert [row["tuned_id"] for row in stored] == [r.tuned_id for r in rows]
    labels = [c["label"] for c in workflow.candidates(source)]
    assert labels == [
        "Original",
        "Natural (rubberband, 90%)",
        "Natural (world, 90%)",
        "Hard-tune (rubberband, 100%)",
        "Hard-tune (world, 100%)",
    ]


def test_presets_packaged_and_your_own(paths: Paths) -> None:
    assert presets.names(paths) == ["natural", "tight", "hard-tune"]
    natural = presets.get(paths, "natural").settings("F minor")
    assert (natural.retune_ms, natural.humanize, natural.amount) == (100.0, 1.0, 0.9)
    tight = presets.get(paths, "tight").settings("F minor")
    assert 20 <= tight.retune_ms <= 40 and 0 < tight.humanize < 1
    hard = presets.get(paths, "hard-tune").settings("F minor", "dorian")
    assert (hard.retune_ms, hard.humanize, hard.amount, hard.engine) == (0.0, 0.0, 1.0, "world")
    assert hard.resolve_scale().label == "F dorian" and hard.preserve_formants

    eased = presets.get(paths, "natural").settings("F minor", amount=0.5, engine=None)
    assert eased.amount == 0.5 and eased.engine == natural.engine, "None leaves the preset's value"
    mine = presets.from_settings("mine", replace(hard, retune_ms=10.0, note_hold=True), "Mine.")
    saved = presets.save(paths, mine)
    assert saved == paths.root / "presets" / "tuning" / "mine.toml"
    assert presets.names(paths)[-1] == "mine" and presets.get(paths, "mine") == mine
    assert presets.get(paths, "mine").settings("C major").note_hold is True
    presets.save(paths, presets.from_settings("natural", hard))
    assert presets.get(paths, "natural").settings("C major").engine == "world", "yours wins"

    for name in ("Nope!", "../x"):
        with pytest.raises(TuningError, match="preset names use"):
            presets.get(paths, name)
    with pytest.raises(TuningError, match="no tuning preset named 'gone'"):
        presets.get(paths, "gone")
    with pytest.raises(TuningError, match="unknown settings: speed"):
        presets.from_toml("speed = 3\n", "x")
    with pytest.raises(TuningError, match="amount must be"):
        presets.from_toml("amount = 3\n", "x")
    with pytest.raises(TuningError, match="not valid TOML"):
        presets.from_toml("amount = = 3\n", "x")


def test_a_vocal_that_keeps_going_sharp_is_a_habit() -> None:
    def point(title: str, cents: float | None) -> TrackPoint:
        return TrackPoint(
            title.lower(), title, None, "2026-11-24", "r", -18.0, -3.0, 5.0, 12.0, 0.9, {}, [],
            vocal_mean_cents=cents, vocal_in_tune_share=None if cents is None else 0.8,
        )  # fmt: skip

    limits = Thresholds()
    mostly = [
        point(n, c) for n, c in zip("ABCDEF", (12.0, None, 9.0, 15.0, 2.0, None), strict=True)
    ]
    (habit,) = progress.habits(mostly, limits)
    assert habit.code == "vocal_sharp" and (habit.tracks, habit.considered) == (3, 4)
    assert habit.statement == (
        "The vocal is sharp on 3 of the last 4 tracks with a pitch report, by 12 cents on average."
    )
    flat = [point(n, -10.0) for n in "ABC"]
    assert [h.code for h in progress.habits(flat, limits)] == ["vocal_flat"]
    assert (
        progress.habits([point(n, c) for n, c in zip("ABC", (9.0, -9.0, 3.0), strict=True)], limits)
        == []
    )
    assert progress.habits([point(n, 20.0) for n in "AB"], limits) == []


def test_pitch_track_is_reused(source, monkeypatch: pytest.MonkeyPatch) -> None:
    first = workflow.pitch_of(source)
    monkeypatch.setattr(pitch_track, "track", lambda *a, **k: pytest.fail("tracked twice"))
    again = workflow.pitch_of(source)
    assert np.array_equal(first.f0_hz, again.f0_hz) and again.sample_rate == SR
