from __future__ import annotations

import shutil
from pathlib import Path

from typer.testing import CliRunner

from masterbench import db
from masterbench.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0 and result.stdout.strip() == "0.1.0"


def test_init_creates_data_root_and_database(data_root: Path) -> None:
    first = runner.invoke(app, ["init"])
    assert first.exit_code == 0, first.output
    assert f"schema version {len(db.load_migrations())}" in first.stdout
    assert (data_root / "Inbox" / "electronic").is_dir()

    conn = db.connect(data_root / "masterbench.db")
    assert db.applied_versions(conn) == [m.version for m in db.load_migrations()]
    conn.close()

    second = runner.invoke(app, ["init"])
    assert second.exit_code == 0 and second.stdout == first.stdout


def test_fixtures_command(tmp_path: Path) -> None:
    result = runner.invoke(app, ["fixtures", str(tmp_path / "fx")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "fx" / "loop.wav").is_file()
    assert (tmp_path / "fx" / "truth.json").is_file()


def test_master_workflow(data_root: Path, fx: dict[str, Path]) -> None:
    added = runner.invoke(app, ["ref", "add", str(fx["reference"]), "--genre", "hip-hop"])
    assert added.exit_code == 0, added.output
    assert added.stdout.startswith("hip-hop/reference (default): -8.7 LUFS")
    listed = runner.invoke(app, ["ref", "list"])
    assert listed.stdout.startswith("* hip-hop/reference")

    mastered = runner.invoke(app, ["master", str(fx["loop"])])
    assert mastered.exit_code == 0, mastered.output
    rows = {line.split()[0]: line.split() for line in mastered.stdout.splitlines() if line.strip()}
    assert rows["Master"][1] == "-14.0" and rows["Mix"][1] == "-19.7"
    assert "35 cents sharp of C2" in mastered.stdout, "the loop's planted detuned 808 note"
    assert "Key F minor" in mastered.stdout and "master_24.wav" in mastered.stdout
    assert len(list(data_root.glob("projects/loop/runs/*/master_16.wav"))) == 1

    louder = runner.invoke(app, ["master", str(fx["loop"]), "--lufs", "-11", "--ceiling", "-1.5"])
    assert louder.exit_code == 0 and " -11.0 " in louder.stdout

    analyzed = runner.invoke(app, ["analyze", str(fx["clipped"]), "--genre", "hip-hop"])
    assert analyzed.exit_code == 0 and "Clipping" in analyzed.stdout

    (data_root / "masterbench.db").unlink()
    reindexed = runner.invoke(app, ["reindex"])
    assert reindexed.stdout.strip() == "2 tracks, 3 runs, 1 references, 0 blind tests, 0 notes"

    run_id = min(p.parent.name for p in data_root.glob("projects/loop/runs/*/master_24.wav"))
    previewed = runner.invoke(app, ["preview", run_id])
    assert previewed.exit_code == 0, previewed.output
    assert "Ogg Vorbis" in previewed.stdout
    assert "Spotify (Normal) plays this unchanged." in previewed.stdout
    assert runner.invoke(app, ["preview", "20260101-000000-abcdef"]).exit_code == 1


def test_errors_exit_nonzero_with_a_message(data_root: Path, fx: dict[str, Path]) -> None:
    no_reference = runner.invoke(app, ["master", str(fx["loop"]), "--genre", "house"])
    assert no_reference.exit_code == 1
    assert "no reference 'default' in genre 'house'" in no_reference.output
    missing = runner.invoke(app, ["master", str(data_root / "nope.wav")])
    assert missing.exit_code == 2
    assert runner.invoke(app, ["ref", "default", "house", "nope"]).exit_code == 1
    assert runner.invoke(app, ["ref", "remove", "house", "nope"]).exit_code == 1


def test_watch_once_processes_the_inbox(data_root: Path, fx: dict[str, Path], monkeypatch) -> None:
    runner.invoke(app, ["init"])
    (data_root / "settings.toml").write_text("watch_stable_s = 0.2\n", encoding="utf-8")
    shutil.copy(fx["vocal"], data_root / "Inbox" / "house" / "Hook idea.wav")
    result = runner.invoke(app, ["watch", "--once"])
    assert result.exit_code == 0, result.output
    assert "Analyzed: Hook idea: Mix: -17.1 LUFS" in result.stdout
    assert len(list(data_root.glob("projects/hook-idea/runs/*/run.json"))) == 1


def test_presets_and_chains_from_the_command_line(data_root: Path, fx: dict[str, Path]) -> None:
    listed = runner.invoke(app, ["preset", "list"])
    assert (
        listed.exit_code == 0
        and "gentle-glue        highpass > compressor > high_shelf" in listed.stdout
    )
    shown = runner.invoke(app, ["preset", "show", "warm"])
    assert 'type = "saturation"' in shown.stdout
    assert runner.invoke(app, ["preset", "show", "nope"]).exit_code == 1

    runner.invoke(app, ["ref", "add", str(fx["reference"]), "--genre", "hip-hop"])
    mastered = runner.invoke(
        app, ["master", str(fx["loop"]), "--pre", "streaming-safe", "--preset", "warm"]
    )
    assert mastered.exit_code == 0, mastered.output
    assert "Running the post-chain (warm)" in mastered.output
    broken = runner.invoke(app, ["master", str(fx["loop"]), "--post", "nope"])
    assert broken.exit_code == 1 and "no chain preset named 'nope'" in broken.output
    assert runner.invoke(app, ["plugins"]).exit_code == 0


def test_init_home_is_remembered_and_a_new_index_fills_itself(
    tmp_path: Path, fx: dict[str, Path]
) -> None:
    synced = tmp_path / "Mobile Documents" / "com~apple~CloudDocs" / "MasterBench"
    first = runner.invoke(app, ["init", "--home", str(synced)])
    assert first.exit_code == 0, first.output
    assert "index is kept on this machine only" in first.stdout
    assert not list(synced.glob("*.db*")), "no database inside the synced folder"

    analyzed = runner.invoke(app, ["analyze", str(fx["loop"])])
    assert analyzed.exit_code == 0 and list(synced.glob("projects/loop/runs/*/run.json"))

    for database in (tmp_path / "state").glob("index-*.db*"):
        database.unlink()
    again = runner.invoke(app, ["init"])
    assert "1 runs indexed" in again.stdout, "a missing index is rebuilt from the files"
    assert runner.invoke(app, ["init", "--icloud"]).exit_code == 1, (
        "no iCloud Drive in the test home"
    )


def test_stems_from_the_command_line(data_root: Path, tmp_path: Path, fx: dict[str, Path]) -> None:
    from masterbench import fixtures

    stems, _ = fixtures.song_stems()
    names = {"drums": "Drums", "bass": "808 Bass", "other": "Pads", "vocals": "Lead Vox"}
    for role, audio in stems.items():
        fixtures.write_wav(tmp_path / "stems" / f"{names[role]}.wav", audio)
    runner.invoke(app, ["ref", "add", str(fx["reference"]), "--genre", "hip-hop"])

    imported = runner.invoke(
        app,
        [
            "stems",
            "import",
            str(tmp_path / "stems"),
            "--title",
            "Night Drive",
            "--genre",
            "hip-hop",
        ],
    )
    assert imported.exit_code == 0, imported.output
    assert (
        "lead-vox                 vocals" in imported.stdout and "Vocal vs beat" in imported.stdout
    )
    listed = runner.invoke(app, ["stems", "list", "night-drive"])
    assert "from Logic" in listed.stdout and "808-bass" in listed.stdout
    assert runner.invoke(app, ["stems", "list", "nope"]).stdout.strip() == "no stem sets"

    mastered = runner.invoke(app, ["rebalance", "night-drive", "--gain", "lead-vox=+2"])
    assert mastered.exit_code == 0, mastered.output
    assert "Master     -14.0" in mastered.stdout
    too_much = runner.invoke(app, ["rebalance", "night-drive", "--gain", "drums=-9"])
    assert too_much.exit_code == 1 and "limited to ±6 dB" in too_much.output
    assert runner.invoke(app, ["rebalance", "night-drive", "--gain", "drums"]).exit_code == 2
    assert runner.invoke(app, ["rebalance", "nope", "--gain", "drums=1"]).exit_code == 1


def test_batch_and_notes_payload_from_the_command_line(
    data_root: Path, tmp_path: Path, fx: dict[str, Path]
) -> None:
    runner.invoke(app, ["ref", "add", str(fx["reference"]), "--genre", "hip-hop"])
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFF nope")
    done = runner.invoke(app, ["batch", str(fx["loop"]), str(fx["sine_sweep"]), "--name", "EP"])
    assert done.exit_code == 0, done.output
    assert "Loudness spread" in done.stdout and "sine_sweep" in done.stdout
    failed = runner.invoke(app, ["batch", str(fx["loop"]), str(broken)])
    assert failed.exit_code == 1 and "broken.wav failed" in failed.stdout

    run_id = min(p.parent.name for p in data_root.glob("projects/loop/runs/*/master_24.wav"))
    shown = runner.invoke(app, ["notes", run_id, "--payload", "--intent", "Punchy."])
    assert shown.exit_code == 0, shown.output
    assert '"intent": "Punchy."' in shown.stdout and "claude-sonnet-5-5" in shown.output
    assert runner.invoke(app, ["notes", "20260101-000000-abcdef", "--payload"]).exit_code == 1


def test_deep_analysis_from_the_command_line(data_root: Path, tmp_path: Path) -> None:
    from masterbench import fixtures

    refs = {"a": -7.0, "b": -5.0}
    for name, verse_db in refs.items():
        song = fixtures.arranged_song(verse_db=verse_db, seed=ord(name))[0]
        added = runner.invoke(
            app,
            [
                "ref",
                "add",
                str(fixtures.write_wav(tmp_path / f"{name}.wav", song)),
                "--genre",
                "hip-hop",
            ],
        )
        assert "quality: good" in added.stdout
    listed = runner.invoke(app, ["ref", "list"])
    assert "good" in listed.stdout and listed.stdout.startswith("* hip-hop/a")

    built = runner.invoke(app, ["curve", "build", "hip-hop"])
    assert built.exit_code == 0, built.output
    assert "hip-hop: built from 2 references" in built.stdout and "Hook minus verse" in built.stdout
    assert "sub" in runner.invoke(app, ["curve", "show", "hiphop"]).stdout
    assert runner.invoke(app, ["curve", "show", "house"]).exit_code == 1
    assert runner.invoke(app, ["curve", "build", "house"]).exit_code == 1

    weak = fixtures.write_wav(tmp_path / "Weak Hook.wav", fixtures.arranged_song(verse_db=-1.0)[0])
    suggested = runner.invoke(app, ["ref", "suggest", str(weak), "--genre", "hip-hop"])
    assert suggested.exit_code == 0 and suggested.stdout.count("hip-hop/") == 2

    mastered = runner.invoke(app, ["master", str(weak), "--curve", "--genre", "hip-hop"])
    assert mastered.exit_code == 0, mastered.output
    assert "Matching to the hip-hop curve" in mastered.output
    assert (
        "Key F minor" in mastered.stdout
        and "Sections (auto): intro 0:00, verse 1" in mastered.stdout
    )
    assert "It may not lift when it arrives." in mastered.stdout
    assert mastered.stdout.count("may not lift") == 1

    run_id = next(data_root.glob("projects/weak-hook/runs/*/master_24.wav")).parent.name
    shown = runner.invoke(app, ["sections", run_id])
    assert (
        "hook 1" in shown.stdout
        and "Hook minus verse: 1." in shown.stdout
        and "To edit: --set" in shown.stdout
    )
    saved = runner.invoke(app, ["sections", run_id, "--set", "0 intro, 6.9 verse, 20.6 hook"])
    assert saved.stdout.startswith("Saved: 0 intro, 6.9 verse, 20.6 hook")
    assert runner.invoke(app, ["sections", run_id, "--set", "5 solo"]).exit_code == 1
    assert "cleared" in runner.invoke(app, ["sections", run_id, "--clear"]).stdout

    assert "key set to F minor" in runner.invoke(app, ["key", "weak-hook", "f minor"]).stdout
    assert runner.invoke(app, ["key", "weak-hook", "H minor"]).exit_code == 1
    assert runner.invoke(app, ["key", "weak-hook"]).exit_code == 2
    assert "key cleared" in runner.invoke(app, ["key", "weak-hook", "--clear"]).stdout

    low = runner.invoke(app, ["lowend", run_id])
    assert low.exit_code == 0, low.output
    assert "C2" in low.stdout and "+35" in low.stdout and "35 cents sharp of C2" in low.stdout
    assert "needs drum and bass stems" in low.stdout

    sims = runner.invoke(app, ["speakers", run_id])
    assert sims.exit_code == 0, sims.output
    assert "Phone speaker: On this speaker the sub and bass are gone" in sims.stdout
    assert "approximations" in sims.stdout
    assert len(list(data_root.glob("projects/weak-hook/runs/*/previews/speaker_*.wav"))) == 6
