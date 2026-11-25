from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from masterbench import batch, config, fixtures, mcp_server, summary, watch
from masterbench.cli import app
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.pipeline import MasterOptions
from masterbench.references import library, suggest, target_curve
from masterbench.references.library import ReferenceError
from masterbench.vocal import mix_assistant
from masterbench.vocal.mix_assistant import MixError, MixOptions

runner = CliRunner()
BOOM, TRAP = "hip-hop/boom-bap ref", "hip-hop/trap ref"


def styled(kind: str, seed: int, drive_db: float = 0.0) -> np.ndarray:
    """Two feels of the loop: slow, light and dull against fast, sub-heavy and bright."""
    slow = kind == "boom"
    audio, _ = fixtures.drum_808_loop(
        tempo_bpm=92.0 if slow else 142.0,
        seed=seed,
        hat_gain=0.05 if slow else 0.28,
        bass_gain=0.35 if slow else 1.1,
        detune=False,
    )
    if drive_db:
        audio = fixtures._normalize_peak(np.tanh(audio * fixtures.db_to_lin(drive_db)), -1.0)
    return audio


@pytest.fixture(scope="module")
def audio_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("styles")
    made = {
        "boom_ref": ("boom-bap ref.wav", styled("boom", 7, 12.0)),
        "trap_ref": ("trap ref.wav", styled("trap", 8, 12.0)),
        "boom_mix": ("my boom mix.wav", styled("boom", 21)),
        "trap_mix": ("my trap mix.wav", styled("trap", 22)),
    }
    return {key: fixtures.write_wav(folder / name, audio) for key, (name, audio) in made.items()}


@pytest.fixture
def bank(paths: Paths, settings: Settings, audio_files: dict[str, Path]) -> Paths:
    """A hip-hop bank with a boom-bap reference (the default) and a trap reference."""
    library.add(paths, audio_files["boom_ref"], "hip-hop", settings)
    library.add(paths, audio_files["trap_ref"], "hip-hop", settings)
    return paths


def test_auto_specs_and_reserved_names(paths: Paths, settings: Settings, tmp_path: Path) -> None:
    assert library.auto_genre("auto") == "" and library.auto_genre("AUTO") == ""
    assert library.auto_genre("hip-hop/auto") == "hip-hop"
    for spec in ("hip-hop/default", "hip-hop/automatic", "/auto", "", "hip-hop"):
        assert library.auto_genre(spec) is None, spec
    real = tmp_path / "auto"
    real.write_bytes(b"RIFF")
    assert library.auto_genre(str(real)) is None, "a file that happens to be called auto is a file"

    assert library.default_spec(settings, "house") == "house/default"
    assert library.default_spec(settings, None) == "hip-hop/default"
    closest = replace(settings, default_reference="auto")
    assert library.default_spec(closest, None) == "hip-hop/auto"
    with pytest.raises(ReferenceError, match="must be 'default' or 'auto', not 'best'"):
        library.default_spec(replace(settings, default_reference="best"), "house")
    with pytest.raises(ReferenceError, match="'auto' picks a reference for each mix"):
        library.resolve(paths, "hip-hop/auto", settings)
    for name in ("auto.wav", "Default.wav"):
        file = fixtures.write_wav(tmp_path / name, fixtures.pink_noise(6.0))
        with pytest.raises(ReferenceError, match="cannot be called"):
            library.add(paths, file, "hip-hop", settings)
    (paths.root / "settings.toml").write_text('default_reference = "auto"\n')
    assert config.load_settings().default_reference == "auto"


def test_the_closest_reference_is_chosen_for_each_mix(
    bank: Paths, settings: Settings, audio_files: dict[str, Path]
) -> None:
    for mix, wanted, other in (("boom_mix", BOOM, TRAP), ("trap_mix", TRAP, BOOM)):
        report = library.analyze(audio_files[mix], settings)[0]
        choice = suggest.choose(bank, report, "hip-hop")
        assert choice.reference.spec == wanted and choice.genre == "hip-hop"
        assert choice.similarity is not None and choice.similarity > 0.8
        assert choice.others == [(other, choice.others[0][1])] and choice.others[0][1] < 0.3
        assert "tempo" in choice.why
        assert choice.to_dict() == {
            "asked": "hip-hop/auto",
            "chosen": wanted,
            "similarity": choice.similarity,
            "why": choice.why,
            "others": [{"reference": other, "similarity": choice.others[0][1]}],
        }
    assert suggest.choose(bank, report, "Hip Hop").reference.spec == TRAP, "genre names are loose"


def test_choice_handles_small_and_doubtful_banks(
    paths: Paths, settings: Settings, audio_files: dict[str, Path], fx: dict[str, Path]
) -> None:
    mix = library.analyze(audio_files["trap_mix"], settings)[0]
    with pytest.raises(ReferenceError, match="no references in genre 'hip-hop' to choose from"):
        suggest.choose(paths, mix, "hip-hop")
    only = library.add(paths, audio_files["boom_ref"], "hip-hop", settings)
    alone = suggest.choose(paths, mix, "hip-hop")
    assert alone.reference == only and alone.why == "the only reference in the genre"
    assert alone.others == []
    with pytest.raises(ReferenceError, match="is this very file"):
        suggest.choose(paths, only.report, "hip-hop")

    # A clipped copy of the trap reference is nearer to a trap mix than boom-bap is, but it
    # is badged poor, so it is passed over while a sound reference exists.
    clipped = fixtures.write_wav(
        audio_files["trap_ref"].parent / "trap clipped.wav",
        np.clip(styled("trap", 8) * 8.0, -1.0, 1.0),
    )
    poor = library.add(paths, clipped, "hip-hop", settings)
    assert poor.quality is not None and poor.quality.badge == "poor"
    assert suggest.choose(paths, mix, "hip-hop").reference == only
    library.remove(paths, "hip-hop", only.name)
    assert suggest.choose(paths, mix, "hip-hop").reference.name == "trap clipped", (
        "better than none"
    )


@pytest.mark.parametrize("spec", ["auto", "hip-hop/auto"])
def test_mastering_with_auto_uses_and_records_the_pick(
    bank: Paths, settings: Settings, audio_files: dict[str, Path], spec: str
) -> None:
    stages: list[str] = []
    options = MasterOptions.from_settings(settings, spec)
    run = pipeline.master(
        audio_files["trap_mix"], options, bank, settings, lambda stage, _: stages.append(stage)
    )
    used = run.manifest.settings
    assert used["reference"]["spec"] == TRAP, "not the genre default, which is boom-bap"
    assert (
        used["reference_choice"]["chosen"] == TRAP
        and used["reference_choice"]["asked"] == "hip-hop/auto"
    )
    assert run.manifest.track.genre == "hip-hop" and "Choosing the closest reference" in stages
    assert Path(run.reports["reference"].source_path).name == "trap ref.wav"
    assert run.reports["master"].loudness.lufs_i == pytest.approx(-14.0, abs=0.3)
    stored = json.loads((run.run_dir / "run.json").read_text())
    assert stored["settings"]["reference_choice"] == used["reference_choice"]
    line = summary.reference_line(run)
    assert line is not None and line.startswith(f"Reference: {TRAP}, picked automatically (closest")
    assert f"the others: {BOOM}" in line and line in summary.as_text(run)
    assert line in summary.as_markdown(run)
    assert mcp_server.run_summary(run)["reference_choice"]["chosen"] == TRAP


def test_named_and_default_references_behave_as_before(
    bank: Paths, settings: Settings, audio_files: dict[str, Path]
) -> None:
    default = pipeline.master(
        audio_files["trap_mix"],
        MasterOptions.from_settings(settings, "hip-hop/default"),
        bank,
        settings,
    )
    assert default.manifest.settings["reference"]["spec"] == BOOM
    assert "reference_choice" not in default.manifest.settings
    assert summary.reference_line(default) is None
    analyzed = pipeline.analyze(audio_files["trap_mix"], bank, settings, "hip-hop")
    assert analyzed.manifest.settings["reference"]["spec"] == BOOM
    assert "reference_choice" not in analyzed.manifest.settings
    for spec, message in (
        ("house/auto", "no references in genre 'house'"),
        ("auto", "genre 'polka'"),
    ):
        wanted = MasterOptions.from_settings(
            settings, spec, genre="polka" if spec == "auto" else None
        )
        with pytest.raises(ReferenceError, match=message):
            pipeline.master(audio_files["trap_mix"], wanted, bank, settings)
    assert len(list((bank.projects / "my-trap-mix" / "runs").iterdir())) == 2, "no third run"


def test_auto_as_the_default_reaches_analysis_curves_watch_and_batch(
    bank: Paths, settings: Settings, audio_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    closest = replace(settings, default_reference="auto", watch_auto_master=True)
    analyzed = pipeline.analyze(audio_files["trap_mix"], bank, closest, "hip-hop")
    assert analyzed.manifest.settings["reference"]["spec"] == TRAP
    assert analyzed.manifest.settings["reference_choice"]["chosen"] == TRAP
    untagged = pipeline.analyze(audio_files["boom_mix"], bank, closest, title="No genre yet")
    assert untagged.reports.keys() == {"mix"}, "without a genre there is no bank to choose from"
    measured = pipeline.measure(audio_files["boom_mix"], bank, closest, "hip-hop")
    assert Path(measured["reference"].source_path).name == "boom-bap ref.wav"

    target_curve.build(bank, "hip-hop")
    by_curve = MasterOptions.from_settings(closest, "hip-hop/auto", mode="curve", genre="hip-hop")
    curved = pipeline.master(audio_files["trap_mix"], by_curve, bank, closest)
    assert curved.manifest.settings["mode"] == "curve"
    assert curved.manifest.settings["reference"]["spec"] == TRAP, "compared with the closest"
    named = replace(by_curve, reference="hip-hop/default")
    assert (
        pipeline.master(audio_files["trap_mix"], named, bank, closest).manifest.settings[
            "reference"
        ]["spec"]
        == BOOM
    )

    monkeypatch.setattr(watch.notify, "notify", lambda *args, **kwargs: None)
    dropped = fixtures.write_wav(bank.inbox / "hip-hop" / "new boom idea.wav", styled("boom", 33))
    event = watch.process_file(dropped, bank.inbox, bank, closest)
    assert event.mastered is not None
    assert event.mastered.manifest.settings["reference_choice"]["chosen"] == BOOM
    assert event.analysis is not None
    assert event.analysis.manifest.settings["reference_choice"]["chosen"] == BOOM

    sources = [audio_files["boom_mix"], audio_files["trap_mix"]]
    options = MasterOptions.from_settings(closest, library.default_spec(closest, "hip-hop"))
    done = batch.run_batch(sources, options, bank, closest, "Two feels")
    picked = [run.manifest.settings["reference_choice"]["chosen"] for run in done.runs]
    assert picked == [BOOM, TRAP], "each track gets its own closest reference"
    assert summary.batch_picks(done) == [
        f"my boom mix: reference {BOOM}, picked automatically (closest key and tempo).",
        f"my trap mix: reference {TRAP}, picked automatically (closest key and tempo).",
    ]
    assert summary.batch_picks(done)[0] in summary.batch_text(done)
    assert f"- {summary.batch_picks(done)[1]}" in summary.batch_markdown(done)


def test_vocal_mix_masters_against_the_closest_reference(
    bank: Paths, settings: Settings, tmp_path: Path
) -> None:
    vocal, beat, _ = fixtures.vocal_over_song()
    files = (
        fixtures.write_wav(tmp_path / "vox.wav", vocal),
        fixtures.write_wav(tmp_path / "Song.wav", beat),
    )
    result = mix_assistant.mix(
        bank, settings, *files, MixOptions(genre="hip-hop", reference="auto")
    )
    assert result.mastered
    choice = result.run.manifest.settings["reference_choice"]
    assert (
        choice["chosen"] in (BOOM, TRAP)
        and choice["chosen"] == result.run.manifest.settings["reference"]["spec"]
    )
    assert result.reference_ratio_db is None and result.target_source == "the default for hip-hop"
    with pytest.raises(MixError, match="'auto' is only picked once the mix exists"):
        mix_assistant.target_for(
            bank,
            settings,
            MixOptions(genre="hip-hop", reference="auto", target_from_reference=True),
            "hip-hop",
        )
    empty = mix_assistant.mix(
        bank, settings, *files, MixOptions(genre="house", reference="house/auto")
    )
    assert not empty.mastered, "no references in house: analyzed, not mastered"


def test_command_line_auto_and_the_setting(data_root: Path, audio_files: dict[str, Path]) -> None:
    for key in ("boom_ref", "trap_ref"):
        assert (
            runner.invoke(
                app, ["ref", "add", str(audio_files[key]), "--genre", "hip-hop"]
            ).exit_code
            == 0
        )
    mix = str(audio_files["trap_mix"])
    made = runner.invoke(app, ["master", mix, "--ref", "auto", "--genre", "hip-hop"])
    assert made.exit_code == 0, made.output
    assert f"Reference: {TRAP}, picked automatically (closest key and tempo)." in made.stdout
    assert "Reference:" not in runner.invoke(app, ["master", mix, "--genre", "hip-hop"]).stdout

    (data_root / "settings.toml").write_text('default_reference = "auto"\n')
    analyzed = runner.invoke(app, ["analyze", mix, "--genre", "hip-hop"])
    assert f"Reference: {TRAP}, picked automatically" in analyzed.stdout
    assert f"Reference: {TRAP}" in runner.invoke(app, ["master", mix, "--genre", "hip-hop"]).stdout
    named = runner.invoke(app, ["master", mix, "--ref", "hip-hop/default"])
    assert named.exit_code == 0 and "picked automatically" not in named.stdout, "a name still wins"

    failed = runner.invoke(app, ["master", mix, "--ref", "house/auto"])
    assert (
        failed.exit_code == 1 and "no references in genre 'house' to choose from" in failed.output
    )
    (data_root / "settings.toml").write_text('default_reference = "nearest"\n')
    failed = runner.invoke(app, ["master", mix, "--genre", "hip-hop"])
    assert failed.exit_code == 1 and "must be 'default' or 'auto', not 'nearest'" in failed.output
