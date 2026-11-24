from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from masterbench import audio_io, fixtures
from masterbench.analysis import loudness
from masterbench.config import Paths, Settings, Thresholds
from masterbench.master.pipeline import MasterOptions
from masterbench.references.library import Reference
from masterbench.stems import rebalance as rebalancing
from masterbench.stems import separate, stem_analysis, workflow
from masterbench.stems.separate import StemError, StemSet

FILE_NAMES = {
    "drums": "Drums.wav",
    "bass": "808 Bass.wav",
    "other": "Pads.wav",
    "vocals": "Lead Vox.wav",
}


def write_song(folder: Path, sr: int = fixtures.SR, vocal_db: float = -6.0) -> tuple[Path, Path]:
    """A folder of Logic-style stems and the matching bounce."""
    stems, mix = fixtures.song_stems(sr=sr, vocal_db=vocal_db)
    for role, audio in stems.items():
        fixtures.write_wav(folder / "stems" / FILE_NAMES[role], audio, sr)
    return folder / "stems", fixtures.write_wav(folder / "Night Drive.wav", mix, sr)


@pytest.fixture(scope="module")
def song(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_song(tmp_path_factory.mktemp("song"))


@pytest.fixture
def logic_set(song: tuple[Path, Path], tmp_path: Path) -> StemSet:
    return separate.import_stems(song[0], tmp_path / "cache")


class FakeSeparator:
    """Stands in for the model: splits the staged input by fixed shares, at 44.1 kHz."""

    def __init__(self, shares: dict[str, float]) -> None:
        self.shares, self.calls, self.staged_peak = shares, 0, 0.0

    def __call__(self, source: Path, model_file: str, out_dir: Path) -> list[Path]:
        self.calls += 1
        audio, info = audio_io.load(source)
        self.staged_peak = float(np.abs(audio).max())
        assert info.subtype == "FLOAT" and info.channels == 2
        if info.sample_rate != 44100:
            audio = signal.resample_poly(audio, 44100, info.sample_rate, axis=0)
        written = []
        for stem, share in self.shares.items():
            path = out_dir / f"input_({stem.capitalize()})_{Path(model_file).stem}.wav"
            written.append(audio_io.save_wav(path, audio * share, 44100, 32))
        return written


@pytest.mark.parametrize(
    ("filename", "role"),
    [
        ("Lead Vox.wav", "vocals"),
        ("ADLIBS 2.wav", "vocals"),
        ("Kick.aif", "drums"),
        ("Hi-Hats_bip.wav", "drums"),
        ("808.wav", "bass"),
        ("Sub Bass.wav", "bass"),
        ("Beat.wav", "instrumental"),
        ("Piano.wav", "other"),
        ("Rhodes FX.wav", "other"),
    ],
)
def test_roles_are_guessed_from_file_names(filename: str, role: str) -> None:
    assert separate.guess_role(filename) == role


def test_logic_stems_import_as_an_aligned_set(logic_set: StemSet, song: tuple[Path, Path]) -> None:
    assert logic_set.source == "logic" and not logic_set.separated
    assert logic_set.roles == {
        "808-bass": "bass",
        "drums": "drums",
        "lead-vox": "vocals",
        "pads": "other",
    }
    mix, info = audio_io.load(song[1])
    assert (logic_set.sample_rate, logic_set.frames) == (info.sample_rate, len(mix))
    assert np.abs(separate.mixdown(logic_set) - mix).max() < 1e-5, "real stems sum to the bounce"
    assert separate.load_set(logic_set.dir) == logic_set
    assert logic_set.lufs is not None and logic_set.lufs["lead-vox"] < 0


def test_import_pads_short_stems_and_widens_mono(tmp_path: Path) -> None:
    sr = fixtures.SR
    fixtures.write_wav(tmp_path / "in" / "Vocal.wav", fixtures.pink_noise(2.0)[:, 0], sr)
    fixtures.write_wav(tmp_path / "in" / "Beat.wav", fixtures.pink_noise(3.0, seed=2), sr)
    (tmp_path / "in" / "notes.txt").write_text("ignore me")
    stem_set = separate.import_stems(tmp_path / "in", tmp_path / "cache")
    vocal = stem_set.load("vocal")
    assert vocal.shape == (3 * sr, 2) and np.array_equal(vocal[:, 0], vocal[:, 1])
    assert np.all(vocal[2 * sr :] == 0) and np.any(vocal[: 2 * sr] != 0)
    assert stem_set.roles == {"beat": "instrumental", "vocal": "vocals"}
    forced = separate.import_stems(tmp_path / "in", tmp_path / "c2", roles={"Beat.wav": "drums"})
    assert forced.roles["beat"] == "drums"


def test_import_rejects_folders_it_cannot_line_up(tmp_path: Path) -> None:
    folder = tmp_path / "in"
    fixtures.write_wav(folder / "Vocal.wav", fixtures.pink_noise(1.0), 44100)
    with pytest.raises(StemError, match="at least two audio files"):
        separate.import_stems(folder, tmp_path / "cache")
    fixtures.write_wav(folder / "Beat.wav", fixtures.pink_noise(1.0), 48000)
    with pytest.raises(StemError, match="different sample rates"):
        separate.import_stems(folder, tmp_path / "cache")
    fixtures.write_wav(folder / "Beat.wav", fixtures.pink_noise(1.0), 44100)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(folder / "Beat.wav")]
    subprocess.run([*cmd, str(folder / "Bass.mp3")], check=True)
    with pytest.raises(StemError, match="must be WAV, AIFF or FLAC"):
        separate.import_stems(folder, tmp_path / "cache")
    (folder / "Bass.mp3").unlink()
    with pytest.raises(StemError, match="unknown roles"):
        separate.import_stems(folder, tmp_path / "cache", roles={"Beat.wav": "guitar"})


def test_balance_of_real_stems_matches_the_stems(
    logic_set: StemSet, song: tuple[Path, Path]
) -> None:
    mix, _ = audio_io.load(song[1])
    result = stem_analysis.balance(logic_set, mix)
    assert result.reconstruction_db is not None and result.reconstruction_db < -90
    sr = logic_set.sample_rate
    beat = mix - logic_set.load("lead-vox")
    expected = loudness.integrated(logic_set.load("lead-vox"), sr) - loudness.integrated(beat, sr)
    assert result.vocal_to_instrumental_db == pytest.approx(expected, abs=0.02)
    assert result.bass_to_total_db == result.levels_db["808-bass"]
    assert result.drums_to_total_db is not None and result.drums_to_total_db < 0
    assert stem_analysis.artifact_warning(result) is None


def test_a_buried_vocal_is_flagged_against_the_reference(tmp_path: Path) -> None:
    quiet_dir, quiet_mix = write_song(tmp_path / "quiet", vocal_db=-12.0)
    ref_dir, ref_mix = write_song(tmp_path / "ref", vocal_db=-6.0)
    mine = stem_analysis.balance(
        separate.import_stems(quiet_dir, tmp_path / "c1"), audio_io.load(quiet_mix)[0]
    )
    theirs = stem_analysis.balance(
        separate.import_stems(ref_dir, tmp_path / "c2"), audio_io.load(ref_mix)[0]
    )
    flags = {f.code: f for f in stem_analysis.compare(mine, theirs, Thresholds())}
    low = flags["vocal_to_instrumental_low"]
    assert low.value == pytest.approx(-6.0, abs=0.3)
    assert "The vocal, relative to the beat, sits 6.0 dB lower than in the reference" in low.message
    assert stem_analysis.compare(theirs, theirs, Thresholds()) == []


@pytest.mark.parametrize("sr", [44100, 48000])
def test_separation_keeps_rate_length_and_level(tmp_path: Path, sr: int) -> None:
    _, mix = fixtures.song_stems(sr=sr)
    source = fixtures.write_wav(tmp_path / "mix.wav", mix, sr)
    fake = FakeSeparator({"vocals": 0.3, "instrumental": 0.7})
    stem_set = separate.separate(source, tmp_path / "cache", "mdx-2", fake)

    assert fake.staged_peak == pytest.approx(0.5, abs=1e-4), "staged quiet so nothing is rescaled"
    assert stem_set.separated and stem_set.source == "mdx-2" and stem_set.seconds is not None
    assert (stem_set.sample_rate, stem_set.frames) == (sr, len(mix))
    vocals, instrumental = stem_set.load("vocals"), stem_set.load("instrumental")
    assert vocals.shape == instrumental.shape == mix.shape
    balance = stem_analysis.balance(stem_set, mix)
    # At 48 kHz the round trip through the model's 44.1 kHz drops what lies above 22 kHz.
    limit = -80 if sr == 44100 else -30
    assert balance.reconstruction_db is not None and balance.reconstruction_db < limit
    assert balance.levels_db["vocals"] == pytest.approx(20 * np.log10(0.3), abs=0.1)
    assert balance.vocal_to_instrumental_db == pytest.approx(20 * np.log10(0.3 / 0.7), abs=0.1)
    assert "separated by a model" in (stem_analysis.artifact_warning(balance) or "")


def test_separation_is_cached_by_content_and_model(tmp_path: Path, fx: dict[str, Path]) -> None:
    fake = FakeSeparator({"vocals": 0.5, "instrumental": 0.5})
    first = separate.separate(fx["loop"], tmp_path / "cache", "mdx-2", fake)
    again = separate.separate(fx["loop"], tmp_path / "cache", "mdx-2", fake)
    assert fake.calls == 1 and again == first
    separate.separate(fx["loop"], tmp_path / "cache", "roformer-2", fake)
    separate.separate(fx["pink_noise"], tmp_path / "cache", "mdx-2", fake)
    assert fake.calls == 3
    first.path("vocals").unlink()
    separate.separate(fx["loop"], tmp_path / "cache", "mdx-2", fake)
    assert fake.calls == 4, "an incomplete cache entry is rebuilt"
    separate.separate(fx["loop"], tmp_path / "cache", "mdx-2", fake, force=True)
    assert fake.calls == 5


def test_separation_errors_are_clear(tmp_path: Path, fx: dict[str, Path]) -> None:
    with pytest.raises(StemError, match="unknown stem model 'best'"):
        separate.separate(fx["loop"], tmp_path / "cache", "best")
    with pytest.raises(StemError, match="did not return these stems: drums, bass, other"):
        separate.separate(fx["loop"], tmp_path / "cache", "demucs-4", FakeSeparator({"vocals": 1}))
    silent = fixtures.write_wav(tmp_path / "silent.wav", np.zeros((44100, 2)), 44100)
    with pytest.raises(StemError, match="is silent"):
        separate.separate(silent, tmp_path / "cache", "mdx-2", FakeSeparator({}))
    assert not list((tmp_path / "cache").glob("*/stems.json"))


def test_rebalance_applies_gains_exactly(logic_set: StemSet, paths: Paths) -> None:
    gains = {"lead-vox": 3.0, "drums": -2.0}
    result = rebalancing.rebalance(logic_set, gains, paths)
    expected = separate.mixdown(logic_set, gains)
    written, info = audio_io.load(result.path)
    assert info.subtype == "PCM_24" and result.trim_db == 0.0
    assert np.abs(written - expected).max() < 1e-6
    assert result.path.name.startswith("rebalanced-")
    assert rebalancing.rebalance(logic_set, gains, paths).path == result.path
    assert rebalancing.rebalance(logic_set, {"lead-vox": 2.0}, paths).path != result.path
    note = json.loads(result.path.with_suffix(".json").read_text())
    assert note["gains_db"] == gains and note["stem_set"] == logic_set.name


def test_rebalance_keeps_headroom_without_clipping(logic_set: StemSet, paths: Paths) -> None:
    gains = dict.fromkeys(logic_set.stems, 6.0)
    result = rebalancing.rebalance(logic_set, gains, paths)
    written, _ = audio_io.load(result.path)
    assert result.trim_db == pytest.approx(-1.0, abs=0.05), "a -6 dBFS mix raised 6 dB, kept at -1"
    assert 20 * np.log10(np.abs(written).max()) == pytest.approx(-1.0, abs=0.01)
    assert result.peak_dbfs == pytest.approx(-1.0, abs=0.01)


def test_rebalance_refuses_big_moves_and_unknown_stems(logic_set: StemSet, paths: Paths) -> None:
    with pytest.raises(StemError, match="limited to ±6 dB"):
        rebalancing.rebalance(logic_set, {"drums": -9.0}, paths)
    with pytest.raises(StemError, match="no such stems: guitar"):
        rebalancing.rebalance(logic_set, {"guitar": 1.0}, paths)


def test_rebalance_can_run_a_chain_on_one_stem(logic_set: StemSet, paths: Paths) -> None:
    plain = audio_io.load(rebalancing.rebalance(logic_set, {}, paths).path)[0]
    result = rebalancing.rebalance(logic_set, {}, paths, chains={"808-bass": "warm"})
    shaped = audio_io.load(result.path)[0]
    assert np.abs(shaped - plain).max() > 1e-3 and result.chains == {"808-bass": "warm"}


def test_import_then_rebalance_and_master_as_one_track(
    paths: Paths, settings: Settings, reference: Reference, song: tuple[Path, Path]
) -> None:
    project, stem_set = workflow.import_logic_stems(paths, song[0], "Night Drive", "hip-hop")
    assert project.track.slug == "night-drive" and project.input_path.name.endswith(
        "(stem sum).wav"
    )
    assert stem_set.dir.parent == project.dir / "stems"
    assert workflow.find_set(paths, "night-drive") == stem_set

    diagnosis = workflow.diagnose(stem_set, project.input_path, settings)
    assert diagnosis.reference is None and diagnosis.flags == [] and diagnosis.warning is None
    assert json.loads((stem_set.dir / "diagnosis.json").read_text())["stem_set"] == stem_set.name

    options = MasterOptions.from_settings(
        settings, "hip-hop/default", title="Night Drive", genre="hip-hop"
    )
    rebalanced, mastered = workflow.rebalance_and_master(
        paths, settings, stem_set, {"lead-vox": 2.0}, options
    )
    assert mastered.manifest.track.slug == "night-drive", "the new run stays with the track"
    assert mastered.reports["master"].loudness.lufs_i == pytest.approx(-14.0, abs=0.2)
    assert mastered.manifest.input_file == f"input/{rebalanced.path.name}"
    assert workflow.sets_for_run(mastered) == [stem_set]


def test_import_with_the_bounce_uses_it_as_the_mix(paths: Paths, song: tuple[Path, Path]) -> None:
    project, _ = workflow.import_logic_stems(paths, song[0], "Night Drive", mix=song[1])
    assert project.input_path.name == "Night Drive.wav"
    assert project.input_sha256 == audio_io.file_sha256(song[1])


def test_diagnosis_compares_with_the_references_separated_stems(
    paths: Paths, settings: Settings, reference: Reference, song: tuple[Path, Path]
) -> None:
    project, stem_set = workflow.import_logic_stems(paths, song[0], "Night Drive", "hip-hop")
    fake = FakeSeparator({"vocals": 0.9, "drums": 0.2, "bass": 0.2, "other": 0.1})
    diagnosis = workflow.diagnose(stem_set, project.input_path, settings, reference, runner=fake)
    assert fake.calls == 1 and diagnosis.reference is not None
    assert diagnosis.reference.separated and diagnosis.reference.source == "demucs-4-fast"
    assert (reference.path.parent / ".stems").is_dir(), "reference stems are cached beside it"
    assert "vocal_to_instrumental_low" in {flag.code for flag in diagnosis.flags}
    workflow.diagnose(stem_set, project.input_path, settings, reference, runner=fake)
    assert fake.calls == 1


def test_find_set_prefers_real_stems_and_explains_misses(
    paths: Paths, song: tuple[Path, Path], fx: dict[str, Path]
) -> None:
    with pytest.raises(StemError, match="has no stems yet"):
        workflow.find_set(paths, "night-drive")
    project, real = workflow.import_logic_stems(paths, song[0], "Night Drive", mix=song[1])
    separated = workflow.separate_mix(
        project, "mdx-2", FakeSeparator({"vocals": 0.5, "instrumental": 0.5})
    )
    assert workflow.find_set(paths, "night-drive") == real
    assert workflow.find_set(paths, "night-drive", separated.name) == separated
    with pytest.raises(StemError, match="no stem set 'nope'"):
        workflow.find_set(paths, "night-drive", "nope")


def test_vocal_stem_is_checked_for_sibilance_and_uneven_phrases(
    paths: Paths, settings: Settings, tmp_path: Path
) -> None:
    stems, _ = fixtures.song_stems()
    hissy, _ = fixtures.sibilant_vocal()
    frames = len(stems["vocals"])
    take = np.tile(hissy, frames // len(hissy) + 1)[:frames]
    level = float(np.sqrt(np.mean(stems["vocals"] ** 2)) / np.sqrt(np.mean(take**2)))
    vocal = np.repeat((level * take)[:, None], 2, axis=1)
    vocal[9 * fixtures.SR : 11 * fixtures.SR] *= 10 ** (-9 / 20)
    for role, audio in {**stems, "vocals": vocal}.items():
        fixtures.write_wav(tmp_path / "stems" / FILE_NAMES[role], audio)
    project, stem_set = workflow.import_logic_stems(paths, tmp_path / "stems", "Hissy")
    flags = workflow.diagnose(stem_set, project.input_path, settings).flags
    codes = [f.code for f in flags]
    assert "sibilance" in codes and codes.count("sibilance_at") >= 3
    quiet = next(f for f in flags if f.code == "vocal_phrase_quiet")
    assert quiet.start_s == pytest.approx(9.0, abs=0.6)
