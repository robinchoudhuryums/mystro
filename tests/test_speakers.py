from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from masterbench import audio_io, fixtures
from masterbench.analysis import spectrum
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.chain import ChainError
from masterbench.master.pipeline import MasterOptions, RunResult
from masterbench.preview import speakers
from masterbench.references.library import Reference

SR = fixtures.SR


def absolute_bands(audio: np.ndarray) -> dict[str, float]:
    return spectrum.band_levels(spectrum.cross_spectra(audio, SR), 0.0)


@pytest.fixture
def mastered(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> RunResult:
    return pipeline.master(
        fx["loop"], MasterOptions.from_settings(settings, "hip-hop/default"), paths, settings
    )


def test_packaged_profiles(paths: Paths) -> None:
    found = speakers.profiles(paths)
    assert list(found) == ["car", "club", "earbuds", "laptop", "mono", "phone"]
    assert found["phone"].mono and found["mono"].mono and not found["car"].mono
    assert found["laptop"].width == 0.5 and found["mono"].modules == ()
    assert all(profile.label and profile.description for profile in found.values())


def test_each_speaker_does_what_it_says(paths: Paths) -> None:
    noise = fixtures.pink_noise(8.0)
    before = absolute_bands(noise)
    found = speakers.profiles(paths)

    def change(name: str) -> dict[str, float]:
        out = speakers.simulate(noise, SR, found[name])
        assert out.shape == noise.shape
        after = absolute_bands(out)
        return {band: after[band] - before[band] for band in before}

    phone, laptop, car, club = change("phone"), change("laptop"), change("car"), change("club")
    assert phone["sub"] < -40 and phone["bass"] < -18 and phone["upper_mids"] > 1
    assert laptop["sub"] < -25 and abs(laptop["presence"]) < 1.5
    assert car["sub"] > 3 and car["air"] < -2
    assert club["sub"] > 5 and abs(club["mids"]) < 1

    side = lambda x: float(np.mean((x[:, 0] - x[:, 1]) ** 2))  # noqa: E731
    assert side(speakers.simulate(noise, SR, found["mono"])) == pytest.approx(0.0, abs=1e-12)
    assert side(speakers.simulate(noise, SR, found["phone"])) == pytest.approx(0.0, abs=1e-12)
    narrowed = side(speakers.simulate(noise, SR, found["laptop"]))
    assert 0 < narrowed < 0.5 * side(noise)
    mono_in = speakers.simulate(noise[:, :1], SR, found["laptop"])
    assert mono_in.shape == (len(noise), 1)


def test_sims_are_rendered_level_measured_and_cached(mastered: RunResult, paths: Paths) -> None:
    sims = {sim.name: sim for sim in speakers.render(mastered, paths)}
    assert set(sims) == set(speakers.profiles(paths))
    master, info = audio_io.load(mastered.output("master_24"))
    for sim in sims.values():
        audio, sim_info = audio_io.load(mastered.run_dir / sim.file)
        assert audio.shape == master.shape and sim_info.sample_rate == info.sample_rate
        assert sim.lufs_i < 0 and set(sim.band_change_db) == set(absolute_bands(master))
    assert sims["phone"].lufs_i < sims["mono"].lufs_i - 5, "losing the 808 costs most of the level"
    assert (
        "sub and bass are gone" in sims["phone"].message and "saturation" in sims["phone"].message
    )
    assert "sub is gone" in sims["laptop"].message
    assert sims["car"].message.startswith("Lifts the sub")
    assert sims["mono"].message.startswith("Mono sum")

    stamp = (mastered.run_dir / sims["phone"].file).stat().st_mtime_ns
    assert speakers.render(mastered, paths) == list(sims.values())
    assert (mastered.run_dir / sims["phone"].file).stat().st_mtime_ns == stamp
    assert speakers.load(mastered) == list(sims.values())
    (mastered.run_dir / sims["car"].file).unlink()
    assert speakers.load(mastered) is None


def test_your_own_profiles_override_and_extend(mastered: RunResult, paths: Paths) -> None:
    folder = speakers.user_dir(paths)
    folder.mkdir(parents=True)
    (folder / "phone.toml").write_text(
        'label = "My phone"\n[[module]]\ntype = "highpass"\ncutoff_hz = 250.0\n'
    )
    (folder / "kitchen.toml").write_text('label = "Kitchen radio"\nmono = true\n')
    found = speakers.profiles(paths)
    assert found["phone"].label == "My phone" and not found["phone"].mono
    assert "kitchen" in found
    assert {sim.name for sim in speakers.render(mastered, paths)} == set(found)

    (folder / "broken.toml").write_text('[[module]]\ntype = "reverb"\n')
    with pytest.raises(ChainError, match="unknown type"):
        speakers.profiles(paths)
    (folder / "broken.toml").write_text("width = 9\n")
    with pytest.raises(ChainError, match="width must be 0 to 2"):
        speakers.profiles(paths)


def test_an_analysis_is_simulated_from_its_mix(paths: Paths, settings: Settings, fx) -> None:
    analyzed = pipeline.analyze(fx["loop"], paths, settings)
    sims = speakers.render(analyzed, paths)
    assert len(sims) == 6 and (analyzed.run_dir / sims[0].file).is_file()
