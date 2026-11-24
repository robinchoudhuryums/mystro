from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from masterbench import audio_io, fixtures, index
from masterbench.analysis import loudness, spectrum
from masterbench.analysis import report as analysis
from masterbench.config import BANDS, Paths, Settings
from masterbench.master import chain, curve_match, pipeline
from masterbench.master.chain import ChainPreset, Module
from masterbench.master.pipeline import MasterOptions
from masterbench.references import library, target_curve
from masterbench.references.target_curve import CurveError

SR = fixtures.SR
TILTS = [(3.0, -2.0, -1.0), (2.0, -3.0, -2.0), (4.0, -1.0, -1.5)]


def tilted_noise(seed: int, low: float, high: float, mid: float) -> np.ndarray:
    """Pink noise with a different broad EQ, standing in for one reference of a genre."""
    modules = (
        Module("low_shelf", {"freq_hz": 120.0, "gain_db": low}),
        Module("high_shelf", {"freq_hz": 6000.0, "gain_db": high}),
        Module("peak", {"freq_hz": 1000.0, "gain_db": mid, "q": 0.7}),
    )
    return chain.run(ChainPreset("tilt", "", modules), fixtures.pink_noise(20.0, seed=seed), SR)


def bands_of(audio: np.ndarray) -> dict[str, float]:
    spectra = spectrum.cross_spectra(audio, SR)
    return spectrum.band_levels(spectra, loudness.integrated(audio, SR))


@pytest.fixture
def house(paths: Paths, settings: Settings, tmp_path: Path) -> target_curve.TargetCurve:
    """A 'house' library of three tilted noises, and its curve."""
    for i, tilt in enumerate(TILTS):
        file = fixtures.write_wav(tmp_path / f"ref{i}.wav", tilted_noise(10 + i, *tilt))
        library.add(paths, file, "house", settings)
    return target_curve.build(paths, "house")


def test_curve_is_the_median_and_middle_half_of_its_references(
    house: target_curve.TargetCurve, paths: Paths
) -> None:
    refs = library.list_references(paths, "house")
    assert house.references == ["ref0", "ref1", "ref2"] and house.genre == "house"
    for band in BANDS:
        levels = sorted(r.report.bands[band] for r in refs if r.report.bands)
        assert house.bands[band].median == pytest.approx(levels[1], abs=0.01)
        assert house.bands[band].p25 <= house.bands[band].median <= house.bands[band].p75
        assert levels[0] <= house.bands[band].p25 and house.bands[band].p75 <= levels[2]
    assert len(house.freqs_hz) == len(house.median_db) == len(house.p25_db) == 120
    assert np.all(np.array(house.p25_db) <= np.array(house.p75_db))
    assert set(house.regions) == {"mud", "harsh"} and house.hook_bands is None

    saved = json.loads((paths.target_curves / "house.json").read_text())
    assert target_curve.TargetCurve.from_dict(saved) == house
    assert target_curve.load(paths, "House") == house and target_curve.load(paths, None) is None


def test_curve_match_lands_on_the_target_in_every_band(house: target_curve.TargetCurve) -> None:
    """Phase 6 criterion: within 1 dB of the target median in every band, on pink noise."""
    pink = fixtures.pink_noise(20.0, seed=3)
    median = house.band("median")
    before = bands_of(pink)
    assert max(abs(before[b] - median[b]) for b in BANDS) > 2.5, "it starts well off the curve"

    matched, eq = curve_match.match(pink, SR, house.freqs_hz, house.median_db)
    after = bands_of(matched)
    assert {b: round(after[b] - median[b], 2) for b in BANDS} == pytest.approx(
        dict.fromkeys(BANDS, 0.0), abs=1.0
    )
    assert max(abs(after[b] - median[b]) for b in BANDS) < 0.25, "in fact within a quarter dB"
    assert eq.max_abs_db <= curve_match.MAX_DB and matched.shape == pink.shape


def test_curve_match_is_static_linear_phase_and_repeatable(house: target_curve.TargetCurve) -> None:
    pink = fixtures.pink_noise(20.0, seed=3)
    first, eq = curve_match.match(pink, SR, house.freqs_hz, house.median_db)
    again, eq_again = curve_match.match(pink, SR, house.freqs_hz, house.median_db)
    assert np.array_equal(first, again) and eq == eq_again
    kernel = curve_match.fir(eq, SR)
    assert np.allclose(kernel, kernel[::-1]), "a symmetric kernel is linear phase"

    click = np.zeros((SR, 2))
    click[SR // 2] = 1.0
    out = curve_match.apply(click, SR, eq)
    assert int(np.argmax(np.abs(out[:, 0]))) == SR // 2, "timing is unchanged"
    halves = curve_match.apply(pink[:, :1], SR, eq) + curve_match.apply(pink[:, 1:], SR, eq)
    whole = curve_match.apply(pink[:, :1] + pink[:, 1:], SR, eq)
    assert np.allclose(halves, whole, atol=1e-9), "parts processed apart still sum to the whole"


def test_the_eq_is_limited(house: target_curve.TargetCurve) -> None:
    dark = chain.run(
        ChainPreset("dark", "", (Module("high_shelf", {"freq_hz": 2000.0, "gain_db": -12.0}),)),
        fixtures.pink_noise(20.0, seed=3),
        SR,
    )
    matched, eq = curve_match.match(dark, SR, house.freqs_hz, house.median_db)
    assert eq.max_abs_db == pytest.approx(curve_match.MAX_DB, abs=0.01)
    before, after = bands_of(dark)["air"], bands_of(matched)["air"]
    assert before < after < house.bands["air"].p25, "closer, but the rest is the mix's job"


def test_only_bands_outside_the_references_range_are_flagged(
    house: target_curve.TargetCurve, settings: Settings, tmp_path: Path
) -> None:
    inside = fixtures.write_wav(tmp_path / "inside.wav", tilted_noise(4, 3.0, -2.0, -1.5))
    report = analysis.against_curve(analysis.analyze_file(inside, "mix", settings), house, settings)
    assert report.curve_deltas is not None
    assert max(abs(v) for v in report.curve_deltas.values()) < 1.0
    assert not [f for f in report.flags if f.code.startswith("band_")]

    flat = fixtures.write_wav(tmp_path / "flat.wav", fixtures.pink_noise(20.0, seed=3))
    report = analysis.against_curve(analysis.analyze_file(flat, "mix", settings), house, settings)
    codes = {f.code: f for f in report.flags}
    assert {"band_sub_under", "band_bass_under", "band_air_over"} <= set(codes)
    assert "under the range of your 3 house references" in codes["band_sub_under"].message
    assert report.bands is not None
    assert codes["band_sub_under"].value == pytest.approx(
        house.bands["sub"].p25 - report.bands["sub"], abs=0.01
    )


def test_curve_needs_usable_references(paths: Paths, settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(CurveError, match="no references in genre 'house'"):
        target_curve.build(paths, "house")
    with pytest.raises(CurveError, match="no target curve for 'house' yet"):
        target_curve.get(paths, "house")
    song, _ = fixtures.arranged_song()
    library.add(paths, fixtures.write_wav(tmp_path / "good.wav", song), "house", settings)
    library.add(
        paths,
        fixtures.write_wav(tmp_path / "clipped.wav", np.clip(song * 6, -1, 1)),
        "house",
        settings,
    )
    with pytest.raises(CurveError, match="needs at least 2 usable references in 'house', found 1"):
        target_curve.build(paths, "house")
    assert target_curve.build(paths, "house", include_low_quality=True).references == [
        "clipped",
        "good",
    ]
    assert target_curve.build(paths, "house", names=["good", "clipped"]).references == [
        "clipped",
        "good",
    ]
    with pytest.raises(CurveError, match="no such references in 'house': nope"):
        target_curve.build(paths, "house", names=["nope"])


def test_mastering_to_the_curve(
    house: target_curve.TargetCurve, paths: Paths, settings: Settings, tmp_path: Path
) -> None:
    mix = fixtures.write_wav(tmp_path / "Flat Track.wav", fixtures.pink_noise(20.0, seed=3))
    options = MasterOptions.from_settings(settings, "", mode="curve", genre="house")
    stages: list[str] = []
    result = pipeline.master(mix, options, paths, settings, lambda s, f: stages.append(s))
    assert "Matching to the house curve" in stages

    used = result.manifest.settings
    assert used["mode"] == "curve" and used["curve"]["references"] == ["ref0", "ref1", "ref2"]
    assert len(used["match_eq"]["gain_db"]) == len(used["match_eq"]["freqs_hz"]) == 120
    master = result.reports["master"]
    assert (
        master.loudness.lufs_i == pytest.approx(-14.0, abs=0.2)
        and master.loudness.true_peak_dbtp <= -1.0
    )
    assert (
        master.curve_deltas is not None and max(abs(v) for v in master.curve_deltas.values()) < 0.5
    )
    assert not [f for f in master.flags if f.code.startswith("band_")]
    assert {"band_sub_under", "band_air_over"} <= {f.code for f in result.reports["mix"].flags}

    again = pipeline.master(mix, options, paths, settings)
    assert again.output("master_24").read_bytes() == result.output("master_24").read_bytes()

    # The stored EQ reproduces the processing: the basis for locked alternate versions.
    audio, info = audio_io.load(result.mix_path)
    eq = curve_match.MatchEQ(**used["match_eq"])
    matched, _ = curve_match.match(audio, info.sample_rate, house.freqs_hz, house.median_db)
    assert np.allclose(curve_match.apply(audio, info.sample_rate, eq), matched, atol=1e-4)

    with index.open_index(paths) as conn:
        rows = conn.execute(
            "SELECT count(*) FROM band_deltas WHERE vs = 'target_curve' AND section = 'full'"
        ).fetchone()
    assert rows[0] == 4 * 7, "mix and master of both runs"


def test_curve_mode_errors_and_the_curve_judging_other_runs(
    house: target_curve.TargetCurve, paths: Paths, settings: Settings, tmp_path: Path, fx
) -> None:
    with pytest.raises(CurveError, match="no target curve for 'hip-hop' yet"):
        pipeline.master(
            fx["loop"], MasterOptions.from_settings(settings, "", mode="curve"), paths, settings
        )
    with pytest.raises(pipeline.MasteringError, match="unknown mode 'magic'"):
        pipeline.master(
            fx["loop"], MasterOptions.from_settings(settings, "", mode="magic"), paths, settings
        )

    # Once a genre has a curve, it does the judging in analyses and reference masters too.
    analyzed = pipeline.analyze(fx["pink_noise"], paths, settings, genre="house")
    assert analyzed.manifest.settings["curve"]["genre"] == "house"
    mix = analyzed.reports["mix"]
    assert mix.curve_deltas is not None and mix.band_deltas is not None
    messages = [f.message for f in mix.flags if f.code.startswith("band_")]
    assert messages and all("house references" in m for m in messages)
