from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from masterbench import audio_io, fixtures
from masterbench.analysis import compare, loudness, spectrum, stereo
from masterbench.config import BANDS, Settings, Thresholds

SR = fixtures.SR


def bands_of(audio: np.ndarray, sr: int = SR) -> dict[str, float]:
    return spectrum.band_levels(spectrum.cross_spectra(audio, sr), loudness.integrated(audio, sr))


def image_of(audio: np.ndarray) -> stereo.Stereo:
    return stereo.analyze(spectrum.cross_spectra(audio, SR))


def test_pink_noise_has_equal_energy_per_octave() -> None:
    bands = bands_of(fixtures.pink_noise(20.0))
    per_octave = [bands[name] - 10 * np.log10(np.log2(hi / lo)) for name, (lo, hi) in BANDS.items()]
    assert max(per_octave) - min(per_octave) < 0.6


def test_a_tone_lands_in_its_band() -> None:
    t = np.arange(SR * 5) / SR
    for freq, expected in ((40.0, "sub"), (1000.0, "mids"), (8000.0, "presence")):
        tone = np.column_stack([np.sin(2 * np.pi * freq * t)] * 2) * 0.25
        bands = bands_of(tone)
        assert max(bands, key=lambda name: bands[name]) == expected
        others = [level for name, level in bands.items() if name != expected]
        assert bands[expected] - max(others) > 40


def test_band_levels_ignore_overall_gain() -> None:
    noise = fixtures.pink_noise(8.0)
    quiet = bands_of(noise * 10 ** (-9 / 20))
    assert quiet == pytest.approx(bands_of(noise), abs=0.02)
    assert spectrum.band_deltas(quiet, bands_of(noise)) == pytest.approx(
        dict.fromkeys(BANDS, 0), abs=0.03
    )


def test_band_deltas_show_an_eq_move() -> None:
    noise = fixtures.pink_noise(10.0)
    sos = signal.butter(2, 6000, "highpass", fs=SR, output="sos")
    bright = noise + 1.0 * np.asarray(signal.sosfilt(sos, noise, axis=0))
    deltas = spectrum.band_deltas(bands_of(bright), bands_of(noise))
    assert deltas["air"] > 3.0 and deltas["presence"] > 1.0
    assert deltas["sub"] < 0 and deltas["bass"] < 0, "at equal loudness the lows sit lower"
    flags = compare.band_flags(deltas, Thresholds(), "the reference")
    assert "band_air_over" in {f.code for f in flags}
    air = next(f for f in flags if f.code == "band_air_over")
    assert "Air (10 kHz to 20 kHz)" in air.message and "over the reference" in air.message


def test_smoothed_spectrum_shape(fx: dict[str, Path]) -> None:
    audio, info = audio_io.load(fx["loop"])
    spectra = spectrum.cross_spectra(audio, info.sample_rate)
    curve = spectrum.smoothed(spectra, loudness.integrated(audio, info.sample_rate))
    assert len(curve.freqs_hz) == len(curve.level_db) == 120
    assert curve.freqs_hz[0] == 20.0 and 19000 < curve.freqs_hz[-1] <= 20000
    assert curve.freqs_hz[12] == pytest.approx(40.0)
    peak = curve.freqs_hz[int(np.argmax(curve.level_db))]
    assert 35 < peak < 60, "the F1 808 dominates the loop"

    short = spectrum.smoothed(spectrum.cross_spectra(audio[:4000], 44100), -20.0)
    assert len(short.level_db) == 120 and all(np.isfinite(short.level_db))


def test_mono_and_centred_material_is_fully_correlated(fx: dict[str, Path]) -> None:
    vocal, _ = audio_io.load(fx["vocal"])
    mono = image_of(vocal)
    assert mono.correlation == 1.0 and mono.side_mid_db == loudness.FLOOR_DB
    loop, _ = audio_io.load(fx["loop"])
    image = image_of(loop)
    assert image.low_mono_corr > 0.99 and image.correlation > 0.9
    assert compare.stereo_flags(image, Thresholds()) == []


def test_inverted_channel_is_a_phase_problem() -> None:
    noise = fixtures.pink_noise(6.0)[:, 0]
    image = image_of(np.column_stack([noise, -noise]))
    assert image.correlation == -1.0 and image.side_mid_db == -loudness.FLOOR_DB
    flags = {f.code: f for f in compare.stereo_flags(image, Thresholds())}
    assert flags["phase_negative"].severity == "problem"
    assert "low_end_not_mono" in flags


def test_independent_channels_are_wide_not_broken() -> None:
    rng = np.random.default_rng(3)
    image = image_of(rng.standard_normal((SR * 6, 2)) * 0.1)
    assert abs(image.correlation) < 0.05
    assert image.side_mid_db == pytest.approx(0.0, abs=0.5)
    codes = [f.code for f in compare.stereo_flags(image, Thresholds())]
    assert codes == ["wide_stereo", "low_end_not_mono"]


def test_stereo_sub_is_caught_even_when_the_rest_is_fine() -> None:
    t = np.arange(SR * 6) / SR
    centre = fixtures.pink_noise(6.0, seed=5)[:, 0]
    sub = 0.3 * np.sin(2 * np.pi * 45 * t)
    audio = np.column_stack([centre + sub, centre - sub])
    image = image_of(audio)
    assert image.low_mono_corr < 0 < image.band_correlation["mids"]
    codes = {f.code for f in compare.stereo_flags(image, Thresholds())}
    assert "low_end_not_mono" in codes and "phase_mids" not in codes


def test_reports_round_trip_with_the_new_fields(fx: dict[str, Path], settings: Settings) -> None:
    from masterbench.analysis.report import AnalysisReport, analyze_file

    report = analyze_file(fx["loop"], "mix", settings)
    assert AnalysisReport.from_dict(report.to_dict()) == report
    old = {k: v for k, v in report.to_dict().items() if k in AnalysisReport.__annotations__}
    for added in ("spectrum", "bands", "stereo", "band_deltas", "short_term_lufs"):
        old.pop(added)
    legacy = AnalysisReport.from_dict(old)
    assert legacy.spectrum is None and legacy.loudness == report.loudness
