from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from masterbench import fixtures
from masterbench.analysis import loudness, spectrum
from masterbench.config import Paths, Settings
from masterbench.master import chain, pipeline, presets
from masterbench.master.chain import ChainError, ChainPreset, Module
from masterbench.master.pipeline import MasterOptions
from masterbench.references.library import Reference

SR = fixtures.SR
# Free plugins used when installed (Debian/Ubuntu: zam-plugins). Stereo, deterministic.
TEST_PLUGIN = Path("/usr/lib/vst3/ZamCompX2.vst3")
needs_plugin = pytest.mark.skipif(not TEST_PLUGIN.exists(), reason="no test VST3 installed")


def one(module_type: str, **params: Any) -> ChainPreset:
    return ChainPreset("test", "", (Module(module_type, params),))


def bands(audio: np.ndarray) -> dict[str, float]:
    return spectrum.band_levels(spectrum.cross_spectra(audio, SR), 0.0)


@pytest.fixture(scope="module")
def noise() -> np.ndarray:
    return fixtures.pink_noise(6.0)


def test_empty_chain_is_transparent(noise: np.ndarray) -> None:
    out = chain.run(ChainPreset("empty"), noise, SR)
    assert out.shape == noise.shape and out.dtype == np.float64
    assert np.abs(out - noise).max() < 1e-6


def test_highpass_removes_sub_only(noise: np.ndarray) -> None:
    before, after = bands(noise), bands(chain.run(one("highpass", cutoff_hz=80.0), noise, SR))
    assert after["sub"] < before["sub"] - 6
    assert after["mids"] == pytest.approx(before["mids"], abs=0.1)


@pytest.mark.parametrize(
    ("module_type", "freq", "band", "untouched"),
    [("low_shelf", 100.0, "sub", "presence"), ("high_shelf", 8000.0, "air", "sub")],
)
def test_shelves_move_their_end_of_the_spectrum(noise, module_type, freq, band, untouched) -> None:
    before = bands(noise)
    after = bands(chain.run(one(module_type, freq_hz=freq, gain_db=6.0), noise, SR))
    assert after[band] - before[band] == pytest.approx(6.0, abs=1.5)
    assert after[untouched] == pytest.approx(before[untouched], abs=0.3)


def test_peak_filter_cuts_around_its_frequency(noise: np.ndarray) -> None:
    before = bands(noise)
    after = bands(chain.run(one("peak", freq_hz=1000.0, gain_db=-9.0, q=0.7), noise, SR))
    assert after["mids"] < before["mids"] - 3
    assert after["sub"] == pytest.approx(before["sub"], abs=0.2)


def test_gain_is_exact(noise: np.ndarray) -> None:
    out = chain.run(one("gain", gain_db=-6.0), noise, SR)
    assert np.allclose(out, noise * 10 ** (-6 / 20), atol=1e-6)


def test_compressor_turns_down_what_is_over_the_threshold() -> None:
    t = np.arange(SR * 2) / SR
    tone = np.column_stack([np.sin(2 * np.pi * 200 * t)] * 2)
    settings = {"threshold_db": -30.0, "ratio": 8.0, "attack_ms": 1.0}
    rms_db = lambda x: 10 * np.log10(np.mean(x[SR:] ** 2))  # noqa: E731

    loud = 0.5 * tone
    squashed = chain.run(one("compressor", **settings), loud, SR)
    reduction = rms_db(loud) - rms_db(squashed)
    assert 15 < reduction < 24, "24 dB over an 8:1 threshold should lose about 21 dB"

    quiet = 0.01 * tone
    assert np.allclose(chain.run(one("compressor", **settings), quiet, SR), quiet, atol=1e-5)

    with_makeup = chain.run(one("compressor", **settings, makeup_db=6.0), loud, SR)
    assert np.allclose(with_makeup, squashed * 10 ** (6 / 20), atol=1e-5)


def test_saturation_adds_harmonics_and_keeps_quiet_signals() -> None:
    t = np.arange(SR * 2) / SR
    tone = np.column_stack([np.sin(2 * np.pi * 100 * t)] * 2)
    spectrum_of = lambda x: np.abs(np.fft.rfft(x[:, 0] * np.hanning(len(x))))  # noqa: E731
    loud = spectrum_of(chain.run(one("saturation", drive_db=12.0), 0.8 * tone, SR))
    assert loud[600] > 0.02 * loud[200], "third harmonic of 100 Hz appears at 300 Hz"
    quiet = chain.run(one("saturation", drive_db=12.0), 0.001 * tone, SR)
    assert np.allclose(quiet, 0.001 * tone, atol=1e-6), "small signals pass at unity gain"
    half = chain.run(one("saturation", drive_db=12.0, mix=0.0), 0.8 * tone, SR)
    assert np.allclose(half, 0.8 * tone, atol=1e-6)


def test_limiter_module_holds_its_ceiling() -> None:
    loop, _ = fixtures.drum_808_loop()
    out = chain.run(one("limiter", ceiling_dbtp=-3.0), loop * 4.0, SR)
    assert loudness.to_db(loudness.true_peak(out)) <= -3.0 + 1e-4


def test_chain_runs_modules_in_order_and_is_repeatable(paths: Paths, noise: np.ndarray) -> None:
    preset = presets.get(paths, "gentle-glue")
    assert [m.type for m in preset.modules] == ["highpass", "compressor", "high_shelf"]
    first, second = chain.run(preset, noise, SR), chain.run(preset, noise, SR)
    assert np.array_equal(first, second)
    mono = chain.run(preset, noise[:, :1], SR)
    assert mono.shape == (len(noise), 1)


@pytest.mark.parametrize(
    ("module", "message"),
    [
        (Module("exciter", {}), "unknown type"),
        (Module("peak", {"frequency": 1000}), "unknown setting 'frequency'"),
        (Module("peak", {"gain_db": 40}), "gain_db must be -12 to 12"),
        (Module("gain", {"gain_db": "loud"}), "gain_db must be a number"),
        (Module("gain", {"gain_db": True}), "gain_db must be a number"),
        (Module("plugin", {}), "needs 'path'"),
        (Module("plugin", {"path": "/x.vst3", "preset": 1}), "unknown keys"),
    ],
)
def test_bad_presets_name_the_module(module: Module, message: str) -> None:
    with pytest.raises(ChainError, match=message) as error:
        chain.validate(ChainPreset("mine", "", (Module("gain", {}), module)))
    assert "mine: module 2" in str(error.value)


def test_missing_plugin_is_reported(noise: np.ndarray) -> None:
    preset = one("plugin", path="/nowhere/Missing.vst3")
    with pytest.raises(ChainError, match="plugin not found: /nowhere/Missing.vst3"):
        chain.run(preset, noise, SR)


def test_presets_round_trip_through_toml(paths: Paths) -> None:
    preset = ChainPreset(
        "mine",
        "Two moves",
        (
            Module("highpass", {"cutoff_hz": 30.0}),
            Module("peak", {"freq_hz": 3000.0, "gain_db": -2.0}),
        ),
    )
    saved = presets.save(paths, preset)
    assert saved == paths.chain_presets / "mine.toml"
    assert presets.get(paths, "mine") == preset
    assert presets.from_toml(presets.to_toml(preset), "mine") == preset
    assert "mine" in presets.names(paths) and "streaming-safe" in presets.names(paths)


def test_user_presets_shadow_packaged_ones_and_keep_comments(paths: Paths) -> None:
    text = '# my take\n[[module]]\ntype = "highpass"\ncutoff_hz = 40.0\n'
    presets.save_text(paths, "streaming-safe", text)
    assert presets.text(paths, "streaming-safe") == text
    assert presets.get(paths, "streaming-safe").modules[0].params == {"cutoff_hz": 40.0}


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("Bad Name", "", "preset names use"),
        ("../escape", "", "preset names use"),
        ("ok", "[[module]\n", "not valid TOML"),
        ("ok", "[[module]]\ncutoff_hz = 30\n", "needs a 'type'"),
        ("ok", "colour = 'red'\n", "unknown keys"),
    ],
)
def test_invalid_preset_text_is_rejected_before_saving(paths: Paths, name, text, message) -> None:
    with pytest.raises(ChainError, match=message):
        presets.save_text(paths, name, text)
    assert not list(paths.chain_presets.glob("*.toml"))
    with pytest.raises(ChainError, match="no chain preset named"):
        presets.get(paths, "nope")


def test_pipeline_runs_chains_and_still_hits_the_targets(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    options = MasterOptions.from_settings(
        settings, "hip-hop/default", pre_chain="streaming-safe", post_chain="warm"
    )
    stages: list[str] = []
    result = pipeline.master(fx["loop"], options, paths, settings, lambda s, f: stages.append(s))
    assert (
        stages.index("Running the pre-chain (streaming-safe)")
        < stages.index("Matching to the reference")
        < stages.index("Running the post-chain (warm)")
        < stages.index("Setting loudness and limiting")
    )
    master = result.reports["master"].loudness
    assert master.lufs_i == pytest.approx(-14.0, abs=0.2) and master.true_peak_dbtp <= -1.0
    assert sf.info(result.output("master_24")).frames == sf.info(fx["loop"]).frames

    stored = result.manifest.settings
    assert stored["pre_chain"] == presets.get(paths, "streaming-safe").to_dict()
    assert [m["type"] for m in stored["post_chain"]["module"]] == ["low_shelf", "saturation"]

    plain = pipeline.master(
        fx["loop"], MasterOptions.from_settings(settings, "hip-hop/default"), paths, settings
    )
    assert plain.manifest.settings["pre_chain"] is None
    assert plain.output("master_24").read_bytes() != result.output("master_24").read_bytes()


def test_a_broken_chain_stops_the_master_before_any_run_is_written(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> None:
    options = MasterOptions.from_settings(settings, "hip-hop/default", post_chain="nope")
    with pytest.raises(ChainError, match="no chain preset named 'nope'"):
        pipeline.master(fx["loop"], options, paths, settings)
    assert not list(paths.projects.glob("*/runs/*"))


def test_scan_lists_plugin_bundles(tmp_path: Path) -> None:
    (tmp_path / "Fake.vst3").mkdir()
    (tmp_path / "Other.component").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    found = chain.scan_plugins([tmp_path, tmp_path / "missing"])
    assert [(p.path.name, p.format) for p in found] == [
        ("Fake.vst3", "VST3"),
        ("Other.component", "AU"),
    ]
    assert chain.scan_plugins([tmp_path / "missing"]) == []


@needs_plugin
def test_plugin_preset_round_trips_with_identical_output(paths: Paths) -> None:
    loop, _ = fixtures.drum_808_loop()
    captured = chain.capture_plugin(TEST_PLUGIN)
    tweaked = {**captured.params["parameters"], "threshold_db": -30.0, "ratio": 6.0}
    module = Module("plugin", {**captured.params, "parameters": tweaked})
    preset = ChainPreset("comp", "hosted compressor", (Module("gain", {"gain_db": 3.0}), module))

    presets.save(paths, preset)
    reloaded = presets.get(paths, "comp")
    assert reloaded == preset
    original, restored = chain.run(preset, loop, SR), chain.run(reloaded, loop, SR)
    assert np.abs(original - restored).max() < 1e-6, "same preset, same audio"
    assert original.shape == loop.shape
    assert np.abs(original - loop * 10 ** (3 / 20)).max() > 0.01, "the plugin really processed it"

    defaults = ChainPreset("d", "", (Module("gain", {"gain_db": 3.0}), captured))
    assert np.abs(chain.run(defaults, loop, SR) - original).max() > 0.01, "parameters are applied"


@needs_plugin
def test_plugin_parameters_and_bad_names_are_reported() -> None:
    listed = {p["name"]: p for p in chain.plugin_parameters(TEST_PLUGIN)}
    assert listed["threshold_db"]["min"] == -80.0 and listed["threshold_db"]["units"] == "dB"
    bad = Module("plugin", {"path": str(TEST_PLUGIN), "parameters": {"warmth": 1.0}})
    with pytest.raises(ChainError, match="has no parameter 'warmth'"):
        chain.load_plugin(bad)


@pytest.mark.skipif(
    not Path("/usr/lib/vst3/ZamComp.vst3").exists(), reason="no mono test VST3 installed"
)
def test_mono_only_plugins_run_per_channel() -> None:
    loop, _ = fixtures.drum_808_loop()
    module = Module(
        "plugin", {"path": "/usr/lib/vst3/ZamComp.vst3", "parameters": {"threshold_db": -30.0}}
    )
    out = chain.run(ChainPreset("mono", "", (module,)), loop, SR)
    assert out.shape == loop.shape and np.abs(out - loop).max() > 0.01
