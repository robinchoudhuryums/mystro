from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from masterbench import audio_io
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.pipeline import MasterOptions, RunResult
from masterbench.preview import streaming
from masterbench.references.library import Reference


@pytest.fixture
def mastered(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]
) -> RunResult:
    return pipeline.master(
        fx["loop"], MasterOptions.from_settings(settings, "hip-hop/default"), paths, settings
    )


def gains(lufs: float, peak: float) -> dict[str, streaming.PlatformGain]:
    return {g.platform: g for g in streaming.platform_gains(lufs, peak)}


def test_spotify_turns_loud_masters_down() -> None:
    result = gains(-8.0, -1.0)
    assert result["Spotify Normal"].gain_db == -6.0
    assert result["Spotify Normal"].message == "Spotify (Normal) will turn this down 6.0 dB."
    assert result["Spotify Loud"].gain_db == -3.0 and result["Spotify Quiet"].gain_db == -11.0
    assert result["Generic -14"].result_lufs == -14.0


def test_spotify_only_turns_quiet_masters_up_into_peak_headroom() -> None:
    # Spotify's own example: -20 LUFS with a -5 dBFS peak is lifted to -16, not -14.
    result = gains(-20.0, -5.0)
    assert result["Spotify Normal"].gain_db == 4.0 and result["Spotify Normal"].result_lufs == -16.0
    assert "plays at -16.0 LUFS" in result["Spotify Normal"].message
    assert result["Spotify Loud"].gain_db == 9.0, "Loud applies the full gain and limits"
    assert "limiter" in result["Spotify Loud"].message
    assert result["Spotify Quiet"].gain_db == 1.0
    assert result["Generic -14"].gain_db == 0.0


def test_a_master_on_target_is_left_alone() -> None:
    result = gains(-14.0, -1.0)
    assert result["Spotify Normal"].message == "Spotify (Normal) plays this unchanged."
    assert result["Spotify Loud"].gain_db == 3.0 and result["Spotify Quiet"].gain_db == -5.0


def test_codec_round_trips_are_measured_and_aligned(
    mastered: RunResult, settings: Settings
) -> None:
    preview = streaming.render(mastered, settings)
    assert [c.name for c in preview.codecs] == ["vorbis", "aac", "opus"] and preview.skipped == []
    master, info = audio_io.load(mastered.output("master_24"))
    for codec in preview.codecs:
        decoded_path = mastered.run_dir / codec.decoded_file
        assert (mastered.run_dir / codec.encoded_file).stat().st_size > 10_000
        decoded, decoded_info = audio_io.load(decoded_path)
        assert decoded.shape == master.shape and decoded_info.sample_rate == info.sample_rate
        assert sf.info(decoded_path).subtype == "FLOAT", "peaks over full scale must survive"
        assert codec.lufs_i == pytest.approx(-14.0, abs=0.3)
        assert codec.peak_change_db == pytest.approx(
            codec.true_peak_dbtp - preview.source_true_peak_dbtp, abs=0.011
        )
        section = slice(100_000, 200_000)
        lags = signal.correlation_lags(100_000, 100_000)
        lag = lags[np.argmax(signal.correlate(decoded[section, 0], master[section, 0]))]
        assert abs(lag) <= 2, f"{codec.name} preview is {lag} samples out of sync"


def test_previews_are_cached(mastered: RunResult, settings: Settings) -> None:
    first = streaming.render(mastered, settings)
    stamp = (mastered.run_dir / first.codecs[0].encoded_file).stat().st_mtime_ns
    assert streaming.render(mastered, settings) == first
    assert (mastered.run_dir / first.codecs[0].encoded_file).stat().st_mtime_ns == stamp
    assert streaming.load(mastered) == first
    (mastered.run_dir / first.codecs[1].decoded_file).unlink()
    assert streaming.load(mastered) is None, "a deleted preview means render again"
    assert streaming.render(mastered, settings).codecs == first.codecs


def test_peaks_over_the_limit_suggest_a_lower_ceiling() -> None:
    def codec(peak: float) -> streaming.CodecPreview:
        return streaming.CodecPreview("opus", "Opus", "libopus", "a", "b", peak, 0.0, -14.0)

    assert streaming._codec_flags([codec(-1.2)], -1.0, -1.0) == []
    (warn,) = streaming._codec_flags([codec(-0.64), codec(-0.9)], -1.0, -1.0)
    assert warn.severity == "warn" and "-0.64 dBTP" in warn.message and "-1.5 dBTP" in warn.message
    (clip,) = streaming._codec_flags([codec(0.3)], -1.0, -1.0)
    assert clip.severity == "problem" and "-2.5 dBTP" in clip.message and "clips" in clip.message


def test_missing_encoders_are_skipped_not_fatal(
    mastered: RunResult, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(streaming, "available_encoders", lambda: frozenset({"libvorbis", "aac"}))
    preview = streaming.render(mastered, settings)
    assert [c.encoder for c in preview.codecs] == ["libvorbis", "aac"]
    assert len(preview.skipped) == 1 and "libopus" in preview.skipped[0]


def test_apple_encoder_is_preferred_when_ffmpeg_has_it() -> None:
    aac = next(c for c in streaming.CODECS if c.name == "aac")
    assert streaming.pick_encoder(aac, frozenset({"aac", "aac_at", "libopus"})) == "aac_at"
    assert streaming.pick_encoder(aac, frozenset({"aac"})) == "aac"
    assert streaming.pick_encoder(aac, frozenset({"libvorbis"})) is None


def test_an_analysis_has_nothing_to_preview(paths: Paths, settings: Settings, fx) -> None:
    analyzed = pipeline.analyze(fx["loop"], paths, settings)
    with pytest.raises(streaming.PreviewError, match="need a master"):
        streaming.render(analyzed, settings)
