"""AnalysisReport: everything measured about one audio file, serializable to JSON."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from masterbench import audio_io, validate
from masterbench.analysis import (
    compare,
    detectors,
    loudness,
    lowend,
    musical,
    sections,
    spectrum,
    stereo,
)
from masterbench.analysis.flags import Flag
from masterbench.analysis.loudness import Loudness
from masterbench.analysis.lowend import LowEnd
from masterbench.analysis.musical import Musical
from masterbench.analysis.sections import Sections
from masterbench.analysis.spectrum import Spectrum
from masterbench.analysis.stereo import Stereo
from masterbench.audio_io import AudioInfo
from masterbench.config import Settings

if TYPE_CHECKING:
    from masterbench.references.target_curve import TargetCurve

Role = Literal["mix", "master", "reference", "stem"]
# Spotify's mastering tips (support.spotify.com, "Loudness normalization", read 2026-11-24).
SPOTIFY_LUFS = -14.0
SPOTIFY_LOUD_PEAK_DBTP = -2.0


@dataclass(frozen=True)
class AnalysisReport:
    source_path: str
    role: Role
    sha256: str
    info: AudioInfo
    loudness: Loudness
    flags: list[Flag] = field(default_factory=list)
    spectrum: Spectrum | None = None
    bands: dict[str, float] | None = None
    stereo: Stereo | None = None
    band_deltas: dict[str, float] | None = None
    short_term_lufs: list[float] | None = None
    regions: dict[str, float] | None = None
    musical: Musical | None = None
    sections: Sections | None = None
    lowend: LowEnd | None = None
    curve_deltas: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisReport:
        return cls(
            source_path=data["source_path"],
            role=data["role"],
            sha256=data["sha256"],
            info=AudioInfo(**data["info"]),
            loudness=Loudness(**data["loudness"]),
            flags=[Flag(**flag) for flag in data["flags"]],
            spectrum=Spectrum(**data["spectrum"]) if data.get("spectrum") else None,
            bands=data.get("bands"),
            stereo=Stereo(**data["stereo"]) if data.get("stereo") else None,
            band_deltas=data.get("band_deltas"),
            short_term_lufs=data.get("short_term_lufs"),
            regions=data.get("regions"),
            musical=Musical(**data["musical"]) if data.get("musical") else None,
            sections=Sections.from_dict(data["sections"]) if data.get("sections") else None,
            lowend=LowEnd.from_dict(data["lowend"]) if data.get("lowend") else None,
            curve_deltas=data.get("curve_deltas"),
        )

    def headline(self) -> str:
        loud = self.loudness
        return f"{loud.lufs_i:.1f} LUFS, {loud.true_peak_dbtp:+.1f} dBTP, LRA {loud.lra:.1f} LU"


def master_flags(loud: Loudness, target_lufs: float | None, ceiling_dbtp: float) -> list[Flag]:
    flags = []
    if loud.true_peak_dbtp > ceiling_dbtp + 0.05:
        message = (
            f"True peak {loud.true_peak_dbtp:+.2f} dBTP is over the {ceiling_dbtp:+.1f} ceiling."
        )
        flags.append(Flag("true_peak_over", "problem", message, loud.true_peak_dbtp))
    if target_lufs is not None and abs(loud.lufs_i - target_lufs) > 0.5:
        message = f"Integrated loudness {loud.lufs_i:.1f} LUFS missed the {target_lufs:.1f} target."
        flags.append(Flag("off_target", "warn", message, loud.lufs_i - target_lufs))
    if loud.lufs_i > SPOTIFY_LUFS + 0.5 and loud.true_peak_dbtp > SPOTIFY_LOUD_PEAK_DBTP:
        message = (
            f"At {loud.lufs_i:.1f} LUFS this is louder than Spotify's {SPOTIFY_LUFS:.0f}. "
            f"Spotify asks for a true peak under {SPOTIFY_LOUD_PEAK_DBTP:.0f} dBTP on louder "
            f"masters (this one is {loud.true_peak_dbtp:+.1f}), since they distort more when "
            "encoded, and it will turn the track down anyway."
        )
        flags.append(Flag("loud_for_spotify", "warn", message, loud.true_peak_dbtp))
    return flags


@dataclass(frozen=True)
class Overrides:
    """What the user has set by hand for a track; it wins over detection."""

    key: str | None = None
    markers: list[dict[str, Any]] | None = None
    envelope: list[float] | None = None


def _musical(
    audio: np.ndarray, info: AudioInfo, settings: Settings, key: str | None
) -> Musical | None:
    found = musical.analyze(
        audio, info.sample_rate, (settings.tempo_low_bpm, settings.tempo_high_bpm)
    )
    if found is None or key is None:
        return found
    return replace(found, key=key, key_confidence=1.0, key_alternatives=[])


def _sections(
    audio: np.ndarray, info: AudioInfo, tempo: float | None, overrides: Overrides
) -> Sections:
    if not overrides.markers:
        return sections.detect(audio, info.sample_rate, tempo)
    shift = 0.0
    if overrides.envelope:
        shift = sections.align_offset(
            overrides.envelope, sections.envelope(audio, info.sample_rate)
        )
    starts = [max(0.0, float(m["start_s"]) + shift) for m in overrides.markers]
    kinds = [str(m["kind"]) for m in overrides.markers]
    if starts[0] > 0.05:
        starts, kinds = [0.0, *starts], ["intro", *kinds]
    return sections.build(audio, info.sample_rate, starts, kinds, source="manual")


def song_shape(
    audio: np.ndarray, info: AudioInfo, settings: Settings, overrides: Overrides | None = None
) -> tuple[Musical | None, Sections]:
    """Key and tempo, and the song's sections, without the rest of the analysis."""
    overrides = overrides or Overrides()
    music = _musical(audio, info, settings, overrides.key)
    return music, _sections(audio, info, music.tempo_bpm if music else None, overrides)


def analyze_audio(
    audio: np.ndarray,
    info: AudioInfo,
    source_path: Path,
    role: Role,
    settings: Settings,
    sha256: str | None = None,
    overrides: Overrides | None = None,
    like: AnalysisReport | None = None,
) -> AnalysisReport:
    """Measure one file.

    Mixes get everything: loudness, balance, stereo, key and tempo, sections, the problem
    detectors and the low-end tools. References get all but the last two. A master passes
    the mix it came from as `like` and is measured over the same sections, with the same key.
    """
    overrides = overrides or Overrides()
    limits = settings.thresholds
    loud = loudness.measure(audio, info.sample_rate)
    spectra = spectrum.cross_spectra(audio, info.sample_rate)
    image = stereo.analyze(spectra)
    flags = validate.check(audio, info, loud, limits) if role == "mix" else []
    if role in ("mix", "master"):
        flags += compare.stereo_flags(image, limits)

    music, parts, low = None, None, None
    if role in ("mix", "reference"):
        music = _musical(audio, info, settings, overrides.key)
        parts = _sections(audio, info, music.tempo_bpm if music else None, overrides)
    elif like is not None and like.sections is not None:
        music = like.musical
        starts = [item.start_s for item in like.sections.items]
        kinds = [item.kind for item in like.sections.items]
        parts = sections.build(audio, info.sample_rate, starts, kinds, like.sections.source)
    if role == "mix":
        flags += detectors.run(audio, info.sample_rate, limits)
        low = lowend.analyze(audio, info.sample_rate, music.key if music else None, limits)
        flags += lowend.flags(low, limits)
    return AnalysisReport(
        source_path=str(source_path),
        role=role,
        sha256=sha256 or audio_io.file_sha256(source_path),
        info=info,
        loudness=loud,
        flags=flags,
        spectrum=spectrum.smoothed(spectra, loud.lufs_i),
        bands=spectrum.band_levels(spectra, loud.lufs_i),
        stereo=image,
        short_term_lufs=loudness.short_term_curve(audio, info.sample_rate),
        regions=spectrum.region_levels(spectra, loud.lufs_i),
        musical=music,
        sections=parts,
        lowend=low,
    )


def _hook(report: AnalysisReport) -> sections.Section | None:
    hooks = report.sections.of_kind("hook") if report.sections else []
    return hooks[0] if hooks else None


def against_reference(
    report: AnalysisReport,
    reference: AnalysisReport,
    settings: Settings,
    genre: str | None = None,
    judge_balance: bool = True,
) -> AnalysisReport:
    """Add band deltas versus a reference, section by section too, and the flags they raise.

    With judge_balance off the deltas are still recorded but raise no balance flags; that is
    for runs where a genre target curve does the judging instead.
    """
    if report.bands is None or reference.bands is None:
        return report
    limits = settings.thresholds
    deltas = spectrum.band_deltas(report.bands, reference.bands)
    flags = list(report.flags)
    parts = report.sections
    if parts is not None:
        parts = sections.compare(parts, reference.sections)
        found = sections.flags(parts, reference.sections, limits, genre)
        flags += [f for f in found if judge_balance or f.code == "weak_hook_contrast"]
    if judge_balance:
        flags += compare.band_flags(deltas, limits, "the reference")
        # One song is a looser yardstick than a curve's range, so it gets the band threshold.
        flags += compare.region_flags(
            report.regions,
            reference.regions,
            limits,
            "the reference",
            threshold_db=limits.band_delta_db,
        )
    return replace(report, band_deltas=deltas, sections=parts, flags=flags)


def against_curve(
    report: AnalysisReport, curve: TargetCurve, settings: Settings, genre: str | None = None
) -> AnalysisReport:
    """Judge the balance against a genre target curve: only what falls outside the range the
    references cover counts as an issue."""
    if report.bands is None:
        return report
    limits = settings.thresholds
    median = curve.band("median")
    deltas = {band: round(report.bands[band] - median[band], 2) for band in median}
    count = len(curve.references)
    against = f"your {count} {curve.genre} references"
    flags = list(report.flags)
    flags += compare.curve_flags(
        report.bands, curve.band("p25"), curve.band("p75"), limits, against
    )
    flags += compare.region_flags(report.regions, curve.region("p75"), limits, against)
    hook = _hook(report)
    if hook is not None and hook.regions:
        name = sections.display_kind("hook", genre)
        flags += [
            f
            for f in compare.region_flags(
                hook.regions,
                curve.region("p75"),
                limits,
                against,
                f" in the {name}",
                hook.start_s,
                hook.end_s,
            )
            if f.code not in {existing.code for existing in flags}
        ]
    parts = report.sections
    if parts and parts.contrast_lu is not None and curve.contrast_lu is not None:
        already = any(f.code == "weak_hook_contrast" for f in flags)
        if not already and parts.contrast_lu < curve.contrast_lu - limits.hook_contrast_gap_lu:
            first = parts.of_kind("hook")[0]
            hook_name, verse = (sections.display_kind(k, genre) for k in ("hook", "verse"))
            message = (
                f"The {hook_name} is only {parts.contrast_lu:.1f} LU louder than the {verse} "
                f"({against}: {curve.contrast_lu:.1f} LU). It may not lift when it arrives."
            )
            flags.append(
                Flag(
                    "weak_hook_contrast",
                    "warn",
                    message,
                    parts.contrast_lu,
                    first.start_s,
                    first.end_s,
                )
            )
    return replace(report, curve_deltas=deltas, flags=flags)


def analyze_file(path: Path, role: Role, settings: Settings) -> AnalysisReport:
    audio, info = audio_io.load(path)
    return analyze_audio(audio, info, path, role, settings)
