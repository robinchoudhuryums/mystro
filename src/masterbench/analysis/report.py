"""AnalysisReport: everything measured about one audio file, serializable to JSON."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from masterbench import audio_io, validate
from masterbench.analysis import compare, loudness, spectrum, stereo
from masterbench.analysis.flags import Flag
from masterbench.analysis.loudness import Loudness
from masterbench.analysis.spectrum import Spectrum
from masterbench.analysis.stereo import Stereo
from masterbench.audio_io import AudioInfo
from masterbench.config import Settings

Role = Literal["mix", "master", "reference", "stem"]


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
    return flags


def analyze_audio(
    audio: np.ndarray,
    info: AudioInfo,
    source_path: Path,
    role: Role,
    settings: Settings,
    sha256: str | None = None,
) -> AnalysisReport:
    loud = loudness.measure(audio, info.sample_rate)
    spectra = spectrum.cross_spectra(audio, info.sample_rate)
    image = stereo.analyze(spectra)
    flags = validate.check(audio, info, loud, settings.thresholds) if role == "mix" else []
    if role in ("mix", "master"):
        flags += compare.stereo_flags(image, settings.thresholds)
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
    )


def against_reference(
    report: AnalysisReport, reference: AnalysisReport, settings: Settings
) -> AnalysisReport:
    """Add band deltas versus a reference, and the flags they raise."""
    if report.bands is None or reference.bands is None:
        return report
    deltas = spectrum.band_deltas(report.bands, reference.bands)
    flags = compare.band_flags(deltas, settings.thresholds, "the reference")
    return replace(report, band_deltas=deltas, flags=[*report.flags, *flags])


def analyze_file(path: Path, role: Role, settings: Settings) -> AnalysisReport:
    audio, info = audio_io.load(path)
    return analyze_audio(audio, info, path, role, settings)
