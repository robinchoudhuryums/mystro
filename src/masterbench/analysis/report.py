"""AnalysisReport: everything measured about one audio file, serializable to JSON."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from masterbench import audio_io, validate
from masterbench.analysis import loudness
from masterbench.analysis.flags import Flag
from masterbench.analysis.loudness import Loudness
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
    flags = validate.check(audio, info, loud, settings.thresholds) if role == "mix" else []
    return AnalysisReport(
        source_path=str(source_path),
        role=role,
        sha256=sha256 or audio_io.file_sha256(source_path),
        info=info,
        loudness=loud,
        flags=flags,
    )


def analyze_file(path: Path, role: Role, settings: Settings) -> AnalysisReport:
    audio, info = audio_io.load(path)
    return analyze_audio(audio, info, path, role, settings)
