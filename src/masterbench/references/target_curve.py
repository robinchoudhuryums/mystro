"""Genre target curves: the median spectrum of several references and the range they cover.

Every reference's spectrum is stored relative to its own integrated loudness, so taking
percentiles across references compares them at equal loudness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from masterbench import projects
from masterbench.analysis.report import AnalysisReport
from masterbench.config import BANDS, REGIONS, Paths
from masterbench.references import library
from masterbench.references.library import Reference

MIN_REFERENCES = 2


class CurveError(LookupError):
    pass


@dataclass(frozen=True)
class Range:
    median: float
    p25: float
    p75: float


@dataclass(frozen=True)
class TargetCurve:
    genre: str
    created_at: str
    references: list[str]
    freqs_hz: list[float]
    median_db: list[float]
    p25_db: list[float]
    p75_db: list[float]
    bands: dict[str, Range]
    regions: dict[str, Range]
    hook_bands: dict[str, Range] | None = None
    contrast_lu: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetCurve:
        def ranges(raw: dict[str, Any] | None) -> dict[str, Range] | None:
            return {k: Range(**v) for k, v in raw.items()} if raw else None

        return cls(
            **{
                **data,
                "bands": ranges(data["bands"]),
                "regions": ranges(data["regions"]),
                "hook_bands": ranges(data.get("hook_bands")),
            }
        )

    def band(self, which: str) -> dict[str, float]:
        """One figure per band: 'median', 'p25' or 'p75'."""
        return {name: getattr(value, which) for name, value in self.bands.items()}

    def region(self, which: str) -> dict[str, float]:
        return {name: getattr(value, which) for name, value in self.regions.items()}


def _range(values: list[float]) -> Range:
    low, mid, high = np.percentile(values, [25, 50, 75])
    return Range(round(float(mid), 2), round(float(low), 2), round(float(high), 2))


def from_reports(genre: str, reports: dict[str, AnalysisReport]) -> TargetCurve:
    """Build a curve from analyzed references (name -> report)."""
    usable = {n: r for n, r in reports.items() if r.spectrum and r.bands and r.regions}
    if len(usable) < MIN_REFERENCES:
        raise CurveError(
            f"a target curve needs at least {MIN_REFERENCES} usable references in '{genre}', "
            f"found {len(usable)}"
        )
    spectra = [r.spectrum for r in usable.values() if r.spectrum]
    points = min(len(s.freqs_hz) for s in spectra)
    stacked = np.array([s.level_db[:points] for s in spectra])
    low, mid, high = np.percentile(stacked, [25, 50, 75], axis=0)
    hooks = [
        section.bands
        for report in usable.values()
        if report.sections
        for section in report.sections.of_kind("hook")
        if section.bands
    ]
    contrasts = [
        r.sections.contrast_lu
        for r in usable.values()
        if r.sections and r.sections.contrast_lu is not None
    ]
    return TargetCurve(
        genre=genre,
        created_at=projects.now_iso(),
        references=sorted(usable),
        freqs_hz=spectra[0].freqs_hz[:points],
        median_db=[round(float(v), 2) for v in mid],
        p25_db=[round(float(v), 2) for v in low],
        p75_db=[round(float(v), 2) for v in high],
        bands={b: _range([r.bands[b] for r in usable.values() if r.bands]) for b in BANDS},
        regions={g: _range([r.regions[g] for r in usable.values() if r.regions]) for g in REGIONS},
        hook_bands={b: _range([h[b] for h in hooks]) for b in BANDS} if len(hooks) >= 2 else None,
        contrast_lu=round(float(np.median(contrasts)), 2) if contrasts else None,
    )


def eligible(references: list[Reference], include_low_quality: bool = False) -> list[Reference]:
    """References fit for a curve: everything except those badged poor, unless asked."""
    return [
        ref
        for ref in references
        if include_low_quality or ref.quality is None or ref.quality.usable_for_curves
    ]


def build(
    paths: Paths,
    genre: str,
    names: list[str] | None = None,
    include_low_quality: bool = False,
) -> TargetCurve:
    """Build and save the curve for a genre from its library references."""
    references = library.list_references(paths, genre)
    if not references:
        raise CurveError(f"no references in genre '{genre}'")
    if names:
        wanted = {name.lower() for name in names}
        chosen = [ref for ref in references if ref.name.lower() in wanted]
        missing = wanted - {ref.name.lower() for ref in chosen}
        if missing:
            raise CurveError(f"no such references in '{genre}': {', '.join(sorted(missing))}")
    else:
        chosen = eligible(references, include_low_quality)
    curve = from_reports(references[0].genre, {ref.name: ref.report for ref in chosen})
    projects.write_json(paths.target_curves / f"{curve.genre}.json", asdict(curve))
    return curve


def load(paths: Paths, genre: str | None) -> TargetCurve | None:
    """The saved curve for a genre, if one has been built."""
    if not genre:
        return None
    wanted = library.normalize(genre)
    for file in sorted(paths.target_curves.glob("*.json")):
        if library.normalize(file.stem) == wanted:
            return TargetCurve.from_dict(json.loads(file.read_text(encoding="utf-8")))
    return None


def get(paths: Paths, genre: str) -> TargetCurve:
    curve = load(paths, genre)
    if curve is None:
        raise CurveError(
            f"no target curve for '{genre}' yet; build one with: masterbench curve build {genre}"
        )
    return curve
