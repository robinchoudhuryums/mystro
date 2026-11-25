"""Suggest the references that sit closest to a mix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from masterbench.analysis.report import AnalysisReport
from masterbench.config import Paths
from masterbench.references import library
from masterbench.references.library import Reference, ReferenceError

# (centre, scale) per feature: values are centred and scaled before the cosine distance, so
# one unit means "a difference you would notice" on every axis.
_SCALES: dict[str, tuple[float, float]] = {
    "tempo": (0.0, 0.12),
    "mode": (0.0, 1.0),
    "sub": (-6.0, 3.0),
    "bass": (-6.0, 3.0),
    "low_mids": (-14.0, 3.0),
    "mids": (-14.0, 3.0),
    "upper_mids": (-18.0, 3.0),
    "presence": (-24.0, 3.0),
    "air": (-28.0, 3.0),
    "lufs": (-12.0, 6.0),
    "crest": (10.0, 4.0),
    "width": (-12.0, 6.0),
}
_GROUPS = {
    "tempo": ("tempo",),
    "key": ("mode",),
    "low-end balance": ("sub", "bass"),
    "midrange balance": ("low_mids", "mids", "upper_mids"),
    "top-end balance": ("presence", "air"),
    "dynamics": ("crest",),
    "stereo width": ("width",),
}


@dataclass(frozen=True)
class Suggestion:
    reference: Reference
    similarity: float
    reasons: list[str]

    @property
    def why(self) -> str:
        return "closest " + " and ".join(self.reasons) if self.reasons else ""


def features(report: AnalysisReport) -> dict[str, float] | None:
    """The numbers a mix and a reference are compared on."""
    if report.bands is None or report.stereo is None:
        return None
    found = {
        **report.bands,
        "lufs": report.loudness.lufs_i,
        "crest": report.loudness.crest_db,
        "width": max(report.stereo.side_mid_db, -40.0),
    }
    if report.musical is not None:
        # Tempo on a log scale, folded to one octave: 70 and 140 BPM are the same feel.
        found["tempo"] = float(np.mod(np.log2(report.musical.tempo_bpm / 120.0) + 0.5, 1.0) - 0.5)
        found["mode"] = 1.0 if report.musical.key.endswith("minor") else -1.0
    return found


def _vector(found: dict[str, float], names: list[str]) -> np.ndarray:
    return np.array([(found[name] - _SCALES[name][0]) / _SCALES[name][1] for name in names])


def suggest(mix: AnalysisReport, references: list[Reference], count: int = 3) -> list[Suggestion]:
    """The references most like the mix, by cosine similarity of normalized features.

    Loudness is left out of the ranking: an unmastered mix is quieter than every reference.
    """
    mine = features(mix)
    if mine is None:
        return []
    ranked = []
    for reference in references:
        theirs = features(reference.report)
        if theirs is None or reference.report.sha256 == mix.sha256:
            continue
        names = [n for n in _SCALES if n in mine and n in theirs and n != "lufs"]
        a, b = _vector(mine, names), _vector(theirs, names)
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        similarity = float(np.dot(a, b) / norm) if norm > 0 else 0.0
        gaps = {
            group: float(np.mean([abs(a[names.index(n)] - b[names.index(n)]) for n in members]))
            for group, members in _GROUPS.items()
            if all(n in names for n in members)
        }
        reasons = [g for g, gap in sorted(gaps.items(), key=lambda item: item[1])[:2] if gap < 1.0]
        ranked.append(Suggestion(reference, round(similarity, 3), reasons))
    ranked.sort(key=lambda s: -s.similarity)
    return ranked[:count]


@dataclass(frozen=True)
class Choice:
    """The reference picked for a mix out of a genre's bank, and what it was picked over."""

    reference: Reference
    genre: str
    similarity: float | None
    why: str
    others: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asked": f"{self.genre}/{library.AUTO}",
            "chosen": self.reference.spec,
            "similarity": self.similarity,
            "why": self.why,
            "others": [{"reference": spec, "similarity": score} for spec, score in self.others],
        }


def choose(paths: Paths, mix: AnalysisReport, genre: str) -> Choice:
    """Pick the reference in a genre's bank that sits closest to a mix.

    References badged poor (lossy rips, clipped, mono) are passed over while a better one
    exists. A mix that is itself in the bank is never matched to itself.
    """
    bank = library.list_references(paths, genre)
    if not bank:
        raise ReferenceError(
            f"no references in genre '{genre}' to choose from; add one with: "
            f"masterbench ref add FILE --genre {genre}"
        )
    sound = [r for r in bank if r.quality is None or r.quality.usable_for_curves]
    ranked = suggest(mix, sound or bank, count=len(bank))
    if ranked:
        best = ranked[0]
        only = len(sound or bank) == 1
        why = "the only reference in the genre" if only else best.why or "nearest overall"
        others = [(s.reference.spec, s.similarity) for s in ranked[1:4]]
        return Choice(best.reference, best.reference.genre, best.similarity, why, others)
    others_exist = [r for r in bank if r.report.sha256 != mix.sha256]
    if not others_exist:
        raise ReferenceError(f"the only reference in genre '{genre}' is this very file")
    # Nothing to compare on (a reference analyzed before balance was measured): fall back.
    fallback = next((r for r in others_exist if r.is_default), others_exist[0])
    return Choice(fallback, fallback.genre, None, "could not be compared; the genre default")
