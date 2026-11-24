"""Stem balance: how loud the vocal, drums and bass sit, and how that compares to a reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from masterbench.analysis import loudness
from masterbench.analysis.flags import Flag
from masterbench.config import Thresholds
from masterbench.stems.separate import StemSet

_LABELS = {
    "vocal_to_instrumental_db": "The vocal, relative to the beat,",
    "drums_to_total_db": "The drums, relative to the whole mix,",
    "bass_to_total_db": "The bass, relative to the whole mix,",
}


@dataclass(frozen=True)
class StemBalance:
    source: str
    separated: bool
    levels_db: dict[str, float]
    vocal_to_instrumental_db: float | None
    drums_to_total_db: float | None
    bass_to_total_db: float | None
    reconstruction_db: float | None

    def ratios(self) -> dict[str, float]:
        named = {key: getattr(self, key) for key in _LABELS}
        return {key: value for key, value in named.items() if value is not None}


def _lufs(audio: np.ndarray, sample_rate: int) -> float:
    return loudness.integrated(audio, sample_rate)


def _role_sum(stem_set: StemSet, roles: tuple[str, ...]) -> np.ndarray | None:
    names = [name for role in roles for name in stem_set.by_role(role)]
    if not names:
        return None
    return sum((stem_set.load(name) for name in names), np.zeros((stem_set.frames, 2)))


def balance(stem_set: StemSet, mix: np.ndarray | None = None) -> StemBalance:
    """Loudness of each stem relative to the full mix, and the three headline ratios.

    With the original mix given, reconstruction_db says how far the stems' sum is from it
    (in dB below the mix): a measure of what separation lost or invented.
    """
    rate = stem_set.sample_rate
    total = sum((stem_set.load(name) for name in stem_set.stems), np.zeros((stem_set.frames, 2)))
    total_lufs = _lufs(total, rate)
    levels = {
        name: round(_lufs(stem_set.load(name), rate) - total_lufs, 2) for name in stem_set.stems
    }

    def relative(roles: tuple[str, ...], against: np.ndarray | None) -> float | None:
        part = _role_sum(stem_set, roles)
        if part is None or against is None:
            return None
        return round(_lufs(part, rate) - _lufs(against, rate), 2)

    vocals = _role_sum(stem_set, ("vocals",))
    beat = total - vocals if vocals is not None else None
    reconstruction = None
    if mix is not None and len(mix) == stem_set.frames:
        stereo = mix if mix.shape[1] == 2 else np.repeat(mix[:, :1], 2, axis=1)
        error = float(np.mean((total - stereo) ** 2))
        power = float(np.mean(stereo**2))
        reconstruction = round(loudness.to_db(error / power, power=True), 1) if power else None
    return StemBalance(
        source=stem_set.source,
        separated=stem_set.separated,
        levels_db=levels,
        vocal_to_instrumental_db=relative(("vocals",), beat),
        drums_to_total_db=relative(("drums",), total),
        bass_to_total_db=relative(("bass",), total),
        reconstruction_db=reconstruction,
    )


def compare(mine: StemBalance, reference: StemBalance, limits: Thresholds) -> list[Flag]:
    """Flags for ratios that sit away from the reference's by more than the threshold."""
    flags = []
    theirs = reference.ratios()
    for key, value in mine.ratios().items():
        if key not in theirs:
            continue
        delta = value - theirs[key]
        if abs(delta) < limits.stem_balance_db:
            continue
        direction = "higher" if delta > 0 else "lower"
        message = (
            f"{_LABELS[key]} sits {abs(delta):.1f} dB {direction} than in the reference "
            f"({value:+.1f} dB against {theirs[key]:+.1f} dB)."
        )
        code = key.removesuffix("_db") + ("_high" if delta > 0 else "_low")
        flags.append(Flag(code, "warn", message, round(delta, 2)))
    return flags


def artifact_warning(balance_: StemBalance) -> str | None:
    if not balance_.separated:
        return None
    text = (
        "These stems were separated by a model, not exported from Logic. Expect bleed and "
        "artifacts; treat the levels as estimates and keep rebalancing moves small."
    )
    if balance_.reconstruction_db is not None and balance_.reconstruction_db > -20:
        text += (
            f" The stems only sum back to within {balance_.reconstruction_db:.0f} dB of the mix."
        )
    return text
