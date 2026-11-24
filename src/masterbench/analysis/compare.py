"""Plain-language flags from frequency balance and stereo measurements."""

from __future__ import annotations

from masterbench.analysis.flags import Flag
from masterbench.analysis.stereo import LOW_END_HZ, Stereo
from masterbench.config import BANDS, REGIONS, Thresholds

BAND_LABELS = {
    "sub": "Sub",
    "bass": "Bass",
    "low_mids": "Low-mids",
    "mids": "Mids",
    "upper_mids": "Upper-mids",
    "presence": "Presence",
    "air": "Air",
}


def _range(name: str) -> str:
    low, high = BANDS[name]

    def hz(value: float) -> str:
        return f"{value / 1000:g} kHz" if value >= 1000 else f"{value:g} Hz"

    return f"{hz(low)} to {hz(high)}"


def band_label(name: str) -> str:
    return f"{BAND_LABELS[name]} ({_range(name)})"


def band_flags(deltas: dict[str, float], limits: Thresholds, against: str) -> list[Flag]:
    flags = []
    for name, delta in deltas.items():
        if abs(delta) < limits.band_delta_db:
            continue
        direction = "over" if delta > 0 else "under"
        message = (
            f"{band_label(name)} is {abs(delta):.1f} dB {direction} {against} at equal loudness."
        )
        flags.append(Flag(f"band_{name}_{direction}", "warn", message, delta))
    return flags


def stereo_flags(stereo: Stereo, limits: Thresholds) -> list[Flag]:
    flags = []
    if stereo.correlation < limits.phase_correlation:
        message = (
            f"Overall stereo correlation is {stereo.correlation:+.2f}: "
            "the mix partly cancels in mono. Check for phase-inverted or over-widened parts."
        )
        flags.append(Flag("phase_negative", "problem", message, stereo.correlation))
    elif stereo.correlation < limits.low_correlation:
        message = (
            f"Stereo correlation is low ({stereo.correlation:+.2f}); "
            "expect level and tone changes on mono speakers."
        )
        flags.append(Flag("wide_stereo", "warn", message, stereo.correlation))
    if stereo.low_mono_corr < limits.low_end_mono_corr:
        message = (
            f"The low end is not mono: correlation below {LOW_END_HZ:.0f} Hz is "
            f"{stereo.low_mono_corr:+.2f}. Sub energy in the sides is lost on clubs and phones."
        )
        flags.append(Flag("low_end_not_mono", "warn", message, stereo.low_mono_corr))
    for name, value in stereo.band_correlation.items():
        if value < limits.phase_correlation and name not in ("sub", "bass"):
            message = f"{band_label(name)} has negative stereo correlation ({value:+.2f})."
            flags.append(Flag(f"phase_{name}", "warn", message, value))
    return flags


_REGION_WORDS = {
    "mud": ("Mud", "Check for overlapping low-mids: guitars, pads, 808 harmonics and vocal body."),
    "harsh": ("Harshness", "Check cymbals, vocal presence and distorted synths."),
}


def region_flags(
    regions: dict[str, float] | None,
    ceiling: dict[str, float] | None,
    limits: Thresholds,
    against: str,
    where: str = "",
    start_s: float | None = None,
    end_s: float | None = None,
    threshold_db: float | None = None,
) -> list[Flag]:
    """Mud (200-500 Hz) and harshness (2-5 kHz) above what the comparison allows."""
    threshold = limits.region_excess_db if threshold_db is None else threshold_db
    flags = []
    for name, level in (regions or {}).items():
        if not ceiling or name not in ceiling:
            continue
        excess = level - ceiling[name]
        if excess < threshold:
            continue
        low, high = REGIONS[name]
        word, hint = _REGION_WORDS[name]
        span = (
            f"{low:g} to {high:g} Hz" if high < 1000 else f"{low / 1000:g} to {high / 1000:g} kHz"
        )
        message = f"{word}{where}: {span} is {excess:.1f} dB above {against}. {hint}"
        flags.append(Flag(name, "warn", message, round(excess, 2), start_s, end_s))
    return flags


def curve_flags(
    bands: dict[str, float],
    low: dict[str, float],
    high: dict[str, float],
    limits: Thresholds,
    against: str,
) -> list[Flag]:
    """A band is only an issue when it falls outside the range the references cover."""
    flags = []
    for name in BANDS:
        if bands[name] > high[name] + limits.curve_margin_db:
            distance, direction = bands[name] - high[name], "over"
        elif bands[name] < low[name] - limits.curve_margin_db:
            distance, direction = low[name] - bands[name], "under"
        else:
            continue
        message = f"{band_label(name)} is {distance:.1f} dB {direction} the range of {against}."
        flags.append(Flag(f"band_{name}_{direction}", "warn", message, round(distance, 2)))
    return flags
