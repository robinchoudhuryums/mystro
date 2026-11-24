"""Song sections: proposed automatically or set by hand, then measured one by one."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any

import librosa
import numpy as np
from scipy import signal

from masterbench.analysis import loudness, musical, spectrum, stereo
from masterbench.analysis.flags import Flag
from masterbench.analysis.loudness import FLOOR_DB
from masterbench.config import BANDS, Thresholds

KINDS = ("intro", "verse", "hook", "bridge", "outro", "section")
# Electronic genres talk about drops and breaks rather than hooks and verses.
_ELECTRONIC_NAMES = {"hook": "drop", "verse": "break"}
ELECTRONIC_GENRES = frozenset({"house", "electronic", "techno", "dnb", "edm"})
ENVELOPE_STEP_S = 0.1
LOUDEST_WINDOW_S = 30.0
_MIN_SECTION_S = 4.0
_SECTION_TARGET_S = 12.0
_STEP_LU = 1.5
_SAME_TIMBRE = 0.5
_SAME_GROUP = 0.6
_HARMONY_WEIGHT = 0.25
_SNAP_REACH_S = 2.5
_HOOK_WITHIN_LU = 1.5
_EDGE_QUIETER_LU = 2.0


@dataclass(frozen=True)
class Section:
    label: str
    kind: str
    start_s: float
    end_s: float
    lufs: float
    bands: dict[str, float] | None = None
    correlation: float | None = None
    band_deltas: dict[str, float] | None = None
    regions: dict[str, float] | None = None


@dataclass(frozen=True)
class Sections:
    source: str
    items: list[Section]
    loudest_start_s: float
    loudest_end_s: float
    loudest_lufs: float
    contrast_lu: float | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sections:
        return cls(**{**data, "items": [Section(**item) for item in data["items"]]})

    def of_kind(self, kind: str) -> list[Section]:
        return [item for item in self.items if item.kind == kind]


def display_kind(kind: str, genre: str | None) -> str:
    if genre and genre.lower() in ELECTRONIC_GENRES:
        return _ELECTRONIC_NAMES.get(kind, kind)
    return kind


def envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Momentary loudness (LUFS) every 100 ms."""
    weighted = loudness.k_weight(audio, sample_rate)
    return loudness.windowed_loudness(weighted, sample_rate, loudness.MOMENTARY_S, ENVELOPE_STEP_S)


def _level(env: np.ndarray, start_s: float, end_s: float) -> float:
    """Loudness of a time span: the energy mean of its momentary values above the gate."""
    span = env[int(start_s / ENVELOPE_STEP_S) : max(int(end_s / ENVELOPE_STEP_S), 1)]
    span = span[span > -70.0]
    if len(span) == 0:
        return FLOOR_DB
    return float(10 * np.log10(np.mean(10 ** (span / 10))))


def loudest_window(env: np.ndarray, duration_s: float) -> tuple[float, float, float]:
    """Start, end and loudness of the loudest 30 seconds (the whole track if shorter)."""
    width = int(min(LOUDEST_WINDOW_S, duration_s) / ENVELOPE_STEP_S)
    if width < 1 or len(env) <= width:
        return 0.0, duration_s, _level(env, 0.0, duration_s)
    energy = np.concatenate([[0.0], np.cumsum(10 ** (env / 10))])
    means = (energy[width:] - energy[:-width]) / width
    start = int(np.argmax(means))
    start_s = start * ENVELOPE_STEP_S
    return start_s, start_s + width * ENVELOPE_STEP_S, float(10 * np.log10(means[start]))


def _features(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame times and standardized timbre and harmony features, smoothed over a second."""
    y = musical.to_mono(audio, sample_rate)
    hop = 2048
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = librosa.feature.mfcc(y=y, sr=musical.ANALYSIS_RATE, n_mfcc=13, hop_length=hop)
        chroma = librosa.feature.chroma_stft(y=y, sr=musical.ANALYSIS_RATE, hop_length=hop)
    times = librosa.frames_to_time(
        np.arange(mfcc.shape[1]), sr=musical.ANALYSIS_RATE, hop_length=hop
    )
    features = np.vstack([mfcc, chroma])
    # About one second of smoothing, so single drum hits do not become boundaries.
    width = max(1, int(musical.ANALYSIS_RATE / hop))
    kernel = np.ones(width) / width
    features = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, features)
    spread = features.std(axis=1, keepdims=True)
    features = (features - features.mean(axis=1, keepdims=True)) / np.where(spread > 0, spread, 1)
    # Chords change inside a section, so harmony counts for little when comparing sections.
    features[len(mfcc) :] *= _HARMONY_WEIGHT
    return times, features


def _boundaries(times: np.ndarray, features: np.ndarray, env: np.ndarray) -> list[float]:
    """Candidate section starts from timbre, harmony and level, at roughly one per 12 s."""
    count = int(np.clip(round(times[-1] / _SECTION_TARGET_S), 1, 16))
    if count < 2:
        return [0.0]
    level = np.interp(times, np.arange(len(env)) * ENVELOPE_STEP_S, np.maximum(env, -70.0))
    stacked = np.vstack([features, 3.0 * (level - level.mean())[None, :] / 10.0])
    frames = librosa.segment.agglomerative(stacked, min(count, stacked.shape[1]))
    return [float(t) for t in times[frames]]


def _centroid(times: np.ndarray, features: np.ndarray, start: float, end: float) -> np.ndarray:
    inside = (times >= start + 1.0) & (times < end - 1.0)
    return features[:, inside if inside.any() else (times >= start) & (times < end)].mean(axis=1)


def _timbre_gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _phrase_s(tempo_bpm: float | None) -> float:
    """Eight bars of 4/4, the usual length of a loop: the scale at which sections differ.

    Comparing eight bars before a point with eight bars after it averages over everything a
    loop does inside itself (chord changes, a bass line, fills), so only a change of section
    shows up. Without a tempo, 12 seconds stands in.
    """
    if not tempo_bpm:
        return 12.0
    return float(np.clip(32 * 60.0 / tempo_bpm, 8.0, 24.0))


def _step_curve(env: np.ndarray, span_s: float) -> np.ndarray:
    """The level step at each envelope point: the phrase after against the phrase before."""
    span = int(span_s / ENVELOPE_STEP_S)
    energy = np.concatenate([[0.0], np.cumsum(10 ** (np.maximum(env, -70.0) / 10))])
    curve = np.zeros(len(env))
    for i in range(1, len(env)):
        before = (energy[i] - energy[max(0, i - span)]) / min(span, i)
        after = (energy[min(len(env), i + span)] - energy[i]) / min(span, len(env) - i)
        curve[i] = abs(10 * np.log10(after / before))
    edge = int(_MIN_SECTION_S / ENVELOPE_STEP_S)
    curve[:edge] = curve[-edge:] = 0.0
    return curve


def _level_steps(curve: np.ndarray) -> list[float]:
    """Times where the level steps by more than 1.5 LU and stays there."""
    spacing = int(_MIN_SECTION_S / ENVELOPE_STEP_S)
    peaks, _ = signal.find_peaks(curve, height=_STEP_LU, distance=spacing, prominence=_STEP_LU)
    # A momentary value describes the 400 ms that start at its index.
    return [round(float(i) * ENVELOPE_STEP_S + loudness.MOMENTARY_S / 2, 2) for i in peaks]


def _supported(
    start: float,
    curve: np.ndarray,
    times: np.ndarray,
    features: np.ndarray,
    duration_s: float,
    span_s: float,
) -> bool:
    """A boundary is real when the level steps there, or the sound changes and stays changed."""
    i = int(start / ENVELOPE_STEP_S)
    if curve[max(0, i - 10) : i + 11].max() >= _STEP_LU:
        return True
    before = (times >= max(0.0, start - span_s)) & (times < start - 0.5)
    after = (times >= start + 0.5) & (times < min(duration_s, start + span_s))
    if not before.any() or not after.any():
        return False
    gap = _timbre_gap(features[:, before].mean(axis=1), features[:, after].mean(axis=1))
    return gap >= _SAME_TIMBRE


def _candidates(starts: list[float], steps: list[float]) -> list[float]:
    """Level steps are precise; timbre boundaries are coarse and yield to a step within 2.5 s."""
    kept = set(steps)
    for start in starts:
        if start > 0 and not any(abs(start - step) <= _SNAP_REACH_S for step in steps):
            kept.add(round(start, 2))
    return sorted(kept)


def _drop_short(starts: list[float], duration_s: float) -> list[float]:
    starts = [0.0, *starts]
    i = 1
    while i < len(starts):
        end = starts[i + 1] if i + 1 < len(starts) else duration_s
        if starts[i] - starts[i - 1] < _MIN_SECTION_S or end - starts[i] < _MIN_SECTION_S:
            del starts[i]
        else:
            i += 1
    return starts


def _kinds(levels: list[float], lengths: list[float], centroids: list[np.ndarray]) -> list[str]:
    """Guess each section's kind.

    Quiet or short sections at either end are the intro and outro. The rest are grouped by
    timbre, so repeats of the same material land together; the loudest group is the hook.
    Grouping by sound rather than by level is what lets a hook that barely lifts still be
    recognised as the hook.
    """
    count = len(levels)
    if count == 1:
        return ["section"]
    kinds = ["verse"] * count
    typical = float(np.median(lengths))
    body = float(np.median(levels[1:-1])) if count > 2 else max(levels)
    for edge, name in ((0, "intro"), (count - 1, "outro")):
        if levels[edge] < body - _EDGE_QUIETER_LU or (count > 2 and lengths[edge] < 0.6 * typical):
            kinds[edge] = name
    inner = [i for i in range(count) if kinds[i] == "verse"]
    if not inner:
        return ["section"] * count

    groups: list[list[int]] = []
    for i in inner:
        home = next(
            (g for g in groups if _timbre_gap(centroids[g[0]], centroids[i]) < _SAME_GROUP), None
        )
        if home is None:
            groups.append([i])
        else:
            home.append(i)
    if len(groups) == 1:
        top = max(levels[i] for i in inner)
        hooks = [i for i in inner if levels[i] >= top - _HOOK_WITHIN_LU]
        if len(hooks) == len(inner):
            for i in inner:
                kinds[i] = "section"
            return kinds
    else:
        hooks = max(groups, key=lambda g: float(np.mean([levels[i] for i in g])))
    for i in hooks:
        kinds[i] = "hook"
    verses = [levels[i] for i in inner if kinds[i] == "verse"]
    first_hook = kinds.index("hook")
    for i in range(first_hook + 1, count - 1):
        if kinds[i] == "verse" and levels[i] < float(np.median(verses)) - 3.0:
            kinds[i] = "bridge"
    return kinds


def _labels(kinds: list[str]) -> list[str]:
    totals = {kind: kinds.count(kind) for kind in set(kinds)}
    seen: dict[str, int] = {}
    labels = []
    for kind in kinds:
        seen[kind] = seen.get(kind, 0) + 1
        labels.append(f"{kind} {seen[kind]}" if totals[kind] > 1 else kind)
    return labels


def build(
    audio: np.ndarray,
    sample_rate: int,
    starts: list[float],
    kinds: list[str] | None = None,
    source: str = "auto",
    env: np.ndarray | None = None,
    features: tuple[np.ndarray, np.ndarray] | None = None,
) -> Sections:
    """Measure the sections that begin at the given times. Kinds are guessed when not given."""
    duration = len(audio) / sample_rate
    env = envelope(audio, sample_rate) if env is None else env
    starts = sorted({0.0, *[round(s, 2) for s in starts if 0 < s < duration]})
    ends = [*starts[1:], round(duration, 2)]
    levels = [_level(env, a, b) for a, b in zip(starts, ends, strict=True)]
    lengths = [b - a for a, b in zip(starts, ends, strict=True)]
    if kinds is None or len(kinds) != len(starts):
        times, feats = features or _features(audio, sample_rate)
        centroids = [_centroid(times, feats, a, b) for a, b in zip(starts, ends, strict=True)]
        kinds = _kinds(levels, lengths, centroids)
    items = []
    for label, kind, start, end, level in zip(
        _labels(kinds), kinds, starts, ends, levels, strict=True
    ):
        piece = audio[int(start * sample_rate) : int(end * sample_rate)]
        bands, correlation, regions = None, None, None
        if len(piece) >= sample_rate and level > -70.0:
            spectra = spectrum.cross_spectra(piece, sample_rate)
            bands = spectrum.band_levels(spectra, level)
            regions = spectrum.region_levels(spectra, level)
            correlation = stereo.analyze(spectra).correlation
        items.append(
            Section(label, kind, start, end, round(level, 2), bands, correlation, regions=regions)
        )
    hooks = [item.lufs for item in items if item.kind == "hook"]
    verses = [item.lufs for item in items if item.kind == "verse"]
    contrast = round(float(np.mean(hooks) - np.mean(verses)), 2) if hooks and verses else None
    window = loudest_window(env, duration)
    return Sections(
        source, items, round(window[0], 2), round(window[1], 2), round(window[2], 2), contrast
    )


def detect(audio: np.ndarray, sample_rate: int, tempo_bpm: float | None = None) -> Sections:
    """Propose sections automatically. The kinds are guesses from level, sound and position."""
    env = envelope(audio, sample_rate)
    duration = len(audio) / sample_rate
    if duration < 2 * _MIN_SECTION_S:
        return build(audio, sample_rate, [0.0], env=env)
    times, features = _features(audio, sample_rate)
    span = _phrase_s(tempo_bpm)
    curve = _step_curve(env, span)
    found = _candidates(_boundaries(times, features, env), _level_steps(curve))
    found = [s for s in found if _supported(s, curve, times, features, duration, span)]
    starts = _drop_short(found, duration)
    return build(audio, sample_rate, starts, env=env, features=(times, features))


def align_offset(saved: list[float], current: np.ndarray, max_shift_s: float = 15.0) -> float:
    """Seconds to add to saved marker times so they fit a new bounce of the same song.

    Compares the two loudness envelopes; a new bounce that starts later or earlier (extra
    silence, a trimmed count-in) shifts every marker by the same amount.
    """
    a = np.maximum(np.asarray(saved, dtype=float), -70.0)
    b = np.maximum(current, -70.0)
    reach = int(max_shift_s / ENVELOPE_STEP_S)
    best_shift, best_score = 0, 0.0
    for shift in range(-reach, reach + 1):
        x = a[max(0, -shift) :]
        y = b[max(0, shift) :]
        n = min(len(x), len(y))
        if n < 50 or x[:n].std() == 0 or y[:n].std() == 0:
            continue
        score = float(np.corrcoef(x[:n], y[:n])[0, 1])
        if score > best_score + 1e-9:
            best_shift, best_score = shift, score
    return round(best_shift * ENVELOPE_STEP_S, 2)


def compare(mine: Sections, reference: Sections | None) -> Sections:
    """Band deltas of each section against the reference's sections of the same kind."""
    if reference is None:
        return mine
    items = []
    for item in mine.items:
        peers = [p.bands for p in reference.of_kind(item.kind) if p.bands]
        if item.bands is None or not peers:
            items.append(item)
            continue
        deltas = {b: round(item.bands[b] - float(np.mean([p[b] for p in peers])), 2) for b in BANDS}
        items.append(replace(item, band_deltas=deltas))
    return replace(mine, items=items)


def flags(
    mine: Sections, reference: Sections | None, limits: Thresholds, genre: str | None = None
) -> list[Flag]:
    """Hook-versus-verse contrast, and the hook's balance against the reference's hook."""
    found: list[Flag] = []
    hook, verse = display_kind("hook", genre), display_kind("verse", genre)
    theirs = reference.contrast_lu if reference else None
    if mine.contrast_lu is not None:
        weak_alone = theirs is None and mine.contrast_lu < limits.min_hook_contrast_lu
        weak_vs_reference = (
            theirs is not None and mine.contrast_lu < theirs - limits.hook_contrast_gap_lu
        )
        if weak_alone or weak_vs_reference:
            versus = f" (reference: {theirs:.1f} LU)" if theirs is not None else ""
            first = mine.of_kind("hook")[0]
            message = (
                f"The {hook} is only {mine.contrast_lu:.1f} LU louder than the {verse}{versus}. "
                f"It may not lift when it arrives."
            )
            found.append(
                Flag(
                    "weak_hook_contrast",
                    "warn",
                    message,
                    mine.contrast_lu,
                    first.start_s,
                    first.end_s,
                )
            )
    for item in mine.of_kind("hook")[:1]:
        for band, delta in (item.band_deltas or {}).items():
            if abs(delta) < limits.band_delta_db:
                continue
            word = "more" if delta > 0 else "less"
            name = band.replace("_", "-")
            message = f"The {hook} has {abs(delta):.1f} dB {word} {name} than the reference {hook}."
            code = f"hook_{band}_{'over' if delta > 0 else 'under'}"
            found.append(Flag(code, "warn", message, delta, item.start_s, item.end_s))
    return found
