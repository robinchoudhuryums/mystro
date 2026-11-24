"""Alternate versions: an instrumental and an a cappella that got the master's processing.

Mastering a stem on its own would process it differently from the full mix, and the parts
would no longer add up to the master. Here the processing is locked first, as one fixed
linear filter and gain, and then applied to each part.

- A curve-match run stored its EQ and gain, so they are applied exactly.
- A Matchering run (or any run with compression or saturation in its chains) is adaptive.
  Its overall effect is fitted as the linear filter that best turns the mix into the master,
  for mid and side separately, since Matchering treats them separately.

The check is that instrumental + a cappella, before limiting, comes out as the master.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from masterbench import audio_io, projects
from masterbench.analysis import loudness
from masterbench.analysis.flags import Flag
from masterbench.config import Paths, Settings
from masterbench.master import chain, curve_match, limiter, presets
from masterbench.master.pipeline import RunResult
from masterbench.stems import workflow
from masterbench.stems.separate import StemSet

SUMMARY_FILE = "versions.json"
LINEAR_MODULES = frozenset({"highpass", "lowpass", "low_shelf", "high_shelf", "peak", "gain"})
_FIT_TAPS = 8192
# How far the parts' sum may sit from the master before the result is called into question.
GOOD_SUM_DB = -25.0
POOR_SUM_DB = -15.0

Process = Callable[[np.ndarray], np.ndarray]


class VersionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Version:
    name: str
    file: str
    lufs_i: float
    true_peak_dbtp: float
    limiter_reduction_db: float


@dataclass(frozen=True)
class Versions:
    run_id: str
    stem_set: str
    separated: bool
    method: str
    sum_error_db: float
    stems_vs_mix_db: float
    versions: list[Version]
    flags: list[Flag]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Versions:
        return cls(
            **{
                **data,
                "versions": [Version(**v) for v in data["versions"]],
                "flags": [Flag(**f) for f in data["flags"]],
            }
        )


def versions_dir(run: RunResult) -> Path:
    return run.run_dir.parent.parent / "versions" / run.manifest.id


def error_db(estimate: np.ndarray, target: np.ndarray) -> float:
    """Power of the difference relative to the target, in dB. Lower is closer."""
    power = float(np.mean(target**2))
    if power <= 0:
        return 0.0
    return loudness.to_db(float(np.mean((estimate - target) ** 2)) / power, power=True)


def _mid_side(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (audio[:, 0] + audio[:, 1]) / 2, (audio[:, 0] - audio[:, 1]) / 2


def _fit(source: np.ndarray, target: np.ndarray, sample_rate: int) -> np.ndarray:
    """The FIR that best turns source into target in the least-squares sense."""
    _, auto = signal.welch(source, sample_rate, nperseg=_FIT_TAPS)
    _, cross = signal.csd(source, target, sample_rate, nperseg=_FIT_TAPS)
    if auto.max() <= 0:
        return np.zeros(_FIT_TAPS)
    response = np.fft.irfft(cross / (auto + 1e-6 * auto.max()), _FIT_TAPS)
    return np.roll(response, _FIT_TAPS // 2) * np.hanning(_FIT_TAPS)


def _filter(audio: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    delay = len(kernel) // 2
    return signal.fftconvolve(audio, kernel, mode="full")[delay : delay + len(audio)]


def fitted_process(mix: np.ndarray, master: np.ndarray, sample_rate: int) -> Process:
    """A fixed mid/side filter pair that reproduces what turned the mix into the master."""
    mix_mid, mix_side = _mid_side(mix)
    master_mid, master_side = _mid_side(master)
    for_mid = _fit(mix_mid, master_mid, sample_rate)
    for_side = _fit(mix_side, master_side, sample_rate)

    def process(audio: np.ndarray) -> np.ndarray:
        mid, side = _mid_side(audio)
        mid, side = _filter(mid, for_mid), _filter(side, for_side)
        return np.column_stack([mid + side, mid - side])

    return process


def _stored_chain(data: dict[str, Any] | None) -> chain.ChainPreset | None:
    return presets.from_dict(data, data.get("name", "chain")) if data else None


def _is_linear(preset: chain.ChainPreset | None) -> bool:
    return preset is None or all(m.type in LINEAR_MODULES for m in preset.modules)


def locked_process(
    run: RunResult, mix: np.ndarray, master: np.ndarray, sample_rate: int
) -> tuple[Process, str]:
    """The run's processing as a fixed linear operation, and how it was obtained."""
    used = run.manifest.settings
    pre, post = _stored_chain(used.get("pre_chain")), _stored_chain(used.get("post_chain"))
    if used.get("mode") == "curve" and "match_eq" in used and _is_linear(pre) and _is_linear(post):
        eq = curve_match.MatchEQ(**used["match_eq"])
        gain = 10 ** (run.manifest.stats["gain_db"] / 20)

        def exact(audio: np.ndarray) -> np.ndarray:
            if pre is not None:
                audio = chain.run(pre, audio, sample_rate)
            audio = curve_match.apply(audio, sample_rate, eq)
            if post is not None:
                audio = chain.run(post, audio, sample_rate)
            return audio * gain

        return exact, "exact"
    return fitted_process(mix, master, sample_rate), "fitted"


def _stereo(audio: np.ndarray) -> np.ndarray:
    return audio if audio.shape[1] == 2 else np.repeat(audio[:, :1], 2, axis=1)


def _pick_set(run: RunResult, name: str | None) -> StemSet:
    sets = workflow.sets_for_run(run)
    if name is not None:
        sets = [s for s in sets if s.name == name]
    usable = [s for s in sets if s.by_role("vocals") and len(s.by_role("vocals")) < len(s.stems)]
    if not usable:
        raise VersionError(
            "alternate versions need stems with a vocal and at least one other part: import the "
            "track's stems from Logic, or separate it first"
        )
    real = [s for s in usable if not s.separated]
    return (real or usable)[0]


def make(
    run: RunResult, paths: Paths, settings: Settings, stem_set_name: str | None = None
) -> Versions:
    """Write instrumental_24.wav and acapella_24.wav for a master run."""
    if "master_24" not in run.manifest.outputs:
        raise VersionError(
            "alternate versions are made from a master; this run only has an analysis"
        )
    stem_set = _pick_set(run, stem_set_name)
    mix, info = audio_io.load(run.mix_path)
    master, _ = audio_io.load(run.output("master_24"))
    mix = _stereo(mix)
    if stem_set.frames != len(mix) or stem_set.sample_rate != info.sample_rate:
        raise VersionError(
            f"the stems ({stem_set.frames} frames at {stem_set.sample_rate} Hz) do not line up "
            f"with this run's mix ({len(mix)} frames at {info.sample_rate} Hz)"
        )
    vocal_names = stem_set.by_role("vocals")
    acapella = np.sum([stem_set.load(n) for n in vocal_names], axis=0)
    instrumental = np.sum(
        [stem_set.load(n) for n in stem_set.stems if n not in vocal_names], axis=0
    )

    process, method = locked_process(run, mix, master, info.sample_rate)
    parts = {"instrumental": process(instrumental), "acapella": process(acapella)}
    sum_error = error_db(parts["instrumental"] + parts["acapella"], master)
    stems_error = error_db(instrumental + acapella, mix)

    ceiling = run.manifest.settings.get("ceiling_dbtp", settings.targets.true_peak_dbtp)
    out_dir = versions_dir(run)
    written = []
    for name, audio in parts.items():
        limited = limiter.true_peak_limit(audio, info.sample_rate, ceiling)
        file = audio_io.save_wav(out_dir / f"{name}_24.wav", limited.audio, info.sample_rate, 24)
        measured = loudness.measure(limited.audio, info.sample_rate)
        written.append(
            Version(
                name=name,
                file=file.name,
                lufs_i=round(measured.lufs_i, 2),
                true_peak_dbtp=round(measured.true_peak_dbtp, 2),
                limiter_reduction_db=round(limited.max_reduction_db, 2),
            )
        )
    flags = _flags(stem_set, method, sum_error, stems_error)
    result = Versions(
        run_id=run.manifest.id,
        stem_set=stem_set.name,
        separated=stem_set.separated,
        method=method,
        sum_error_db=round(sum_error, 1),
        stems_vs_mix_db=round(stems_error, 1),
        versions=written,
        flags=flags,
    )
    projects.write_json(out_dir / SUMMARY_FILE, asdict(result))
    return result


def _flags(stem_set: StemSet, method: str, sum_error: float, stems_error: float) -> list[Flag]:
    flags = []
    if stem_set.separated:
        message = (
            "Made from stems a model separated, not from Logic. Expect bleed and artifacts: "
            "fine for a reference or a rough, not for release."
        )
        flags.append(Flag("separated_stems", "warn", message))
    if stems_error > GOOD_SUM_DB:
        message = (
            f"The stems only add up to within {stems_error:.0f} dB of the mix this master was "
            "made from. Something on the mix bus, or a part, is missing from the stems."
        )
        flags.append(Flag("stems_differ_from_mix", "warn", message, round(stems_error, 1)))
    elif sum_error > POOR_SUM_DB:
        message = (
            f"Instrumental plus a cappella is only within {sum_error:.0f} dB of the master. The "
            "master's processing is too far from one fixed filter to be reproduced on the parts."
        )
        flags.append(Flag("versions_do_not_sum", "problem", message, round(sum_error, 1)))
    elif sum_error > GOOD_SUM_DB:
        message = (
            f"Instrumental plus a cappella is within {sum_error:.0f} dB of the master: close, "
            "with small differences where the processing reacted to the full mix."
        )
        flags.append(Flag("versions_sum_loosely", "info", message, round(sum_error, 1)))
    if method == "fitted":
        message = (
            "This run's processing adapts to the audio, so it was fitted as one fixed filter. "
            "A curve-match master is reproduced exactly instead."
        )
        flags.append(Flag("fitted_processing", "info", message))
    return flags


def load(run: RunResult) -> Versions | None:
    summary = versions_dir(run) / SUMMARY_FILE
    if not summary.exists():
        return None
    found = Versions.from_dict(json.loads(summary.read_text(encoding="utf-8")))
    return found if all((versions_dir(run) / v.file).exists() for v in found.versions) else None
