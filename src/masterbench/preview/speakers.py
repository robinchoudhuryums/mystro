"""Speaker simulation: rough stand-ins for a phone, earbuds, a car, a laptop, a club and mono.

These are approximations built from a few filters, good for spotting what disappears on small
speakers. They do not replace checking on the real thing.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from importlib import resources
from typing import Any

import numpy as np

from masterbench import audio_io, projects
from masterbench.analysis import loudness, spectrum
from masterbench.analysis.compare import BAND_LABELS
from masterbench.config import BANDS, Paths
from masterbench.master import chain
from masterbench.master.chain import ChainError, ChainPreset, Module
from masterbench.master.pipeline import RunResult

PREVIEW_DIR = "previews"
SUMMARY_FILE = "speakers.json"
NOTICE = "Speaker simulations are approximations, not a replacement for the real check."
_GONE_DB = -18.0


@dataclass(frozen=True)
class Profile:
    name: str
    label: str
    description: str
    mono: bool
    width: float
    modules: tuple[Module, ...]


@dataclass(frozen=True)
class SpeakerPreview:
    name: str
    label: str
    description: str
    file: str
    lufs_i: float
    band_change_db: dict[str, float]
    message: str


def user_dir(paths: Paths):
    return paths.root / "presets" / "speakers"


def _parse(name: str, text: str) -> Profile:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ChainError(f"speaker profile {name}: not valid TOML ({exc})") from exc
    unknown = set(data) - {"label", "description", "mono", "width", "module"}
    if unknown:
        raise ChainError(f"speaker profile {name}: unknown keys {sorted(unknown)}")
    modules = tuple(
        Module(entry["type"], {k: v for k, v in entry.items() if k != "type"})
        for entry in data.get("module", [])
    )
    width = float(data.get("width", 1.0))
    if not 0.0 <= width <= 2.0:
        raise ChainError(f"speaker profile {name}: width must be 0 to 2")
    profile = Profile(
        name=name,
        label=str(data.get("label", name)),
        description=str(data.get("description", "")),
        mono=bool(data.get("mono", False)),
        width=width,
        modules=modules,
    )
    chain.validate(ChainPreset(name, "", modules))
    return profile


def profiles(paths: Paths) -> dict[str, Profile]:
    """Every profile by name; files in <data root>/presets/speakers override packaged ones."""
    texts = {
        entry.name.removesuffix(".toml"): entry.read_text(encoding="utf-8")
        for entry in resources.files("masterbench.preview").joinpath("speakers").iterdir()
        if entry.name.endswith(".toml")
    }
    if user_dir(paths).is_dir():
        texts |= {f.stem: f.read_text(encoding="utf-8") for f in user_dir(paths).glob("*.toml")}
    return {name: _parse(name, text) for name, text in sorted(texts.items())}


def simulate(audio: np.ndarray, sample_rate: int, profile: Profile) -> np.ndarray:
    """Audio as the profile's speaker would roughly play it. Shape and length are kept."""
    out = chain.run(ChainPreset(profile.name, "", profile.modules), audio, sample_rate)
    if out.shape[1] == 2 and (profile.mono or profile.width != 1.0):
        mid, side = out.mean(axis=1), (out[:, 0] - out[:, 1]) / 2
        side = side * (0.0 if profile.mono else profile.width)
        out = np.column_stack([mid + side, mid - side])
    return out


def _absolute_bands(audio: np.ndarray, sample_rate: int) -> tuple[dict[str, float], float]:
    spectra = spectrum.cross_spectra(audio, sample_rate)
    return spectrum.band_levels(spectra, 0.0), float(
        spectra.freqs[np.argmax(np.where(spectra.band(100.0, 600.0), spectra.power, 0.0))]
    )


def _message(profile: Profile, change: dict[str, float], carrier_hz: float) -> str:
    gone = [BAND_LABELS[b].lower() for b in ("sub", "bass") if change[b] <= _GONE_DB]
    if gone:
        return (
            f"On this speaker the {' and '.join(gone)} {'are' if len(gone) > 1 else 'is'} gone "
            f"({change['sub']:.0f} dB in the sub). "
            f"The bass line has to be carried by what sits around {carrier_hz:.0f} Hz; if the 808 "
            "disappears here, saturation adds harmonics that survive."
        )
    lifted = [BAND_LABELS[b].lower() for b in BANDS if change[b] >= 2.5]
    if lifted:
        return f"Lifts the {', '.join(lifted)}: listen for boom or harshness that this exposes."
    if profile.mono:
        return "Mono sum: anything that gets quieter here was relying on stereo width."
    return profile.description


def render(run: RunResult, paths: Paths, force: bool = False) -> list[SpeakerPreview]:
    """Render every profile for a run's master (or its mix), cached under previews/."""
    cached = load(run)
    if cached is not None and not force and {p.name for p in cached} == set(profiles(paths)):
        return cached
    source = run.output("master_24") if "master_24" in run.manifest.outputs else run.mix_path
    audio, info = audio_io.load(source)
    before, _ = _absolute_bands(audio, info.sample_rate)
    out_dir = run.run_dir / PREVIEW_DIR
    previews = []
    for profile in profiles(paths).values():
        simulated = simulate(audio, info.sample_rate, profile)
        file = audio_io.save_wav(
            out_dir / f"speaker_{profile.name}.wav", simulated, info.sample_rate, 32
        )
        after, carrier = _absolute_bands(simulated, info.sample_rate)
        change = {band: round(after[band] - before[band], 1) for band in BANDS}
        previews.append(
            SpeakerPreview(
                name=profile.name,
                label=profile.label,
                description=profile.description,
                file=f"{PREVIEW_DIR}/{file.name}",
                lufs_i=round(loudness.integrated(simulated, info.sample_rate), 2),
                band_change_db=change,
                message=_message(profile, change, carrier),
            )
        )
    projects.write_json(out_dir / SUMMARY_FILE, [asdict(p) for p in previews])
    return previews


def load(run: RunResult) -> list[SpeakerPreview] | None:
    summary = run.run_dir / PREVIEW_DIR / SUMMARY_FILE
    if not summary.exists():
        return None
    raw: list[dict[str, Any]] = json.loads(summary.read_text(encoding="utf-8"))
    previews = [SpeakerPreview(**item) for item in raw]
    return previews if all((run.run_dir / p.file).exists() for p in previews) else None
