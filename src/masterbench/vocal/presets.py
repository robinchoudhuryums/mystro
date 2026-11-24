"""Tuning presets: TOML files holding everything about a correction except the key."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import tomli_w

from masterbench.config import Paths
from masterbench.vocal.tuning import TuningError, TuningSettings

DEFAULT = "natural"
FIELDS = (
    "retune_ms",
    "humanize",
    "amount",
    "preserve_formants",
    "formant_shift",
    "note_hold",
    "engine",
)
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class TuningPreset:
    name: str
    description: str
    values: dict[str, Any]

    def settings(
        self,
        key: str,
        scale: str | None = None,
        custom_notes: tuple[str, ...] = (),
        **overrides: Any,
    ) -> TuningSettings:
        """This preset in a key. Overrides that are None leave the preset's value."""
        chosen = {name: value for name, value in overrides.items() if value is not None}
        values = {**self.values, **chosen}
        return TuningSettings(key=key, scale=scale, custom_notes=custom_notes, **values)


def from_toml(text: str, name: str) -> TuningPreset:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise TuningError(f"preset '{name}' is not valid TOML: {exc}") from exc
    description = str(data.pop("description", ""))
    unknown = sorted(set(data) - set(FIELDS))
    if unknown:
        raise TuningError(f"preset '{name}' has unknown settings: {', '.join(unknown)}")
    preset = TuningPreset(name, description, data)
    preset.settings("C major")
    return preset


def from_settings(name: str, settings: TuningSettings, description: str = "") -> TuningPreset:
    data = settings.to_dict()
    return TuningPreset(name, description, {field: data[field] for field in FIELDS})


def to_toml(preset: TuningPreset) -> str:
    return tomli_w.dumps({"description": preset.description, **preset.values})


def _check_name(name: str) -> None:
    if not _NAME.match(name):
        raise TuningError(f"preset names use lowercase letters, digits, '-' and '_': {name!r}")


def _packaged() -> dict[str, str]:
    folder = resources.files("masterbench.vocal").joinpath("presets")
    return {
        entry.name.removesuffix(".toml"): entry.read_text(encoding="utf-8")
        for entry in folder.iterdir()
        if entry.name.endswith(".toml")
    }


def names(paths: Paths) -> list[str]:
    """Every preset name, the packaged three first; your own shadow them by name."""
    folder = paths.tuning_presets
    own = {p.stem for p in folder.glob("*.toml")} if folder.is_dir() else set()
    packaged = ["natural", "tight", "hard-tune"]
    return packaged + sorted(own - set(packaged))


def get(paths: Paths, name: str) -> TuningPreset:
    _check_name(name)
    own = paths.tuning_presets / f"{name}.toml"
    if own.is_file():
        return from_toml(own.read_text(encoding="utf-8"), name)
    packaged = _packaged()
    if name not in packaged:
        raise TuningError(f"no tuning preset named '{name}'; have {', '.join(names(paths))}")
    return from_toml(packaged[name], name)


def save(paths: Paths, preset: TuningPreset) -> Path:
    _check_name(preset.name)
    preset.settings("C major")
    paths.tuning_presets.mkdir(parents=True, exist_ok=True)
    dest = paths.tuning_presets / f"{preset.name}.toml"
    dest.write_text(to_toml(preset), encoding="utf-8")
    return dest
