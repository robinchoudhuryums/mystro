"""Chain presets as TOML: the user's in <data root>/presets/chains, plus a few packaged ones."""

from __future__ import annotations

import re
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

import tomli_w

from masterbench.config import Paths
from masterbench.master import chain
from masterbench.master.chain import ChainError, ChainPreset, Module

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def from_dict(data: dict[str, Any], name: str) -> ChainPreset:
    unknown = set(data) - {"name", "description", "module"}
    if unknown:
        raise ChainError(f"{name}: unknown keys {sorted(unknown)}")
    modules = []
    for number, entry in enumerate(data.get("module", []), start=1):
        if not isinstance(entry, dict) or not isinstance(entry.get("type"), str):
            raise ChainError(f"{name}: module {number} needs a 'type'")
        modules.append(Module(entry["type"], {k: v for k, v in entry.items() if k != "type"}))
    preset = ChainPreset(name, str(data.get("description", "")), tuple(modules))
    chain.validate(preset)
    return preset


def from_toml(text: str, name: str) -> ChainPreset:
    try:
        return from_dict(tomllib.loads(text), name)
    except tomllib.TOMLDecodeError as exc:
        raise ChainError(f"{name}: not valid TOML ({exc})") from exc


def to_toml(preset: ChainPreset) -> str:
    data = preset.to_dict()
    del data["name"]
    return tomli_w.dumps(data)


def _check_name(name: str) -> None:
    if not _NAME.match(name):
        raise ChainError(f"preset names use lowercase letters, digits, '-' and '_': {name!r}")


def _packaged() -> dict[str, str]:
    folder = resources.files("masterbench.master").joinpath("presets")
    return {
        entry.name.removesuffix(".toml"): entry.read_text(encoding="utf-8")
        for entry in folder.iterdir()
        if entry.name.endswith(".toml")
    }


def names(paths: Paths) -> list[str]:
    """Every preset name; the user's presets shadow packaged ones of the same name."""
    own = (
        {p.stem for p in paths.chain_presets.glob("*.toml")}
        if paths.chain_presets.is_dir()
        else set()
    )
    return sorted(own | set(_packaged()))


def text(paths: Paths, name: str) -> str:
    _check_name(name)
    own = paths.chain_presets / f"{name}.toml"
    if own.is_file():
        return own.read_text(encoding="utf-8")
    packaged = _packaged()
    if name not in packaged:
        raise ChainError(
            f"no chain preset named '{name}'; have {', '.join(names(paths)) or 'none'}"
        )
    return packaged[name]


def get(paths: Paths, name: str) -> ChainPreset:
    return from_toml(text(paths, name), name)


def save(paths: Paths, preset: ChainPreset) -> Path:
    _check_name(preset.name)
    chain.validate(preset)
    paths.chain_presets.mkdir(parents=True, exist_ok=True)
    dest = paths.chain_presets / f"{preset.name}.toml"
    dest.write_text(to_toml(preset), encoding="utf-8")
    return dest


def save_text(paths: Paths, name: str, toml_text: str) -> ChainPreset:
    """Validate TOML typed by the user and store it as written, comments included."""
    _check_name(name)
    preset = from_toml(toml_text, name)
    paths.chain_presets.mkdir(parents=True, exist_ok=True)
    (paths.chain_presets / f"{name}.toml").write_text(toml_text, encoding="utf-8")
    return preset
