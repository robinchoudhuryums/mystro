"""Paths, defaults and targets. User overrides live in <data root>/settings.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

DATA_ROOT_ENV = "MASTERBENCH_HOME"
DEFAULT_GENRES = ("hip-hop", "house", "electronic")

# Named bands for frequency-balance deltas (Hz, low inclusive, high exclusive).
BANDS: dict[str, tuple[float, float]] = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 250.0),
    "low_mids": (250.0, 500.0),
    "mids": (500.0, 2000.0),
    "upper_mids": (2000.0, 6000.0),
    "presence": (6000.0, 10000.0),
    "air": (10000.0, 20000.0),
}


@dataclass(frozen=True)
class Targets:
    lufs_i: float = -14.0
    true_peak_dbtp: float = -1.0


@dataclass(frozen=True)
class Thresholds:
    """Pre-flight warnings on input mixes (spec 5.1)."""

    already_mastered_lufs: float = -12.0
    near_full_scale_dbtp: float = -0.3
    low_crest_db: float = 8.0
    clip_run_samples: int = 3
    dc_offset: float = 0.001
    edge_silence_s: float = 2.0


@dataclass(frozen=True)
class Settings:
    targets: Targets = field(default_factory=Targets)
    thresholds: Thresholds = field(default_factory=Thresholds)
    notes_model: str = "claude-sonnet-5-5"
    notes_monthly_cap_usd: float = 5.0
    watch_auto_master: bool = False
    mcp_allow_writes: bool = False
    break_interval_min: int = 50


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def settings(self) -> Path:
        return self.root / "settings.toml"

    @property
    def db(self) -> Path:
        return self.root / "masterbench.db"

    @property
    def inbox(self) -> Path:
        return self.root / "Inbox"

    @property
    def references(self) -> Path:
        return self.root / "references"

    @property
    def target_curves(self) -> Path:
        return self.root / "target_curves"

    @property
    def chain_presets(self) -> Path:
        return self.root / "presets" / "chains"

    @property
    def projects(self) -> Path:
        return self.root / "projects"

    def ensure(self) -> None:
        for directory in (
            self.inbox,
            self.references,
            self.target_curves,
            self.chain_presets,
            self.projects,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for genre in DEFAULT_GENRES:
            (self.inbox / genre).mkdir(exist_ok=True)
            (self.references / genre).mkdir(exist_ok=True)


def data_root() -> Path:
    override = os.environ.get(DATA_ROOT_ENV)
    return Path(override).expanduser() if override else Path.home() / "MasterBench"


def get_paths() -> Paths:
    return Paths(data_root())


def _merge(instance: Any, overrides: dict[str, Any], where: str) -> Any:
    known = {f.name: f for f in fields(instance)}
    changes: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in known:
            raise ValueError(f"unknown setting {where}{key}")
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"setting {where}{key} must be a table")
            changes[key] = _merge(current, value, f"{where}{key}.")
        elif isinstance(value, bool) != isinstance(current, bool) or not isinstance(
            value, (int, float) if isinstance(current, float) else type(current)
        ):
            raise ValueError(f"setting {where}{key} must be {type(current).__name__}")
        else:
            changes[key] = type(current)(value)
    return replace(instance, **changes)


def load_settings(paths: Paths | None = None) -> Settings:
    paths = paths or get_paths()
    if not paths.settings.exists():
        return Settings()
    with paths.settings.open("rb") as handle:
        return _merge(Settings(), tomllib.load(handle), "")
