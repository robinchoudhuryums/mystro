"""Processing chains: built-in modules and hosted VST3/AU plugins, described by presets."""

from __future__ import annotations

import base64
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pedalboard as pb

from masterbench.master import limiter

PLUGIN_SUFFIXES = {".vst3": "VST3", ".component": "AU"}
# Pedalboard re-exports these without declaring them, which the type checker flags.
_host = pb.load_plugin  # pyright: ignore[reportPrivateImportUsage]
_Board = pb.Pedalboard  # pyright: ignore[reportPrivateImportUsage]


class ChainError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    default: float
    low: float
    high: float


@dataclass(frozen=True)
class Module:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainPreset:
    name: str
    description: str = ""
    modules: tuple[Module, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        modules = [{"type": m.type, **m.params} for m in self.modules]
        return {"name": self.name, "description": self.description, "module": modules}


@dataclass(frozen=True)
class PluginInfo:
    path: Path
    format: str
    names: tuple[str, ...]


BUILTIN: dict[str, dict[str, Param]] = {
    "highpass": {"cutoff_hz": Param(30.0, 10.0, 1000.0)},
    "lowpass": {"cutoff_hz": Param(12000.0, 200.0, 20000.0)},
    "low_shelf": {
        "freq_hz": Param(100.0, 20.0, 1000.0),
        "gain_db": Param(0.0, -12.0, 12.0),
        "q": Param(0.707, 0.1, 4.0),
    },
    "high_shelf": {
        "freq_hz": Param(8000.0, 1000.0, 20000.0),
        "gain_db": Param(0.0, -12.0, 12.0),
        "q": Param(0.707, 0.1, 4.0),
    },
    "peak": {
        "freq_hz": Param(1000.0, 20.0, 20000.0),
        "gain_db": Param(0.0, -12.0, 12.0),
        "q": Param(1.0, 0.1, 10.0),
    },
    "compressor": {
        "threshold_db": Param(-18.0, -60.0, 0.0),
        "ratio": Param(2.0, 1.0, 20.0),
        "attack_ms": Param(10.0, 0.1, 200.0),
        "release_ms": Param(120.0, 5.0, 2000.0),
        "makeup_db": Param(0.0, -12.0, 12.0),
    },
    "gain": {"gain_db": Param(0.0, -24.0, 24.0)},
    "saturation": {"drive_db": Param(6.0, 0.0, 24.0), "mix": Param(1.0, 0.0, 1.0)},
    "limiter": {"ceiling_dbtp": Param(-1.0, -12.0, 0.0)},
}
PLUGIN_KEYS = {"path", "plugin_name", "parameters", "state"}


def validate(preset: ChainPreset) -> None:
    """Raise ChainError naming the first module that is wrong."""
    for number, module in enumerate(preset.modules, start=1):
        where = f"{preset.name}: module {number} ({module.type})"
        if module.type == "plugin":
            unknown = set(module.params) - PLUGIN_KEYS
            if unknown or not isinstance(module.params.get("path"), str):
                raise ChainError(f"{where}: needs 'path'; unknown keys {sorted(unknown)}")
            continue
        if module.type not in BUILTIN:
            raise ChainError(f"{where}: unknown type; choose from {', '.join(BUILTIN)}, plugin")
        spec = BUILTIN[module.type]
        for key, value in module.params.items():
            if key not in spec:
                raise ChainError(f"{where}: unknown setting '{key}'; it has {', '.join(spec)}")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ChainError(f"{where}: {key} must be a number")
            if not spec[key].low <= value <= spec[key].high:
                raise ChainError(f"{where}: {key} must be {spec[key].low:g} to {spec[key].high:g}")


def _settings(module: Module) -> dict[str, float]:
    return {
        key: float(module.params.get(key, p.default)) for key, p in BUILTIN[module.type].items()
    }


def _native(module: Module) -> list[pb.Plugin]:
    """Pedalboard processors for a built-in module (none for the numpy-only ones)."""
    s = _settings(module)
    if module.type == "highpass":
        return [pb.HighpassFilter(cutoff_frequency_hz=s["cutoff_hz"])]
    if module.type == "lowpass":
        return [pb.LowpassFilter(cutoff_frequency_hz=s["cutoff_hz"])]
    if module.type == "low_shelf":
        return [pb.LowShelfFilter(cutoff_frequency_hz=s["freq_hz"], gain_db=s["gain_db"], q=s["q"])]
    if module.type == "high_shelf":
        return [
            pb.HighShelfFilter(cutoff_frequency_hz=s["freq_hz"], gain_db=s["gain_db"], q=s["q"])
        ]
    if module.type == "peak":
        return [pb.PeakFilter(cutoff_frequency_hz=s["freq_hz"], gain_db=s["gain_db"], q=s["q"])]
    if module.type == "gain":
        return [pb.Gain(gain_db=s["gain_db"])]
    if module.type == "compressor":
        compressor = pb.Compressor(
            threshold_db=s["threshold_db"],
            ratio=s["ratio"],
            attack_ms=s["attack_ms"],
            release_ms=s["release_ms"],
        )
        return [compressor, pb.Gain(gain_db=s["makeup_db"])]
    return []


def _saturate(audio: np.ndarray, drive_db: float, mix: float) -> np.ndarray:
    """Soft clipping that leaves quiet passages at unity gain, blended with the dry signal."""
    drive = 10 ** (drive_db / 20)
    return (1 - mix) * audio + mix * np.tanh(audio * drive) / drive


def load_plugin(module: Module) -> Any:
    """Instantiate a hosted plugin and restore its saved state and parameters."""
    path = Path(module.params["path"]).expanduser()
    if not path.exists():
        raise ChainError(f"plugin not found: {path}")
    try:
        name = module.params.get("plugin_name")
        plugin: Any = _host(str(path), plugin_name=name) if name else _host(str(path))
        if module.params.get("state"):
            plugin.raw_state = base64.b64decode(module.params["state"])
        for key, value in module.params.get("parameters", {}).items():
            if key not in plugin.parameters:
                raise ChainError(f"{path.name} has no parameter '{key}'")
            setattr(plugin, key, value)
    except ChainError:
        raise
    except Exception as exc:  # plugin hosts fail in many ways
        raise ChainError(f"could not load {path.name}: {exc}") from exc
    return plugin


def _run_plugin(plugin: Any, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Process (channels, frames) float32. Mono-only plugins run once per channel."""
    try:
        return plugin(audio, sample_rate)
    except ValueError as exc:
        if "channel" not in str(exc) or audio.shape[0] == 1:
            raise ChainError(f"plugin failed: {exc}") from exc
    return np.concatenate([plugin(audio[i : i + 1], sample_rate) for i in range(audio.shape[0])])


def _stages(preset: ChainPreset, sample_rate: int) -> list[Callable[[np.ndarray], np.ndarray]]:
    """Processing steps over (channels, frames) float32; adjacent built-ins share a board."""
    stages: list[Callable[[np.ndarray], np.ndarray]] = []
    pending: list[pb.Plugin] = []

    def flush() -> None:
        if pending:
            board = _Board(list(pending))
            stages.append(lambda audio: board(audio, sample_rate))
            pending.clear()

    for module in preset.modules:
        native = _native(module) if module.type in BUILTIN else []
        if native:
            pending.extend(native)
            continue
        flush()
        if module.type == "saturation":
            s = _settings(module)
            stages.append(lambda audio, s=s: _saturate(audio, s["drive_db"], s["mix"]))
        elif module.type == "limiter":
            ceiling = _settings(module)["ceiling_dbtp"]
            stages.append(
                lambda audio, ceiling=ceiling: (
                    limiter.true_peak_limit(
                        audio.T.astype(np.float64), sample_rate, ceiling
                    ).audio.T
                )
            )
        else:
            plugin = load_plugin(module)
            stages.append(lambda audio, plugin=plugin: _run_plugin(plugin, audio, sample_rate))
    flush()
    return stages


def run(preset: ChainPreset, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply a chain to float audio shaped (frames, channels). Length and shape are kept."""
    validate(preset)
    work = np.ascontiguousarray(audio.T, dtype=np.float32)
    for stage in _stages(preset, sample_rate):
        work = np.asarray(stage(work), dtype=np.float32)
        if work.shape != (audio.shape[1], audio.shape[0]):
            raise ChainError(f"{preset.name}: a module changed the audio's length or channels")
        if not np.all(np.isfinite(work)):
            raise ChainError(f"{preset.name}: a module produced invalid audio (NaN or infinity)")
    return work.T.astype(np.float64)


def plugin_dirs() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        bases = [Path("/Library/Audio/Plug-Ins"), home / "Library/Audio/Plug-Ins"]
        return [base / kind for base in bases for kind in ("VST3", "Components")]
    return [home / ".vst3", Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3")]


def scan_plugins(dirs: list[Path] | None = None) -> list[PluginInfo]:
    """Plugins in the standard folders. Names inside a shell (Waves) are listed when readable."""
    found = []
    for directory in dirs or plugin_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix not in PLUGIN_SUFFIXES:
                continue
            names: tuple[str, ...] = ()
            if path.suffix == ".vst3":
                try:
                    names = tuple(pb.VST3Plugin.get_plugin_names_for_file(str(path)))
                except Exception:  # unreadable bundles are still listed
                    names = ()
            found.append(PluginInfo(path, PLUGIN_SUFFIXES[path.suffix], names))
    return found


def capture_plugin(path: Path, plugin_name: str | None = None) -> Module:
    """A preset module holding a plugin's current parameters and opaque state."""
    params: dict[str, Any] = {"path": str(path)}
    if plugin_name:
        params["plugin_name"] = plugin_name
    plugin = load_plugin(Module("plugin", params))
    values = {}
    for key in plugin.parameters:
        value = getattr(plugin, key)
        values[key] = value if isinstance(value, bool | str) else float(value)
    params["parameters"] = values
    params["state"] = base64.b64encode(plugin.raw_state).decode("ascii")
    return Module("plugin", params)


def plugin_parameters(path: Path, plugin_name: str | None = None) -> list[dict[str, Any]]:
    """Name, value, range and unit of each parameter, for building controls."""
    params: dict[str, Any] = {
        "path": str(path),
        **({"plugin_name": plugin_name} if plugin_name else {}),
    }
    plugin = load_plugin(Module("plugin", params))
    return [
        {
            "name": key,
            "value": getattr(plugin, key),
            "min": parameter.min_value,
            "max": parameter.max_value,
            "units": parameter.units,
        }
        for key, parameter in plugin.parameters.items()
    ]
