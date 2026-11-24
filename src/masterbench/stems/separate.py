"""Stem sets: separated with a model, or real stems exported from Logic.

A stem set is a folder of equal-length WAVs plus stems.json. Whatever produced it, the stems
keep the source's sample rate, length and level, so they can be measured against the mix and
summed back together.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from masterbench import audio_io, config, projects
from masterbench.analysis import loudness

MANIFEST = "stems.json"
ROLES = ("vocals", "drums", "bass", "other", "instrumental")
# The separator scales any stem that peaks above its threshold, which would change the
# balance between stems. Feeding it a quieter copy keeps every stem below that point.
_SEPARATOR_PEAK = 0.5
_MODEL_RATE = 44100

_ROLE_WORDS = {
    "vocals": ("vocal", "vox", "lead", "adlib", "ad-lib", "bgv", "backing", "hook", "verse", "rap"),
    "drums": ("drum", "kick", "snare", "hat", "perc", "clap", "cymbal", "tom"),
    "bass": ("bass", "808", "sub"),
    "instrumental": ("instrumental", "beat", "inst"),
}


class StemError(RuntimeError):
    pass


@dataclass(frozen=True)
class StemModel:
    key: str
    filename: str
    stems: tuple[str, ...]
    description: str


MODELS = {
    m.key: m
    for m in (
        StemModel(
            "mdx-2",
            "UVR-MDX-NET-Inst_HQ_3.onnx",
            ("vocals", "instrumental"),
            "Vocals and instrumental. Small and fast.",
        ),
        StemModel(
            "roformer-2",
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            ("vocals", "instrumental"),
            "Vocals and instrumental. Cleaner vocals, much slower.",
        ),
        StemModel(
            "demucs-4",
            "htdemucs_ft.yaml",
            ("vocals", "drums", "bass", "other"),
            "Vocals, drums, bass and other. Best four-stem quality, slow.",
        ),
        StemModel(
            "demucs-4-fast",
            "htdemucs.yaml",
            ("vocals", "drums", "bass", "other"),
            "Vocals, drums, bass and other. About four times faster than demucs-4.",
        ),
    )
}
DEFAULT_MODEL = "mdx-2"

# (input file, model file, output folder) -> files written
Runner = Callable[[Path, str, Path], list[Path]]


@dataclass(frozen=True)
class StemSet:
    name: str
    dir: Path
    source: str
    separated: bool
    input_sha256: str
    sample_rate: int
    frames: int
    stems: dict[str, str]
    roles: dict[str, str]
    seconds: float | None = None
    created_at: str = ""
    lufs: dict[str, float] | None = None

    def path(self, stem: str) -> Path:
        return self.dir / self.stems[stem]

    def load(self, stem: str) -> np.ndarray:
        return audio_io.load(self.path(stem))[0]

    def by_role(self, role: str) -> list[str]:
        return [name for name, assigned in self.roles.items() if assigned == role]

    def save(self) -> None:
        data = {key: value for key, value in asdict(self).items() if key != "dir"}
        projects.write_json(self.dir / MANIFEST, data)


def load_set(folder: Path) -> StemSet | None:
    manifest = folder / MANIFEST
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    stem_set = StemSet(dir=folder, **data)
    return stem_set if all(stem_set.path(s).is_file() for s in stem_set.stems) else None


def list_sets(project_dir: Path) -> list[StemSet]:
    """Complete stem sets of a project, real stems first, then newest."""
    folders = (project_dir / "stems").glob("*/") if (project_dir / "stems").is_dir() else []
    found = [s for s in map(load_set, folders) if s is not None]
    return sorted(found, key=lambda s: (s.separated, s.created_at), reverse=False)


def models_dir() -> Path:
    """Model weights are large and local to a machine; they never go in the data root."""
    return config.state_dir() / "models"


def audio_separator_runner(source: Path, model_file: str, out_dir: Path) -> list[Path]:
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise StemError(f"audio-separator is not installed: {exc}") from exc
    separator = Separator(
        log_level=30,
        model_file_dir=str(models_dir()),
        output_dir=str(out_dir),
        output_format="WAV",
        normalization_threshold=1.0,
        use_soundfile=True,
    )
    try:
        separator.load_model(model_filename=model_file)
    except Exception as exc:
        raise StemError(f"could not load or download the model {model_file}: {exc}") from exc
    return [out_dir / Path(name).name for name in separator.separate(str(source))]


def _stem_role(filename: str, expected: tuple[str, ...]) -> str | None:
    """Which of the model's stems a separator output file is, from its '(Stem)' tag."""
    tag = re.search(r"\(([A-Za-z ]+)\)", filename)
    name = tag.group(1).strip().lower() if tag else ""
    return name if name in expected else None


def _fit(audio: np.ndarray, rate: int, target_rate: int, frames: int) -> np.ndarray:
    """Bring separator output back to the source's sample rate, length and channel count."""
    if rate != target_rate:
        audio = signal.resample_poly(audio, target_rate, rate, axis=0)
    fitted = np.zeros((frames, 2))
    count = min(frames, len(audio))
    fitted[:count] = audio[:count, :2] if audio.shape[1] >= 2 else audio[:count, :1]
    return fitted


def separate(
    source: Path,
    cache_root: Path,
    model_key: str = DEFAULT_MODEL,
    runner: Runner = audio_separator_runner,
    force: bool = False,
) -> StemSet:
    """Separate a file into stems, cached under cache_root by content hash and model."""
    if model_key not in MODELS:
        raise StemError(f"unknown stem model '{model_key}'; choose from {', '.join(MODELS)}")
    model = MODELS[model_key]
    sha256 = audio_io.file_sha256(source)
    folder = cache_root / f"{model_key}-{sha256[:8]}"
    cached = load_set(folder)
    if cached is not None and not force:
        return cached

    audio, info = audio_io.load(source)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak == 0.0:
        raise StemError(f"{source.name} is silent")
    gain = _SEPARATOR_PEAK / peak
    stereo = audio if audio.shape[1] == 2 else np.repeat(audio[:, :1], 2, axis=1)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="masterbench-stems-") as tmp:
        staged = audio_io.save_wav(Path(tmp) / "input.wav", stereo * gain, info.sample_rate, 32)
        outputs = runner(staged, model.filename, Path(tmp))
        stems: dict[str, np.ndarray] = {}
        for output in outputs:
            role = _stem_role(output.name, model.stems)
            if role is None:
                continue
            data, out_info = audio_io.load(output)
            stems[role] = _fit(data, out_info.sample_rate, info.sample_rate, info.frames) / gain
    seconds = time.perf_counter() - started
    missing = [name for name in model.stems if name not in stems]
    if missing:
        raise StemError(f"{model_key} did not return these stems: {', '.join(missing)}")

    if folder.exists():
        shutil.rmtree(folder)
    for name, data in stems.items():
        audio_io.save_wav(folder / f"{name}.wav", data, info.sample_rate, 32)
    stem_set = StemSet(
        name=folder.name,
        dir=folder,
        source=model_key,
        separated=True,
        input_sha256=sha256,
        sample_rate=info.sample_rate,
        frames=info.frames,
        stems={name: f"{name}.wav" for name in model.stems},
        roles={name: name for name in model.stems},
        seconds=round(seconds, 1),
        created_at=projects.now_iso(),
        lufs={n: round(loudness.integrated(d, info.sample_rate), 2) for n, d in stems.items()},
    )
    stem_set.save()
    return stem_set


def guess_role(filename: str) -> str:
    """Role of a Logic stem from its file name; anything unrecognised is 'other'."""
    words = Path(filename).stem.lower()
    for role, hints in _ROLE_WORDS.items():
        if any(hint in words for hint in hints):
            return role
    return "other"


def import_stems(folder: Path, cache_root: Path, roles: dict[str, str] | None = None) -> StemSet:
    """Copy a folder of stems exported from Logic into a stem set.

    All files must share a sample rate. Shorter files are padded with silence to the longest,
    and mono files become stereo, so every stem lines up with the others sample for sample.
    """
    files = sorted(f for f in folder.iterdir() if f.is_file() and audio_io.is_audio(f))
    if len(files) < 2:
        raise StemError(f"need at least two audio files in {folder}, found {len(files)}")
    loaded = {f: audio_io.load(f) for f in files}
    rates = {info.sample_rate for _, info in loaded.values()}
    if len(rates) != 1:
        raise StemError(f"stems have different sample rates: {sorted(rates)}")
    lossy = [f.name for f, (_, info) in loaded.items() if info.lossy]
    if lossy:
        raise StemError(f"stems must be WAV, AIFF or FLAC; got {', '.join(lossy)}")
    rate, frames = rates.pop(), max(info.frames for _, info in loaded.values())
    sha256 = audio_io.file_sha256(files[0])[:4] + audio_io.file_sha256(files[-1])[:4]
    target = cache_root / f"logic-{sha256}"
    if target.exists():
        shutil.rmtree(target)

    stems, assigned, levels = {}, {}, {}
    for file, (audio, _) in loaded.items():
        name = projects.slugify(file.stem)
        if name in stems:
            raise StemError(f"two stems reduce to the same name '{name}': rename one")
        fitted = _fit(audio, rate, rate, frames)
        audio_io.save_wav(target / f"{name}.wav", fitted, rate, 32)
        levels[name] = round(loudness.integrated(fitted, rate), 2)
        stems[name] = f"{name}.wav"
        assigned[name] = (roles or {}).get(file.name) or guess_role(file.name)
    unknown = sorted(set(assigned.values()) - set(ROLES))
    if unknown:
        raise StemError(f"unknown roles {unknown}; use {', '.join(ROLES)}")
    stem_set = StemSet(
        name=target.name,
        dir=target,
        source="logic",
        separated=False,
        input_sha256=sha256,
        sample_rate=rate,
        frames=frames,
        stems=stems,
        roles=assigned,
        created_at=projects.now_iso(),
        lufs=levels,
    )
    stem_set.save()
    return stem_set


def mixdown(stem_set: StemSet, gains_db: dict[str, float] | None = None) -> np.ndarray:
    """Sum of the stems, each with an optional gain."""
    total = np.zeros((stem_set.frames, 2))
    for name in stem_set.stems:
        total += stem_set.load(name) * 10 ** ((gains_db or {}).get(name, 0.0) / 20)
    return total
