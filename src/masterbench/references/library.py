"""The genre reference library: files under references/<genre>/ with cached analyses."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from masterbench import audio_io, projects
from masterbench.analysis.report import AnalysisReport, analyze_audio
from masterbench.config import Paths, Settings
from masterbench.references import quality as quality_check
from masterbench.references.quality import Quality

SIDECAR_SUFFIX = ".analysis.json"


class ReferenceError(LookupError):
    pass


# Names that stand for a choice rather than a file: the genre's default, and "the closest".
AUTO = "auto"
RESERVED = ("default", AUTO)


def auto_genre(spec: str) -> str | None:
    """The genre of an automatic spec: '' for 'auto', 'house' for 'house/auto', else None."""
    genre, slash, name = spec.strip().rpartition("/")
    if name.lower() != AUTO or (slash and not genre) or Path(spec).expanduser().is_file():
        return None
    return genre


def default_spec(settings: Settings, genre: str | None) -> str:
    """The reference to use when none was named: the genre's default, or its closest."""
    if settings.default_reference not in RESERVED:
        raise ReferenceError(
            f"default_reference in settings.toml must be 'default' or 'auto', "
            f"not '{settings.default_reference}'"
        )
    return f"{genre or settings.default_genre}/{settings.default_reference}"


@dataclass(frozen=True)
class Reference:
    genre: str
    name: str
    path: Path
    report: AnalysisReport
    notes: str = ""
    added_at: str = ""
    is_default: bool = False
    quality: Quality | None = None

    @property
    def spec(self) -> str:
        return f"{self.genre}/{self.name}" if self.genre else str(self.path)


def normalize(text: str) -> str:
    """Genre and reference names compare without case, spaces or punctuation."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


_norm = normalize


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + SIDECAR_SUFFIX)


def _defaults_file(paths: Paths) -> Path:
    return paths.references / "defaults.json"


def _defaults(paths: Paths) -> dict[str, str]:
    file = _defaults_file(paths)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}


def genres(paths: Paths) -> list[str]:
    if not paths.references.is_dir():
        return []
    return sorted(p.name for p in paths.references.iterdir() if p.is_dir())


def _match_genre(paths: Paths, genre: str) -> str | None:
    return next((g for g in genres(paths) if _norm(g) == _norm(genre)), None)


def list_references(paths: Paths, genre: str | None = None) -> list[Reference]:
    """References on disk, grouped by genre then name. Files without a sidecar are skipped."""
    wanted = [_match_genre(paths, genre)] if genre else genres(paths)
    defaults = _defaults(paths)
    found = []
    for name in filter(None, wanted):
        for file in sorted((paths.references / name).iterdir()):
            sidecar = _sidecar(file)
            if not audio_io.is_audio(file) or not sidecar.exists():
                continue
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            found.append(
                Reference(
                    genre=name,
                    name=file.stem,
                    path=file,
                    report=AnalysisReport.from_dict(data["analysis"]),
                    notes=data.get("notes", ""),
                    added_at=data.get("added_at", ""),
                    is_default=defaults.get(name) == file.name,
                    quality=Quality.from_dict(data["quality"]) if data.get("quality") else None,
                )
            )
    return found


def analyze(path: Path, settings: Settings) -> tuple[AnalysisReport, Quality]:
    """Measure a reference file and judge whether it is fit to compare against."""
    audio, info = audio_io.load(path)
    report = analyze_audio(audio, info, path, "reference", settings)
    assert report.stereo is not None
    quality = quality_check.assess(audio, info, report.loudness, report.stereo, settings.thresholds)
    return report, quality


def refresh(paths: Paths, settings: Settings, genre: str | None = None) -> int:
    """Re-analyze references whose cached analysis predates the current measurements."""
    stale = [
        ref
        for ref in list_references(paths, genre)
        if ref.quality is None or ref.report.sections is None or ref.report.regions is None
    ]
    for ref in stale:
        report, quality = analyze(ref.path, settings)
        data = json.loads(_sidecar(ref.path).read_text(encoding="utf-8"))
        projects.write_json(
            _sidecar(ref.path), {**data, "analysis": report.to_dict(), "quality": asdict(quality)}
        )
    return len(stale)


def add(
    paths: Paths,
    source: Path,
    genre: str,
    settings: Settings,
    notes: str = "",
    make_default: bool = False,
) -> Reference:
    """Copy a file into the library and cache its analysis. The first in a genre is the default."""
    if not audio_io.is_audio(source):
        raise ReferenceError(f"not an audio file: {source.name}")
    if _norm(source.stem) in RESERVED:
        raise ReferenceError(
            f"a reference cannot be called '{source.stem}': that name means "
            "the genre's default or the closest match. Rename the file first."
        )
    genre = _match_genre(paths, genre) or projects.slugify(genre)
    dest = paths.references / genre / source.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and audio_io.file_sha256(dest) != audio_io.file_sha256(source):
        raise ReferenceError(f"a different reference named {source.name} already exists in {genre}")
    if not dest.exists():
        shutil.copy2(source, dest)
    report, quality = analyze(dest, settings)
    sidecar = {
        "notes": notes,
        "added_at": projects.now_iso(),
        "analysis": report.to_dict(),
        "quality": asdict(quality),
    }
    projects.write_json(_sidecar(dest), sidecar)
    if make_default or genre not in _defaults(paths):
        set_default(paths, genre, dest.stem)
    return get(paths, genre, dest.stem)


def get(paths: Paths, genre: str, name: str) -> Reference:
    candidates = list_references(paths, genre)
    if _norm(name) == "default":
        match = next((r for r in candidates if r.is_default), None)
    else:
        match = next(
            (r for r in candidates if name.lower() in (r.name.lower(), r.path.name.lower())), None
        )
    if match is None:
        raise ReferenceError(f"no reference '{name}' in genre '{genre}'")
    return match


def set_default(paths: Paths, genre: str, name: str) -> None:
    matched = _match_genre(paths, genre)
    files = {
        f.stem.lower(): f
        for f in (paths.references / (matched or genre)).glob("*")
        if audio_io.is_audio(f)
    }
    if matched is None or name.lower() not in files:
        raise ReferenceError(f"no reference '{name}' in genre '{genre}'")
    projects.write_json(
        _defaults_file(paths), {**_defaults(paths), matched: files[name.lower()].name}
    )


def remove(paths: Paths, genre: str, name: str) -> None:
    reference = get(paths, genre, name)
    reference.path.unlink()
    _sidecar(reference.path).unlink(missing_ok=True)
    defaults = _defaults(paths)
    if reference.is_default:
        remaining = list_references(paths, reference.genre)
        if remaining:
            defaults[reference.genre] = remaining[0].path.name
        else:
            defaults.pop(reference.genre, None)
        projects.write_json(_defaults_file(paths), defaults)


def resolve(paths: Paths, spec: str, settings: Settings) -> Reference:
    """Turn 'genre/default', 'genre/name' or a file path into a Reference."""
    file = Path(spec).expanduser()
    if file.is_file():
        report, quality = analyze(file, settings)
        return Reference("", file.stem, file, report, quality=quality)
    if auto_genre(spec) is not None:
        raise ReferenceError("'auto' picks a reference for each mix; it is not one reference")
    genre, _, name = spec.partition("/")
    if not name:
        raise ReferenceError(
            f"reference must be 'genre/name', 'genre/default', 'auto' or a file: {spec}"
        )
    return get(paths, genre, name)
