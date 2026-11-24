"""Project and run storage. The JSON written here is the source of truth; the DB indexes it."""

from __future__ import annotations

import json
import re
import secrets
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from masterbench import audio_io
from masterbench.analysis.report import AnalysisReport
from masterbench.config import Paths

RunKind = Literal["master", "analysis_only", "batch", "version", "tune"]

TRACKED_LIBRARIES = ("masterbench", "matchering", "pedalboard", "pyloudnorm", "numpy", "scipy")
# "Song v3", "Song_mix2", "Song bounce 4" are takes of one track.
_TAKE_SUFFIX = re.compile(r"[\s_\-]+(?:v|ver|mix|bounce|master)[\s_\-]*\d+$", re.IGNORECASE)


def now_iso() -> str:
    """UTC timestamp with milliseconds, so runs made in the same second still sort in order."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def title_from_filename(path: Path) -> str:
    return _TAKE_SUFFIX.sub("", path.stem).strip() or path.stem


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def new_run_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def lib_versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in TRACKED_LIBRARIES}


@dataclass(frozen=True)
class Track:
    slug: str
    title: str
    genre: str | None
    created_at: str


@dataclass(frozen=True)
class RunManifest:
    id: str
    track: Track
    kind: RunKind
    created_at: str
    input_file: str
    input_sha256: str
    original_path: str
    settings: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    lib_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        return cls(**{**data, "track": Track(**data["track"])})


@dataclass(frozen=True)
class Project:
    track: Track
    dir: Path
    input_path: Path
    input_sha256: str

    def run_dir(self, run_id: str) -> Path:
        return self.dir / "runs" / run_id


def _store_input(project_dir: Path, source: Path, sha256: str) -> Path:
    """Copy the bounce into the project. An existing different file is never overwritten."""
    inputs = project_dir / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    dest = inputs / source.name
    if dest.exists() and audio_io.file_sha256(dest) != sha256:
        dest = inputs / f"{source.stem}-{sha256[:8]}{source.suffix}"
    if not dest.exists():
        shutil.copy2(source, dest)
    return dest


def open_project(
    paths: Paths, source: Path, title: str | None = None, genre: str | None = None
) -> Project:
    """Find or create the project for a bounce and store the bounce in it."""
    title = title or title_from_filename(source)
    project_dir = paths.projects / slugify(title)
    track_file = project_dir / "track.json"
    if track_file.exists():
        track = Track(**json.loads(track_file.read_text(encoding="utf-8")))
        if genre and genre != track.genre:
            track = Track(track.slug, track.title, genre, track.created_at)
    else:
        track = Track(slugify(title), title, genre, now_iso())
    project_dir.mkdir(parents=True, exist_ok=True)
    write_json(track_file, asdict(track))
    sha256 = audio_io.file_sha256(source)
    return Project(track, project_dir, _store_input(project_dir, source, sha256), sha256)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_run(run_dir: Path, manifest: RunManifest, reports: dict[str, AnalysisReport]) -> None:
    write_json(run_dir / "analysis.json", {name: r.to_dict() for name, r in reports.items()})
    write_json(run_dir / "run.json", manifest.to_dict())


def read_run(run_dir: Path) -> tuple[RunManifest, dict[str, AnalysisReport]]:
    manifest = RunManifest.from_dict(json.loads((run_dir / "run.json").read_text("utf-8")))
    raw = json.loads((run_dir / "analysis.json").read_text("utf-8"))
    return manifest, {name: AnalysisReport.from_dict(data) for name, data in raw.items()}


def iter_run_dirs(paths: Paths) -> list[Path]:
    """Every complete run on disk, oldest first."""
    found = [p.parent for p in paths.projects.glob("*/runs/*/run.json")]
    return sorted(found, key=lambda run_dir: run_dir.name)
