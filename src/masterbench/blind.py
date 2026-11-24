"""Blind A/B/C tests: which version was preferred once the labels were hidden."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from masterbench import projects
from masterbench.config import Paths


@dataclass(frozen=True)
class BlindTest:
    id: str
    track_slug: str
    created_at: str
    candidates: list[dict[str, str]]
    ranking: list[str]
    revealed: dict[str, str]

    @property
    def winner(self) -> str:
        return self.revealed[self.ranking[0]]


def _dir(paths: Paths, track_slug: str) -> Path:
    return paths.projects / track_slug / "blind_tests"


def record(
    paths: Paths,
    track_slug: str,
    candidates: list[dict[str, str]],
    ranking: list[str],
    revealed: dict[str, str],
) -> BlindTest:
    """Store a finished test. candidates: [{"id", "label"}]; ranking: blind letters, best first;
    revealed: blind letter -> candidate id."""
    ids = {c["id"] for c in candidates}
    if not (paths.projects / track_slug / "track.json").exists():
        raise LookupError(f"no track {track_slug}")
    if sorted(ranking) != sorted(revealed) or set(revealed.values()) != ids or len(ids) < 2:
        raise ValueError("ranking and revealed must cover every candidate exactly once")
    test = BlindTest(
        projects.new_run_id(), track_slug, projects.now_iso(), candidates, ranking, revealed
    )
    projects.write_json(_dir(paths, track_slug) / f"{test.id}.json", asdict(test))
    return test


def load_all(paths: Paths) -> list[BlindTest]:
    files = sorted(paths.projects.glob("*/blind_tests/*.json"), key=lambda f: f.name)
    return [BlindTest(**json.loads(f.read_text(encoding="utf-8"))) for f in files]
