"""AI trend summary: the top habits across tracks and one practice exercise for each.

Like the mix notes, this sends numbers and track titles only.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

import anthropic

from masterbench import progress, projects
from masterbench.config import Paths, Settings
from masterbench.notes import mix_notes
from masterbench.notes.mix_notes import NotesError
from masterbench.progress import Habit, TrackPoint

PROMPT_FILE = "trends_prompt.md"
MAX_TRACKS = 12

_HABIT = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "evidence": {"type": "string"},
        "why_it_matters": {"type": "string"},
        "exercise": {"type": "string"},
        "how_to_tell_it_worked": {"type": "string"},
    },
    "required": ["title", "evidence", "why_it_matters", "exercise", "how_to_tell_it_worked"],
    "additionalProperties": False,
}
TRENDS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "habits": {"type": "array", "items": _HABIT},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "habits", "caveats"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class TrendSummary:
    model: str
    served_by: str
    prompt_version: int
    payload_hash: str
    created_at: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool
    body: dict[str, Any]


def _track(point: TrackPoint) -> dict[str, Any]:
    return {
        "title": point.title,
        "genre": point.genre,
        "date": point.created_at[:10],
        "lufs_i": round(point.lufs_i, 1),
        "crest_db": round(point.crest_db, 1),
        "lra": round(point.lra, 1),
        "band_deltas_db": point.band_deltas,
        "compared_with": point.delta_vs,
        "hook_minus_verse_lu": point.hook_contrast_lu,
        "bass_in_tune_share": point.bass_in_tune_share,
        "flags": sorted(set(point.flags)),
    }


def build_payload(
    points: list[TrackPoint], habits: list[Habit], blind_wins: dict[str, int] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tracks_considered": min(len(points), progress.HABIT_WINDOW),
        "habits": [asdict(habit) for habit in habits],
        "tracks": [_track(point) for point in points[-MAX_TRACKS:]],
    }
    if blind_wins:
        payload["blind_tests"] = {"preferred": blind_wins}
    return payload


def gather(conn: sqlite3.Connection, settings: Settings) -> dict[str, Any]:
    """The payload for the tracks currently in the index."""
    points = progress.track_points(conn)
    habits = progress.habits(points, settings.thresholds)
    return build_payload(points, habits, progress.blind_summary(progress.blind_results(conn)))


def generate(
    payload: dict[str, Any],
    paths: Paths,
    settings: Settings,
    model: str | None = None,
    client: anthropic.Anthropic | None = None,
    force: bool = False,
) -> TrendSummary:
    """A trend summary for a payload. The same numbers, model and prompt reuse a stored answer."""
    if not payload["tracks"]:
        raise NotesError("there are no analyzed tracks to summarize yet")
    model = model or settings.notes_model
    prompt, version = mix_notes.load_prompt(PROMPT_FILE)
    digest = mix_notes.payload_hash(payload, model, version)
    earlier = next((t for t in mix_notes.stored_trends(paths) if t["payload_hash"] == digest), None)
    if earlier is not None:
        return TrendSummary(**{**earlier, "cost_usd": 0.0, "cached": True})

    quote = mix_notes.estimate(payload, model, paths, settings, prompt)
    if quote.over_cap and not force:
        raise NotesError(
            f"This would pass the monthly cap: ${quote.spent_this_month_usd:.2f} spent, about "
            f"${quote.cost_usd:.2f} more, cap ${quote.cap_usd:.2f}. Raise notes_monthly_cap_usd "
            "in settings.toml or force it."
        )
    response = mix_notes.ask(
        client or mix_notes.make_client(), model, prompt, payload, TRENDS_SCHEMA
    )
    body = mix_notes.parse_reply(response)
    usage = response.usage
    summary = TrendSummary(
        model=model,
        served_by=response.model,
        prompt_version=version,
        payload_hash=digest,
        created_at=projects.now_iso(),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=mix_notes.cost_usd(model, usage.input_tokens, usage.output_tokens),
        cached=False,
        body=body,
    )
    stamp = summary.created_at[:19].replace(":", "").replace("-", "")
    projects.write_json(mix_notes.trends_dir(paths) / f"{stamp}-{digest[:8]}.json", asdict(summary))
    return summary


def latest(paths: Paths) -> TrendSummary | None:
    stored = mix_notes.stored_trends(paths)
    return TrendSummary(**stored[-1]) if stored else None


def render(summary: TrendSummary) -> str:
    body = summary.body
    lines = [body["summary"], ""]
    for number, habit in enumerate(body["habits"], start=1):
        lines += [
            f"### {number}. {habit['title']}",
            "",
            f"- **Evidence:** {habit['evidence']}",
            f"- **Why it matters:** {habit['why_it_matters']}",
            f"- **Exercise:** {habit['exercise']}",
            f"- **You will know it worked when:** {habit['how_to_tell_it_worked']}",
            "",
        ]
    if body["caveats"]:
        lines += ["### Caveats", "", *[f"- {item}" for item in body["caveats"]], ""]
    source = "stored answer, no charge" if summary.cached else f"${summary.cost_usd:.3f}"
    lines.append(
        f"_{summary.served_by}, prompt v{summary.prompt_version}, {source}. Written from "
        "measurements only; the model did not hear the audio._"
    )
    return "\n".join(lines)
