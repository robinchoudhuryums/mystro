"""AI mix notes: send a run's measurements to Claude and get back prioritized feedback.

Only numbers and short text labels leave the machine. No audio, no file paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import resources
from typing import Any

import anthropic

from masterbench import projects
from masterbench.analysis.report import AnalysisReport
from masterbench.config import Paths, Settings
from masterbench.master.pipeline import RunResult
from masterbench.preview import streaming
from masterbench.stems import workflow

# USD per million tokens (input, output).
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.00, 20.00),
}
# Refusals on these models can be retried on another model by the API itself.
SERVER_FALLBACK_MODELS = frozenset({"claude-sonnet-5-5", "claude-opus-5-5"})
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16000
EFFORT = "medium"
_TYPICAL_OUTPUT_TOKENS = 3000
_CHARS_PER_TOKEN = 3.0
KEYCHAIN_SERVICE = "masterbench-anthropic"
NOTES_FILE = "notes.json"

_ISSUE = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "impact": {"type": "string", "enum": ["high", "medium", "low"]},
        "what": {"type": "string"},
        "evidence": {"type": "string"},
        "fix_in_mix": {"type": "string"},
        "fix_in_master": {"type": "string"},
    },
    "required": ["title", "impact", "what", "evidence", "fix_in_mix", "fix_in_master"],
    "additionalProperties": False,
}
NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "issues": {"type": "array", "items": _ISSUE},
        "already_good": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "issues", "already_good", "caveats"],
    "additionalProperties": False,
}


class NotesError(RuntimeError):
    pass


@dataclass(frozen=True)
class Estimate:
    model: str
    input_tokens: int
    cost_usd: float
    spent_this_month_usd: float
    cap_usd: float

    @property
    def over_cap(self) -> bool:
        return self.spent_this_month_usd + self.cost_usd > self.cap_usd


@dataclass(frozen=True)
class MixNotes:
    run_id: str
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


def system_prompt() -> tuple[str, int]:
    text = resources.files("masterbench.notes").joinpath("prompt.md").read_text(encoding="utf-8")
    version = re.search(r"<!-- prompt-version: (\d+) -->", text)
    if version is None:
        raise NotesError("notes/prompt.md has no prompt-version marker")
    return text[version.end() :].strip(), int(version.group(1))


def _analysis(report: AnalysisReport) -> dict[str, Any]:
    info = report.info
    data: dict[str, Any] = {
        "file": {
            "sample_rate": info.sample_rate,
            "channels": info.channels,
            "bit_depth": info.bit_depth,
            "duration_s": round(info.duration_s, 1),
            "lossy": info.lossy,
        },
        "loudness": {key: round(value, 2) for key, value in asdict(report.loudness).items()},
        "flags": [
            {"code": f.code, "severity": f.severity, "message": f.message, "value": f.value}
            for f in report.flags
        ],
    }
    if report.bands:
        data["bands_db"] = report.bands
    if report.band_deltas:
        data["band_deltas_db"] = report.band_deltas
    if report.stereo:
        data["stereo"] = asdict(report.stereo)
    if report.spectrum:
        # One point per third of an octave is plenty for a written review.
        freqs, levels = report.spectrum.freqs_hz[::4], report.spectrum.level_db[::4]
        data["spectrum"] = {"freqs_hz": freqs, "level_db": levels}
    if report.short_term_lufs:
        curve = report.short_term_lufs
        step = max(1, len(curve) // 60)
        data["short_term_lufs"] = {"step_s": step, "values": curve[::step]}
    return data


def _processing(run: RunResult) -> dict[str, Any]:
    settings = run.manifest.settings
    chains = {
        key: [module["type"] for module in settings[key]["module"]]
        for key in ("pre_chain", "post_chain")
        if settings.get(key)
    }
    return {
        "kind": run.manifest.kind,
        "target_lufs": settings.get("target_lufs"),
        "ceiling_dbtp": settings.get("ceiling_dbtp"),
        "chains": chains,
        "stats": {k: v for k, v in run.manifest.stats.items() if k != "seconds"},
    }


def _stems(run: RunResult) -> dict[str, Any] | None:
    for stem_set in workflow.sets_for_run(run):
        diagnosis_file = stem_set.dir / "diagnosis.json"
        if diagnosis_file.exists():
            diagnosis = json.loads(diagnosis_file.read_text(encoding="utf-8"))
            return {
                "separated": stem_set.separated,
                "source": "model" if stem_set.separated else "exported from the DAW",
                "roles": dict(stem_set.roles),
                "balance": diagnosis["balance"],
                "reference_balance": diagnosis["reference"],
            }
    return None


def build_payload(run: RunResult, intent: str = "") -> dict[str, Any]:
    """Everything that is sent: measurements and labels, never audio or paths."""
    payload: dict[str, Any] = {
        "track": {"title": run.manifest.track.title, "genre": run.manifest.track.genre},
        "processing": _processing(run),
        "analyses": {role: _analysis(report) for role, report in sorted(run.reports.items())},
    }
    if intent.strip():
        payload["intent"] = intent.strip()
    stems = _stems(run)
    if stems:
        payload["stems"] = stems
    preview = streaming.load(run) if run.manifest.outputs else None
    if preview:
        payload["streaming"] = {
            "codec_true_peaks_dbtp": {c.name: c.true_peak_dbtp for c in preview.codecs},
            "platform_gain_db": {p.platform: p.gain_db for p in preview.platforms},
        }
    return payload


def payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=1, sort_keys=True)


def payload_hash(payload: dict[str, Any], model: str, prompt_version: int) -> str:
    key = json.dumps([payload, model, prompt_version], sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()


def price(model: str) -> tuple[float, float]:
    if model not in PRICES:
        raise NotesError(f"no price known for model '{model}'; use one of {', '.join(PRICES)}")
    return PRICES[model]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    per_input, per_output = price(model)
    return round((input_tokens * per_input + output_tokens * per_output) / 1_000_000, 5)


def _stored_notes(paths: Paths) -> list[dict[str, Any]]:
    files = paths.projects.glob(f"*/runs/*/{NOTES_FILE}")
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(files)]


def spent_this_month(paths: Paths) -> float:
    month = projects.now_iso()[:7]
    return round(
        sum(n["cost_usd"] for n in _stored_notes(paths) if n["created_at"][:7] == month), 5
    )


def estimate(payload: dict[str, Any], model: str, paths: Paths, settings: Settings) -> Estimate:
    """A rough cost before sending. The real figure comes back with the response."""
    prompt, _ = system_prompt()
    tokens = int((len(prompt) + len(payload_json(payload))) / _CHARS_PER_TOKEN)
    return Estimate(
        model=model,
        input_tokens=tokens,
        cost_usd=cost_usd(model, tokens, _TYPICAL_OUTPUT_TOKENS),
        spent_this_month_usd=spent_this_month(paths),
        cap_usd=settings.notes_monthly_cap_usd,
    )


def _keychain_key() -> str | None:
    if sys.platform != "darwin" or shutil.which("security") is None:
        return None
    cmd = ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"]
    found = subprocess.run(cmd, capture_output=True, text=True)
    return found.stdout.strip() or None if found.returncode == 0 else None


def key_status() -> str:
    """Where an API key would come from, without revealing it."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY environment variable"
    if _keychain_key():
        return f"macOS Keychain item '{KEYCHAIN_SERVICE}'"
    return "none found (an `ant auth login` profile, if you have one, is used automatically)"


def make_client() -> anthropic.Anthropic:
    """The environment variable or an SDK login profile, else the macOS Keychain."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        key = _keychain_key()
        if key:
            return anthropic.Anthropic(api_key=key)
    return anthropic.Anthropic()


def _request(client: anthropic.Anthropic, model: str, payload: dict[str, Any]) -> Any:
    prompt, _ = system_prompt()
    arguments: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": prompt,
        "messages": [{"role": "user", "content": payload_json(payload)}],
        "output_config": {
            "effort": EFFORT,
            "format": {"type": "json_schema", "schema": NOTES_SCHEMA},
        },
    }
    try:
        if model in SERVER_FALLBACK_MODELS:
            return client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **arguments
            )
        return client.messages.create(**arguments)
    except anthropic.AuthenticationError as exc:
        raise NotesError(
            "The Anthropic API rejected the key. Set ANTHROPIC_API_KEY, or store one with: "
            f"security add-generic-password -s {KEYCHAIN_SERVICE} -a anthropic -w"
        ) from exc
    except anthropic.RateLimitError as exc:
        raise NotesError("The Anthropic API is rate limiting this key; try again shortly.") from exc
    except anthropic.APIStatusError as exc:
        raise NotesError(
            f"The Anthropic API returned an error ({exc.status_code}): {exc.message}"
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise NotesError(f"Could not reach the Anthropic API: {exc}") from exc
    except anthropic.AnthropicError as exc:
        raise NotesError(f"No usable Anthropic credentials: {exc}") from exc


def _body(response: Any) -> dict[str, Any]:
    if response.stop_reason == "refusal":
        raise NotesError("The model declined to answer this request.")
    if response.stop_reason == "max_tokens":
        raise NotesError("The reply was cut off before it finished; nothing was saved.")
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise NotesError("The reply contained no text.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise NotesError("The reply was not valid JSON; nothing was saved.") from exc


def generate(
    run: RunResult,
    paths: Paths,
    settings: Settings,
    intent: str = "",
    model: str | None = None,
    client: anthropic.Anthropic | None = None,
    force: bool = False,
) -> MixNotes:
    """Notes for a run. Identical measurements, model and prompt reuse the stored answer.

    force sends the request even when it would pass the monthly spend cap.
    """
    model = model or settings.notes_model
    payload = build_payload(run, intent)
    _, version = system_prompt()
    digest = payload_hash(payload, model, version)

    earlier = next((n for n in _stored_notes(paths) if n["payload_hash"] == digest), None)
    if earlier is not None:
        notes = MixNotes(**{**earlier, "run_id": run.manifest.id, "cost_usd": 0.0, "cached": True})
        if earlier["run_id"] != run.manifest.id:
            _save(run, notes)
        return notes

    quote = estimate(payload, model, paths, settings)
    if quote.over_cap and not force:
        raise NotesError(
            f"This would pass the monthly cap: ${quote.spent_this_month_usd:.2f} spent, about "
            f"${quote.cost_usd:.2f} more, cap ${quote.cap_usd:.2f}. Raise notes_monthly_cap_usd "
            "in settings.toml or force it."
        )
    response = _request(client or make_client(), model, payload)
    body = _body(response)
    usage = response.usage
    notes = MixNotes(
        run_id=run.manifest.id,
        model=model,
        served_by=response.model,
        prompt_version=version,
        payload_hash=digest,
        created_at=projects.now_iso(),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost_usd(model, usage.input_tokens, usage.output_tokens),
        cached=False,
        body=body,
    )
    _save(run, notes)
    return notes


def _save(run: RunResult, notes: MixNotes) -> None:
    projects.write_json(run.run_dir / NOTES_FILE, asdict(notes))
    (run.run_dir / "notes.md").write_text(render(notes) + "\n", encoding="utf-8")


def load(run: RunResult) -> MixNotes | None:
    file = run.run_dir / NOTES_FILE
    return MixNotes(**json.loads(file.read_text(encoding="utf-8"))) if file.exists() else None


def render(notes: MixNotes) -> str:
    """Markdown for the UI, the CLI and notes.md."""
    body = notes.body
    lines = [body["summary"], ""]
    for number, issue in enumerate(body["issues"], start=1):
        lines += [
            f"### {number}. {issue['title']} ({issue['impact']} impact)",
            "",
            issue["what"],
            "",
            f"- **Evidence:** {issue['evidence']}",
            f"- **In the mix:** {issue['fix_in_mix']}",
            f"- **In the master:** {issue['fix_in_master']}",
            "",
        ]
    if body["already_good"]:
        lines += ["### Already working", "", *[f"- {item}" for item in body["already_good"]], ""]
    if body["caveats"]:
        lines += ["### Caveats", "", *[f"- {item}" for item in body["caveats"]], ""]
    source = "stored answer, no charge" if notes.cached else f"${notes.cost_usd:.3f}"
    served = notes.served_by if notes.served_by == notes.model else f"{notes.served_by} (fallback)"
    lines.append(
        f"_{served}, prompt v{notes.prompt_version}, {notes.input_tokens} tokens in, "
        f"{notes.output_tokens} out, {source}. Written from measurements only; "
        "the model did not hear the audio._"
    )
    return "\n".join(lines)
