"""Python side of the A/B/C player: which sources a run offers, and saving blind tests."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import gradio as gr

from masterbench import blind, index
from masterbench.config import Paths
from masterbench.master.pipeline import RunResult
from masterbench.preview.streaming import StreamingPreview
from masterbench.stems.separate import StemSet

# The JSON config travels in the component's value; the script reads it from props.
TEMPLATE = """
<div class="mb-player">
  <div class="mb-sources"></div>
  <div class="mb-transport">
    <button class="mb-play">Play</button>
    <span class="mb-time">0:00 / 0:00</span>
    <input class="mb-seek" type="range" min="0" max="1000" value="0" aria-label="Position">
  </div>
  <div class="mb-options">
    <label><input class="mb-match" type="checkbox" checked> Match loudness</label>
    <button class="mb-loop-in">Loop from here</button>
    <button class="mb-loop-out">Loop to here</button>
    <button class="mb-loop-clear">Clear loop</button>
    <span class="mb-loop">No loop</span>
    <button class="mb-blind">Blind test</button>
    <button class="mb-save">Reveal and save</button>
  </div>
  <div class="mb-hint"></div>
  <div class="mb-status"></div>
</div>
"""

CSS = """
.mb-player { display: flex; flex-direction: column; gap: 10px; padding: 12px;
  border: 1px solid var(--border-color-primary); border-radius: 8px; }
.mb-sources { display: flex; flex-direction: column; gap: 6px; }
.mb-source { display: flex; align-items: center; gap: 10px; }
.mb-source > button:first-child { flex: 1; text-align: left; white-space: pre; }
.mb-player button { padding: 6px 12px; border-radius: 6px; cursor: pointer;
  border: 1px solid var(--border-color-primary);
  background: var(--button-secondary-background-fill);
  color: var(--button-secondary-text-color); }
.mb-player button:disabled { opacity: 0.5; cursor: default; }
.mb-active > button:first-child { background: var(--button-primary-background-fill);
  color: var(--button-primary-text-color); border-color: transparent; font-weight: 600; }
.mb-level, .mb-time, .mb-loop, .mb-hint, .mb-status { font-variant-numeric: tabular-nums;
  color: var(--body-text-color-subdued); font-size: 0.9em; }
.mb-transport, .mb-options { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.mb-seek { flex: 1; min-width: 200px; }
.mb-play { min-width: 72px; }
"""


def file_url(path: Path | str) -> str:
    return f"/gradio_api/file={path}"


def sources(run: RunResult, preview: StreamingPreview | None = None) -> list[dict[str, Any]]:
    """Everything this run can put in the player, with the loudness needed to level-match."""
    found = [{"id": "mix", "label": "Mix", "path": run.mix_path, "lufs": run.reports["mix"]}]
    if "master_24" in run.manifest.outputs:
        found.append(
            {
                "id": "master",
                "label": "Master",
                "path": run.output("master_24"),
                "lufs": run.reports["master"],
            }
        )
    if "reference" in run.reports:
        reference = run.reports["reference"]
        found.append(
            {
                "id": "reference",
                "label": "Reference",
                "path": reference.source_path,
                "lufs": reference,
            }
        )
    listed = [
        {
            "id": s["id"],
            "label": s["label"],
            "url": file_url(s["path"]),
            "lufs": round(s["lufs"].loudness.lufs_i, 2),
        }
        for s in found
    ]
    for codec in preview.codecs if preview else []:
        listed.append(
            {
                "id": f"codec_{codec.name}",
                "label": codec.label,
                "url": file_url(run.run_dir / codec.decoded_file),
                "lufs": codec.lufs_i,
            }
        )
    return listed


def other_master(run: RunResult) -> dict[str, Any]:
    """A master from another run of the same track, labelled by its target."""
    target = run.manifest.settings.get("target_lufs")
    what = f"{target:g} LUFS" if target is not None else "reference loudness"
    return {
        "id": f"master_{run.manifest.id}",
        "label": f"Master {what} ({run.manifest.created_at[5:16].replace('T', ' ')})",
        "url": file_url(run.output("master_24")),
        "lufs": round(run.reports["master"].loudness.lufs_i, 2),
    }


def stem_sources(stem_set: StemSet) -> list[dict[str, Any]]:
    """Stem solos. Stems silent enough to have no loudness are left out."""
    kind = "separated" if stem_set.separated else "Logic"
    return [
        {
            "id": f"stem_{name}",
            "label": f"Stem: {name} ({kind})",
            "url": file_url(stem_set.path(name)),
            "lufs": level,
        }
        for name, level in (stem_set.lufs or {}).items()
        if level > -70
    ]


def config(
    run: RunResult | None,
    preview: StreamingPreview | None = None,
    extra: list[RunResult] | None = None,
    stems: StemSet | None = None,
) -> str:
    """JSON for the player component's value."""
    if run is None:
        return json.dumps({"sources": [], "message": "Choose a run to load its audio."})
    listed = sources(run, preview) + [other_master(other) for other in extra or []]
    listed += stem_sources(stems) if stems else []
    selected = "master" if any(s["id"] == "master" for s in listed) else "mix"
    return json.dumps({"track": run.manifest.track.slug, "selected": selected, "sources": listed})


def component(paths: Paths) -> gr.HTML:
    def save_blind_test(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            test = blind.record(
                paths,
                payload["track"],
                payload["candidates"],
                payload["ranking"],
                payload["revealed"],
            )
        except (LookupError, ValueError, KeyError, TypeError) as exc:
            return {"error": str(exc)}
        with index.open_index(paths) as conn:
            index.sync_blind_tests(conn, paths)
        return {"id": test.id, "winner": test.winner}

    script = resources.files("masterbench.ui").joinpath("ab_player.js").read_text(encoding="utf-8")
    return gr.HTML(
        value=config(None),
        html_template=TEMPLATE,
        css_template=CSS,
        js_on_load=script,
        server_functions=[save_blind_test],
        elem_id="mb-ab-player",
    )
