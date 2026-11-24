"""The Vocal tab: pitch report, tuning, A/B against the original, and the pitch editor."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import gradio as gr

from masterbench import audio_io, summary
from masterbench.analysis.musical import NOTE_NAMES
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.stems import separate
from masterbench.stems.separate import StemError
from masterbench.ui import player
from masterbench.vocal import engines, presets, scales, tuning, workflow
from masterbench.vocal.pitch_report import PitchReport
from masterbench.vocal.tuning import NoteEdit, TuningError
from masterbench.vocal.workflow import VocalError, VocalSource

KEYS = [f"{note} {mode}" for mode in ("minor", "major") for note in NOTE_NAMES]
ERRORS = (
    VocalError,
    TuningError,
    scales.ScaleError,
    engines.EngineError,
    StemError,
    audio_io.AudioError,
    LookupError,
)
EDITOR_TEMPLATE = """
<div class="mb-pitch">
  <div class="mb-pitch-empty">Check a vocal's pitch to see its notes here.</div>
  <div class="mb-pitch-view">
    <svg class="mb-pitch-labels"></svg>
    <div class="mb-pitch-scroll"><svg class="mb-pitch-grid"></svg></div>
  </div>
  <div class="mb-pitch-status"></div>
</div>
"""
EDITOR_CSS = """
.mb-pitch { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px; }
.mb-pitch-view { display: flex; max-height: 420px; overflow-y: auto; }
.mb-pitch-scroll { overflow-x: auto; flex: 1; }
.mb-pitch svg { display: block; flex: none; user-select: none; }
.mb-pitch text { font-size: 10px; fill: var(--body-text-color-subdued); }
.mb-pitch-labels text { text-anchor: end; }
.mb-pitch .mb-label-in { fill: var(--body-text-color); }
.mb-row-in { fill: var(--background-fill-secondary); }
.mb-row-out { fill: var(--background-fill-primary); }
.mb-second { stroke: var(--border-color-primary); stroke-width: 1; }
.mb-curve { fill: none; stroke: var(--body-text-color-subdued); stroke-width: 1.2; }
.mb-note { cursor: grab; }
.mb-note-target { fill: #3b82f6; fill-opacity: 0.55; }
.mb-note-sung { stroke: #16a34a; stroke-width: 3; }
.mb-note[data-state="off"] .mb-note-sung { stroke: #d97706; }
.mb-note[data-state="far"] .mb-note-sung { stroke: #dc2626; }
.mb-note[data-state="edited"] .mb-note-target { fill: #9333ea; fill-opacity: 0.7; }
.mb-note[data-state="bypass"] .mb-note-target { fill: #9ca3af; fill-opacity: 0.35; }
.mb-pitch .mb-note-label { fill: #fff; pointer-events: none; }
.mb-pitch-status, .mb-pitch-empty { margin-top: 8px; color: var(--body-text-color-subdued); }
"""
INTRO = (
    "Tune a **dry vocal** exported from Logic. Check the pitch first: it changes nothing and "
    "shows which notes are out. Logic's Flex Pitch is still the tool for detailed note-by-note "
    "work; this is for a quick, scale-aware pass and for seeing how in tune a take is."
)


def source_choices(paths: Paths) -> list[tuple[str, str]]:
    """Vocals already brought in, then tracks whose stems hold a vocal that is not."""
    found = [(f"{s.track.title}: {s.label}", s.id) for s in workflow.list_sources(paths)]
    seen = {s.path for s in workflow.list_sources(paths)}
    for project_dir in sorted(paths.projects.glob("*/")):
        for stem_set in separate.list_sets(project_dir):
            vocals = [stem_set.path(name) for name in stem_set.by_role("vocals")]
            if len(vocals) == 1 and vocals[0] not in seen:
                kind = "separated" if stem_set.separated else "Logic stem"
                found.append(
                    (f"{project_dir.name}: vocal stem ({kind})", f"stems:{project_dir.name}")
                )
                break
    return found


def editor_value(source: VocalSource | None, found: PitchReport | None) -> str:
    """JSON for the pitch editor: the curve, the notes and the edits made so far."""
    if source is None or found is None:
        return json.dumps({"notes": []})
    edits = workflow.read_edits(source)
    return json.dumps(
        {
            "source": source.id,
            "duration_s": found.duration_s,
            "step_s": found.curve_step_s,
            "curve": found.curve,
            "scale": [NOTE_NAMES.index(name) for name in found.scale_notes],
            "notes": [
                {
                    "index": note.index,
                    "start_s": note.start_s,
                    "end_s": note.end_s,
                    "center": note.center,
                    "target": note.target,
                    "name": note.name,
                    "cents": note.cents,
                }
                for note in found.notes
            ],
            "edits": {
                str(index): {"bypass": edit.bypass, "target": edit.target}
                for index, edit in edits.items()
            },
        }
    )


def player_config(source: VocalSource | None, selected: str | None = None) -> str:
    """The vocal, its tuned versions and anything added for comparison, for the A/B player."""
    if source is None:
        return json.dumps({"sources": [], "message": "Check or tune a vocal to hear it here."})
    listed = workflow.candidates(source) + workflow.mix_candidates(source)
    found = workflow.stored_report(source)
    events = [
        {
            "label": f"note {note.index + 1} ({round(note.cents):+d} c)",
            "start": note.start_s,
            "title": f"{note.name}, {round(note.cents):+d} cents from {note.target_name}",
        }
        for note in (found.notes if found else [])
        if abs(note.cents) >= 20
    ]
    ids = [item["id"] for item in listed]
    return json.dumps(
        {
            "track": source.track.slug,
            "selected": selected if selected in ids else ids[-1] if len(ids) > 1 else ids[0],
            "sources": [
                {
                    "id": item["id"],
                    "label": item["label"],
                    "url": player.file_url(item["path"]),
                    "lufs": item["lufs"],
                }
                for item in listed
            ],
            "sections": [],
            "events": events[:24],
        }
    )


def key_note(found: PitchReport) -> str:
    where = {
        "track": "from the track's analysis",
        "detected": "guessed from the sung notes",
        "given": "set here",
    }[found.key_source]
    confidence = ""
    if found.key_confidence is not None and found.key_source != "given":
        confidence = f", confidence {found.key_confidence:.0%}"
    return f"Key: **{found.key}** ({where}{confidence}). Change it on the left if it is wrong."


def build(paths: Paths, settings: Settings) -> None:
    """Add the Vocal tab to the Blocks being built."""
    names = presets.names(paths)
    start = presets.get(paths, presets.DEFAULT).settings("C major")

    def save_edits(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            source = workflow.load_source(paths, str(payload["source"]))
            assert source is not None
            edits = {
                int(index): NoteEdit(bool(edit.get("bypass")), edit.get("target"))
                for index, edit in payload["edits"].items()
            }
            return {"saved": len(workflow.write_edits(source, edits))}
        except (*ERRORS, KeyError, TypeError, ValueError) as exc:
            return {"error": str(exc)}

    with gr.Tab("Vocal", elem_id="mb-vocal-tab"):
        gr.Markdown(INTRO)
        current = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                voc_file = gr.File(
                    label="Vocal (or a full mix, with the box below ticked)",
                    file_types=[".wav", ".aif", ".aiff", ".flac"],
                    type="filepath",
                )
                voc_pick = gr.Dropdown(
                    label="Or a vocal already here",
                    choices=source_choices(paths),
                    value=None,
                )
                voc_title = gr.Textbox(
                    label="Track it belongs to (optional)",
                    info="Uses that track's key and files the vocal with it.",
                )
                voc_mix = gr.Checkbox(
                    label="This is a full mix: separate the vocal first (preview quality)"
                )
                voc_key = gr.Dropdown(label="Key", choices=KEYS, value=None)
                voc_preset = gr.Radio(label="Style", choices=names, value=presets.DEFAULT)
                voc_amount = gr.Slider(
                    0, 100, value=round(start.amount * 100), step=1, label="Amount (%)"
                )
                with gr.Row():
                    voc_check = gr.Button("Check pitch")
                    voc_tune = gr.Button("Tune", variant="primary")
            with gr.Column(scale=2):
                voc_key_md = gr.Markdown()
                voc_result = gr.Markdown()
                voc_player = player.component(paths, "mb-vocal-player", player_config(None))
                voc_files = gr.File(label="Tuned files", file_count="multiple", interactive=False)
                voc_report = gr.Markdown()
        with gr.Accordion("Advanced", open=False):
            with gr.Row():
                voc_scale = gr.Dropdown(
                    label="Scale",
                    choices=[("From the key", ""), *[(name, name) for name in scales.SCALES]],
                    value="",
                )
                voc_notes = gr.Textbox(
                    label="Or your own notes",
                    placeholder="F G# A# C D#",
                    info="Overrides the scale.",
                )
                voc_engine = gr.Radio(
                    label="Engine", choices=list(tuning.ENGINES), value=start.engine
                )
            with gr.Row():
                voc_retune = gr.Slider(
                    0, 400, value=start.retune_ms, step=5, label="Retune speed (ms; 0 is instant)"
                )
                voc_humanize = gr.Slider(
                    0,
                    100,
                    value=round(start.humanize * 100),
                    step=5,
                    label="Humanize (%; 100 keeps vibrato and slides)",
                )
            with gr.Row():
                voc_formants = gr.Checkbox(label="Keep the formants", value=True)
                voc_shift = gr.Slider(
                    -12, 12, value=0, step=0.5, label="Formant shift (semitones; WORLD only)"
                )
                voc_hold = gr.Checkbox(label="Note hold: one step per slide", value=False)
            gr.Markdown(
                "**Notes.** The thin line is where a note was sung, the bar where it will go. "
                "Click a note to leave it alone, drag it to send it elsewhere, click an "
                "edited note to undo."
            )
            script = (
                resources.files("masterbench.ui").joinpath("pitch_editor.js").read_text("utf-8")
            )
            voc_editor = gr.HTML(
                value=editor_value(None, None),
                html_template=EDITOR_TEMPLATE,
                css_template=EDITOR_CSS,
                js_on_load=script,
                server_functions=[save_edits],
                elem_id="mb-pitch-editor",
            )
            with gr.Row():
                voc_preset_name = gr.Textbox(label="Save these settings as a preset", scale=2)
                voc_preset_save = gr.Button("Save preset", scale=1)
            voc_preset_md = gr.Markdown()
            gr.Markdown(
                "**Compare engines.** Tunes this vocal with both engines, Natural and Hard-tune, "
                "and loads them all in the player above. Add the same vocal tuned in Waves or "
                "Flex Pitch, then use the player's blind test to choose without knowing which "
                "is which."
            )
            with gr.Row():
                voc_ext = gr.File(
                    label="The same vocal tuned elsewhere (optional)",
                    file_types=[".wav", ".aif", ".aiff", ".flac"],
                    type="filepath",
                )
                voc_ext_label = gr.Textbox(label="What it is", placeholder="Waves Tune")
                voc_bench = gr.Button("Compare engines")
            voc_bench_md = gr.Markdown()

    def resolve(state, file, picked, title, is_mix, progress) -> VocalSource:
        """The vocal the controls point at; brought into the project the first time."""
        wanted = [file, picked, (title or "").strip(), bool(is_mix)]
        if state and state["wanted"] == wanted:
            source = workflow.load_source(paths, state["id"])
            assert source is not None
            return source
        name = (title or "").strip() or None
        if file and is_mix:
            progress(0.1, desc="Analyzing the mix for its key")
            pipeline.analyze(Path(file), paths, settings, None, name)
            progress(0.3, desc="Separating the vocal (this takes a while)")
            return workflow.separate_vocal(paths, Path(file), name)
        if file:
            return workflow.open_vocal(paths, Path(file), name)
        if picked and picked.startswith("stems:"):
            return workflow.from_stems(paths, picked.removeprefix("stems:"))
        if picked:
            source = workflow.load_source(paths, picked)
            assert source is not None
            return source
        raise VocalError("Choose a vocal file first.")

    def remember(source: VocalSource, file, picked, title, is_mix) -> dict[str, Any]:
        return {"id": source.id, "wanted": [file, picked, (title or "").strip(), bool(is_mix)]}

    def custom(notes: str | None) -> tuple[str, ...]:
        return tuple((notes or "").replace(",", " ").split())

    def on_check(state, file, picked, title, is_mix, key, scale, notes, progress=gr.Progress()):
        try:
            source = resolve(state, file, picked, title, is_mix, progress)
            progress(0.7, desc="Tracking pitch")
            same = state and state["id"] == source.id
            found = workflow.report(
                source, paths, key if same else None, scale or None, custom(notes)
            )
        except ERRORS as exc:
            raise gr.Error(str(exc)) from exc
        return (
            remember(source, file, picked, title, is_mix),
            gr.Dropdown(value=found.key if found.key in KEYS else None),
            key_note(found),
            summary.pitch_markdown(found),
            editor_value(source, found),
            player_config(source),
            gr.Dropdown(choices=source_choices(paths)),
        )

    def on_rescale(state, key, scale, notes):
        """The key or scale was changed by hand: measure the same vocal against it."""
        if not state or not key:
            return gr.skip(), gr.skip(), gr.skip()
        try:
            source = workflow.load_source(paths, state["id"])
            assert source is not None
            found = workflow.report(source, paths, key, scale or None, custom(notes))
        except ERRORS as exc:
            raise gr.Error(str(exc)) from exc
        return key_note(found), summary.pitch_markdown(found), editor_value(source, found)

    def chosen_settings(key, scale, notes, amount, retune, humanize, engine, formants, shift, hold):
        return tuning.TuningSettings(
            key=key,
            scale=scale or None,
            custom_notes=custom(notes),
            retune_ms=float(retune),
            humanize=humanize / 100,
            amount=amount / 100,
            preserve_formants=bool(formants),
            formant_shift=float(shift),
            note_hold=bool(hold),
            engine=engine,
        )

    def on_tune(state, file, picked, title, is_mix, key, preset, *controls, progress=gr.Progress()):
        try:
            source = resolve(state, file, picked, title, is_mix, progress)
            same = state and state["id"] == source.id
            found = workflow.stored_report(source) if same else None
            if found is None or not key:
                progress(0.4, desc="Tracking pitch")
                found = workflow.report(source, paths, key if same else None)
            progress(0.6, desc="Tuning")
            wanted = chosen_settings(key if same and key else found.key, *controls)
            version = workflow.tune(source, wanted, label=preset.capitalize())
        except ERRORS as exc:
            raise gr.Error(str(exc)) from exc
        files = [str(source.folder / version.file)]
        selected = f"tuned_{version.id}"
        if version.preview_file:
            files.append(str(source.folder / version.preview_file))
            selected = f"preview_{version.id}"
        return (
            remember(source, file, picked, title, is_mix),
            gr.Dropdown(value=found.key if found.key in KEYS else None),
            key_note(found),
            "```\n" + summary.tuned_text(version, source.folder) + "\n```",
            player_config(source, selected),
            files,
            summary.pitch_markdown(found),
            editor_value(source, found),
            gr.Dropdown(choices=source_choices(paths)),
        )

    def on_preset(name: str):
        try:
            values = presets.get(paths, name).settings("C major")
        except TuningError as exc:
            raise gr.Error(str(exc)) from exc
        return (
            round(values.amount * 100),
            values.retune_ms,
            round(values.humanize * 100),
            values.engine,
            values.preserve_formants,
            values.formant_shift,
            values.note_hold,
        )

    def on_preset_save(name: str, *controls):
        try:
            wanted = chosen_settings("C major", "", "", *controls)
            saved = presets.save(paths, presets.from_settings(name.strip(), wanted, "Saved here."))
        except ERRORS as exc:
            raise gr.Error(str(exc)) from exc
        return f"Saved `{saved}`.", gr.Radio(choices=presets.names(paths), value=name.strip())

    def on_benchmark(state, key, scale, external, label, progress=gr.Progress()):
        if not state:
            raise gr.Error("Check a vocal's pitch first.")
        try:
            source = workflow.load_source(paths, state["id"])
            assert source is not None
            if external:
                if not (label or "").strip():
                    raise VocalError("Say what the added file is, e.g. 'Waves Tune'.")
                workflow.add_external(source, Path(external), label.strip())
            progress(0.2, desc="Tuning with both engines")
            found = workflow.stored_report(source)
            rows = workflow.benchmark(
                source, paths, key or (found.key if found else "C major"), scale or None
            )
        except ERRORS as exc:
            raise gr.Error(str(exc)) from exc
        return "```\n" + summary.benchmark_text(rows) + "\n```", player_config(source, "original")

    which = [current, voc_file, voc_pick, voc_title, voc_mix]
    scale_controls = [voc_scale, voc_notes]
    shape = [voc_amount, voc_retune, voc_humanize, voc_engine, voc_formants, voc_shift, voc_hold]
    voc_check.click(
        on_check,
        [*which, voc_key, *scale_controls],
        [current, voc_key, voc_key_md, voc_report, voc_editor, voc_player, voc_pick],
    )
    voc_tune.click(
        on_tune,
        [*which, voc_key, voc_preset, *scale_controls, *shape],
        [
            current,
            voc_key,
            voc_key_md,
            voc_result,
            voc_player,
            voc_files,
            voc_report,
            voc_editor,
            voc_pick,
        ],
    )
    for control in (voc_key, voc_scale):
        control.input(
            on_rescale, [current, voc_key, *scale_controls], [voc_key_md, voc_report, voc_editor]
        )
    voc_notes.submit(
        on_rescale, [current, voc_key, *scale_controls], [voc_key_md, voc_report, voc_editor]
    )
    voc_preset.input(on_preset, voc_preset, shape)
    voc_preset_save.click(on_preset_save, [voc_preset_name, *shape], [voc_preset_md, voc_preset])
    voc_bench.click(
        on_benchmark,
        [current, voc_key, voc_scale, voc_ext, voc_ext_label],
        [voc_bench_md, voc_player],
    )
