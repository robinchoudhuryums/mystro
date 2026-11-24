"""Gradio app. Thin by design: every handler calls an engine function and formats the result."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from masterbench import audio_io, batch, index, progress, projects, summary
from masterbench.analysis.compare import BAND_LABELS
from masterbench.config import Paths, Settings
from masterbench.master import chain, pipeline, presets
from masterbench.master.matchering_engine import MasteringError
from masterbench.master.pipeline import MasterOptions, RunResult
from masterbench.notes import mix_notes
from masterbench.preview import streaming
from masterbench.preview.streaming import StreamingPreview
from masterbench.references import library
from masterbench.stems import separate as stem_separation
from masterbench.stems import workflow as stem_workflow
from masterbench.stems.separate import StemSet
from masterbench.stems.workflow import Diagnosis
from masterbench.ui import charts, player

UPLOAD_TYPES = sorted(audio_io.LOSSLESS_EXTENSIONS)
STEM_MODELS = tuple(stem_separation.MODELS.values())
REFERENCE_TYPES = sorted(audio_io.AUDIO_EXTENSIONS)
ENGINE_ERRORS = (
    MasteringError,
    library.ReferenceError,
    audio_io.AudioError,
    streaming.PreviewError,
    chain.ChainError,
    stem_separation.StemError,
    mix_notes.NotesError,
    LookupError,
)
NO_REFERENCES = (
    "No references yet. Add one in the **Library** tab, "
    "or run `masterbench ref add <file> --genre <genre>`."
)


def genre_choices(paths: Paths, settings: Settings) -> tuple[list[str], str | None]:
    genres = library.genres(paths)
    default = settings.default_genre if settings.default_genre in genres else None
    return genres, default or (genres[0] if genres else None)


def reference_choices(paths: Paths, genre: str | None) -> tuple[list[tuple[str, str]], str | None]:
    """Dropdown (label, spec) pairs for a genre, with the genre default first and selected."""
    if not genre:
        return [], None
    refs = sorted(library.list_references(paths, genre), key=lambda r: (not r.is_default, r.name))
    choices = [(f"{r.name} (default)" if r.is_default else r.name, r.spec) for r in refs]
    return choices, choices[0][1] if choices else None


def preset_choices(paths: Paths) -> list[tuple[str, str]]:
    return [("None", ""), *[(name, name) for name in presets.names(paths)]]


def run_choices(paths: Paths) -> list[tuple[str, str]]:
    """Recent runs, newest first: Inbox analyses and masters."""
    with index.open_index(paths) as conn:
        rows = index.recent_runs(conn)
    kinds = {"analysis_only": "analyzed", "master": "mastered"}
    return [
        (f"{r['title']} · {kinds.get(r['kind'], r['kind'])} {r['created_at'][:16]}", r["id"])
        for r in rows
    ]


def library_rows(paths: Paths) -> list[list[str]]:
    return [
        [
            ref.genre,
            ref.name,
            "yes" if ref.is_default else "",
            f"{ref.report.loudness.lufs_i:.1f}",
            f"{ref.report.loudness.true_peak_dbtp:+.1f}",
            ref.notes,
        ]
        for ref in library.list_references(paths)
    ]


def run_view(result: RunResult) -> tuple:
    """Values for (summary, flags, mix player, master player, reference player, downloads)."""
    outputs = [str(result.output(name)) for name in result.manifest.outputs]
    reference = result.reports.get("reference")
    return (
        summary.as_markdown(result),
        summary.flags_markdown(result),
        str(result.mix_path),
        outputs[0] if outputs else None,
        reference.source_path if reference else None,
        outputs or None,
    )


def other_masters(paths: Paths, run: RunResult) -> list[tuple[str, str]]:
    """Other master runs of the same track, as (label, run id), newest first."""
    with index.open_index(paths) as conn:
        rows = index.recent_runs(conn, limit=200, kind="master")
    return [
        (f"{r['created_at'][:16].replace('T', ' ')} ({r['id'][-6:]})", r["id"])
        for r in rows
        if r["slug"] == run.manifest.track.slug and r["id"] != run.manifest.id
    ]


def streaming_markdown(preview: StreamingPreview | None) -> str:
    if preview is None:
        return "Streaming previews have not been rendered for this run."
    lines = [
        f"Master true peak: {preview.source_true_peak_dbtp:+.2f} dBTP.",
        "",
        "| Codec | Encoder | True peak after decode | Change | LUFS-I |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {c.label} | {c.encoder} | {c.true_peak_dbtp:+.2f} dBTP | {c.peak_change_db:+.2f} dB"
        f" | {c.lufs_i:.1f} |"
        for c in preview.codecs
    ]
    lines += ["", *[f"- Skipped {reason}" for reason in preview.skipped]]
    lines += [f"- {summary.flag_line(flag)}" for flag in preview.flags]
    lines += [f"- {gain.message}" for gain in preview.platforms]
    return "\n".join(lines)


def stereo_markdown(result: RunResult) -> str:
    rows = [(role, r.stereo) for role, r in result.reports.items() if r.stereo is not None]
    if not rows:
        return ""
    bands = list(BAND_LABELS)
    header = ["", "Correlation", "Side vs mid", "Below 120 Hz", *[BAND_LABELS[b] for b in bands]]
    lines = ["**Stereo**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for role, stereo in rows:
        cells = [
            role.capitalize(),
            f"{stereo.correlation:+.2f}",
            f"{stereo.side_mid_db:.1f} dB",
            f"{stereo.low_mono_corr:+.2f}",
            *[f"{stereo.band_correlation[b]:+.2f}" for b in bands],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def analysis_view(result: RunResult, settings: Settings) -> tuple:
    """Values for (spectrum plot, band plot, loudness plot, stereo table, flags)."""
    return (
        charts.spectrum(result.reports),
        charts.band_deltas(result.reports, settings.thresholds.band_delta_db),
        charts.loudness_over_time(result.reports),
        stereo_markdown(result),
        summary.flags_markdown(result),
    )


def progress_view(paths: Paths) -> tuple:
    """Values for (band trend plot, loudness trend plot, flag plot, blind-test rows)."""
    with index.open_index(paths) as conn:
        points = progress.track_points(conn)
        tests = progress.blind_results(conn)
    rows = [
        [t.created_at[:16].replace("T", " "), t.title, t.winner, " > ".join(t.ranking)]
        for t in tests
    ]
    return (
        charts.band_trends(points),
        charts.loudness_trends(points),
        charts.flag_frequency(progress.flag_counts(points), len(points)),
        rows,
    )


def gain_rows(stem_set: StemSet, diagnosis: Diagnosis) -> list[list]:
    """Rows for the rebalance table: stem, role, level relative to the mix, gain to apply."""
    return [
        [name, stem_set.roles[name], diagnosis.balance.levels_db[name], 0.0]
        for name in stem_set.stems
    ]


def build(paths: Paths, settings: Settings) -> gr.Blocks:
    paths.ensure()
    genres, genre_default = genre_choices(paths, settings)
    refs, ref_default = reference_choices(paths, genre_default)

    with gr.Blocks(title="MasterBench") as app:
        gr.Markdown("# MasterBench")
        with gr.Tab("Master"), gr.Row():
            with gr.Column(scale=1):
                mix = gr.File(label="Mix", file_types=UPLOAD_TYPES, type="filepath")
                previous = gr.Dropdown(
                    label="Or open a recent run or Inbox file",
                    choices=run_choices(paths),
                    value=None,
                )
                genre = gr.Dropdown(label="Genre", choices=genres, value=genre_default)
                reference = gr.Dropdown(label="Reference", choices=refs, value=ref_default)
                with gr.Accordion("Targets", open=False):
                    target = gr.Number(label="Loudness (LUFS)", value=settings.targets.lufs_i)
                    ceiling = gr.Number(
                        label="True-peak ceiling (dBTP)", value=settings.targets.true_peak_dbtp
                    )
                with gr.Accordion("Chains", open=False):
                    pre_chain = gr.Dropdown(
                        label="Before matching", choices=preset_choices(paths), value=""
                    )
                    post_chain = gr.Dropdown(
                        label="After matching", choices=preset_choices(paths), value=""
                    )
                run = gr.Button("Master", variant="primary")
                analyze = gr.Button("Analyze only")
            with gr.Column(scale=2):
                result_md = gr.Markdown("" if refs else NO_REFERENCES)
                flags_md = gr.Markdown()
                mix_player = gr.Audio(label="Mix", type="filepath", interactive=False)
                master_player = gr.Audio(label="Master", type="filepath", interactive=False)
                ref_player = gr.Audio(label="Reference", type="filepath", interactive=False)
                downloads = gr.File(label="Files", file_count="multiple", interactive=False)
        with gr.Tab("Compare") as compare_tab:
            with gr.Row():
                cmp_run = gr.Dropdown(label="Run", choices=run_choices(paths), value=None, scale=2)
                cmp_extra = gr.Dropdown(
                    label="Also load masters from other runs of this track",
                    choices=[],
                    multiselect=True,
                    scale=2,
                )
                cmp_render = gr.Button("Render streaming previews", scale=1)
            cmp_stems = gr.Checkbox(label="Add stem solos (when this track has stems)")
            ab_player = player.component(paths)
            gr.Markdown(
                "Loudness matching only changes playback level; exported files are untouched."
            )
            stream_md = gr.Markdown()
        with gr.Tab("Analyze") as analyze_tab:
            an_run = gr.Dropdown(label="Run", choices=run_choices(paths), value=None)
            spectrum_plot = gr.Plot()
            band_plot = gr.Plot()
            loudness_plot = gr.Plot()
            stereo_md = gr.Markdown()
            an_flags = gr.Markdown()
            with gr.Accordion("AI mix notes (optional; sends numbers, never audio)", open=False):
                with gr.Row():
                    notes_intent = gr.Textbox(
                        label="What were you going for? (optional)", scale=3, lines=2
                    )
                    notes_model = gr.Dropdown(
                        label="Model", choices=list(mix_notes.PRICES), value=settings.notes_model
                    )
                with gr.Row():
                    notes_preview = gr.Button("Show what gets sent")
                    notes_send = gr.Button("Get AI mix notes", variant="primary")
                notes_quote = gr.Markdown()
                notes_payload = gr.Code(label="Exactly what is sent", language="json", lines=12)
                notes_md = gr.Markdown()
        with gr.Tab("Stems") as stems_tab:
            with gr.Row():
                st_run = gr.Dropdown(label="Run", choices=run_choices(paths), value=None, scale=2)
                st_model = gr.Dropdown(
                    label="Separation model",
                    choices=[(f"{m.key}: {m.description}", m.key) for m in STEM_MODELS],
                    value=stem_separation.DEFAULT_MODEL,
                    scale=2,
                )
                st_compare = gr.Checkbox(label="Compare with the reference's stems", value=True)
                st_separate = gr.Button("Separate", variant="primary", scale=1)
            with gr.Accordion("Import stems exported from Logic", open=False):
                gr.Markdown(
                    "Export all tracks as audio files from Logic into one folder, same start "
                    "and length. Roles are guessed from the file names (vocal, drum, 808, bass)."
                )
                with gr.Row():
                    st_folder = gr.Textbox(label="Folder with the stems", scale=3)
                    st_title = gr.Textbox(label="Track title", scale=2)
                    st_genre = gr.Dropdown(
                        label="Genre", choices=genres, value=genre_default, allow_custom_value=True
                    )
                    st_import = gr.Button("Import", scale=1)
            st_set = gr.Dropdown(label="Stem set", choices=[], value=None)
            st_report = gr.Markdown()
            st_gains = gr.Dataframe(
                headers=["Stem", "Role", "Level vs mix (dB)", "Gain (dB)"],
                datatype=["str", "str", "number", "number"],
                type="array",
                label="Rebalance: edit the Gain column (±6 dB)",
                static_columns=[0, 1, 2],
                interactive=True,
            )
            st_master = gr.Button("Rebalance and master")
            st_result = gr.Markdown()
        with gr.Tab("Batch"):
            gr.Markdown(
                "Master a whole EP the same way. Uses the reference, targets and chains chosen "
                "on the Master tab, then checks that the tracks sit together."
            )
            batch_files = gr.File(
                label="Mixes", file_count="multiple", file_types=UPLOAD_TYPES, type="filepath"
            )
            batch_name = gr.Textbox(label="Name of the set (optional)")
            batch_run = gr.Button("Master the set", variant="primary")
            batch_md = gr.Markdown()
        with gr.Tab("Progress") as progress_tab:
            trend_plot = gr.Plot()
            loudness_trend_plot = gr.Plot()
            flag_plot = gr.Plot()
            blind_table = gr.Dataframe(
                headers=["When", "Track", "Preferred", "Ranking"],
                label="Blind tests",
                type="array",
                interactive=False,
            )
        with gr.Tab("Library"):
            table = gr.Dataframe(
                value=library_rows(paths),
                headers=["Genre", "Name", "Default", "LUFS-I", "True peak", "Notes"],
                type="array",
                interactive=False,
            )
            with gr.Row():
                new_file = gr.File(label="Reference", file_types=REFERENCE_TYPES, type="filepath")
                with gr.Column():
                    new_genre = gr.Dropdown(
                        label="Genre", choices=genres, value=genre_default, allow_custom_value=True
                    )
                    new_notes = gr.Textbox(label="Notes")
                    new_default = gr.Checkbox(label="Make it the genre default")
                    add = gr.Button("Add reference", variant="primary")
            library_status = gr.Markdown()
            gr.Markdown("### Chain presets")
            with gr.Row():
                with gr.Column(scale=2):
                    preset_pick = gr.Dropdown(
                        label="Preset", choices=presets.names(paths), value=None
                    )
                    preset_text = gr.Code(label="TOML", language=None, lines=14)
                    with gr.Row():
                        preset_name = gr.Textbox(label="Save as", scale=2)
                        preset_save = gr.Button("Save preset", variant="primary", scale=1)
                    preset_status = gr.Markdown(
                        "Modules: " + ", ".join(chain.BUILTIN) + ", plugin. "
                        "The final true-peak limiter always runs after the post-chain."
                    )
                with gr.Column(scale=2):
                    plugin_scan = gr.Button("Scan plugins")
                    plugin_table = gr.Dataframe(
                        headers=["Format", "Path", "Plugins inside"],
                        type="array",
                        interactive=False,
                    )
                    plugin_path = gr.Textbox(label="Plugin path")
                    plugin_inner = gr.Textbox(label="Plugin name inside a shell (optional)")
                    plugin_add = gr.Button("Append this plugin to the preset")

        view = [result_md, flags_md, mix_player, master_player, ref_player, downloads]

        def source_file(uploaded: str | None, run_id: str | None) -> Path:
            if uploaded:
                return Path(uploaded)
            if run_id:
                return pipeline.load_run(paths, run_id).mix_path
            raise gr.Error("Choose a mix first.")

        def on_genre(selected: str | None):
            choices, value = reference_choices(paths, selected)
            return gr.Dropdown(choices=choices, value=value)

        def on_master(
            uploaded, run_id, selected_genre, spec, lufs, peak, pre, post, progress=gr.Progress()
        ):
            if not spec:
                raise gr.Error("Add a reference in the Library tab first.")
            options = MasterOptions(
                spec,
                lufs,
                peak,
                genre=selected_genre,
                pre_chain=pre or None,
                post_chain=post or None,
            )
            try:
                result = pipeline.master(
                    source_file(uploaded, run_id),
                    options,
                    paths,
                    settings,
                    progress=lambda stage, fraction: progress(fraction, desc=stage),
                )
            except ENGINE_ERRORS as exc:
                raise gr.Error(str(exc)) from exc
            choices = run_choices(paths)
            run_id = result.manifest.id
            return (
                *run_view(result),
                gr.Dropdown(choices=choices),
                gr.Dropdown(choices=choices, value=run_id),
                gr.Dropdown(choices=choices, value=run_id),
            )

        def on_analyze(uploaded, run_id, selected_genre, progress=gr.Progress()):
            try:
                result = pipeline.analyze(
                    source_file(uploaded, run_id),
                    paths,
                    settings,
                    genre=selected_genre,
                    progress=lambda stage, fraction: progress(fraction, desc=stage),
                )
            except ENGINE_ERRORS as exc:
                raise gr.Error(str(exc)) from exc
            choices = run_choices(paths)
            run_id = result.manifest.id
            return (
                *run_view(result),
                gr.Dropdown(choices=choices),
                gr.Dropdown(choices=choices, value=run_id),
                gr.Dropdown(choices=choices, value=run_id),
            )

        def on_open(run_id: str | None):
            if not run_id:
                return (gr.skip(),) * len(view)
            return run_view(pipeline.load_run(paths, run_id))

        def on_load(request: gr.Request):
            """Deep link from a notification: /?run=<id> opens that run on every tab."""
            run_id = request.query_params.get("run")
            choices = run_choices(paths)
            if not run_id or run_id not in {value for _, value in choices}:
                same = gr.Dropdown(choices=choices)
                return (*(gr.skip(),) * len(view), same, same, same)
            chosen = gr.Dropdown(choices=choices, value=run_id)
            return (*run_view(pipeline.load_run(paths, run_id)), chosen, chosen, chosen)

        def on_add(file, selected_genre, notes, make_default, current_genre):
            if not file or not selected_genre:
                raise gr.Error("Choose a file and a genre.")
            try:
                ref = library.add(paths, Path(file), selected_genre, settings, notes, make_default)
            except ENGINE_ERRORS as exc:
                raise gr.Error(str(exc)) from exc
            with index.open_index(paths) as conn:
                index.sync_references(conn, paths)
            all_genres = library.genres(paths)
            shown = current_genre or ref.genre
            choices, value = reference_choices(paths, shown)
            return (
                f"Added **{ref.name}** to {ref.genre}: {ref.report.headline()}.",
                library_rows(paths),
                gr.Dropdown(choices=all_genres, value=shown),
                gr.Dropdown(choices=choices, value=value),
                gr.Dropdown(choices=all_genres, value=ref.genre),
            )

        def load(run_id: str | None) -> RunResult | None:
            return pipeline.load_run(paths, run_id) if run_id else None

        def compare_config(run_id, extra_ids, with_stems) -> tuple[str, StreamingPreview | None]:
            result = load(run_id)
            if result is None:
                return player.config(None), None
            preview = streaming.load(result) if result.manifest.outputs else None
            extras = [pipeline.load_run(paths, i) for i in extra_ids or []]
            sets = stem_workflow.sets_for_run(result) if with_stems else []
            return player.config(result, preview, extras, sets[0] if sets else None), preview

        def on_compare_run(run_id: str | None, with_stems: bool):
            result = load(run_id)
            if result is None:
                return player.config(None), gr.Dropdown(choices=[], value=[]), ""
            config_json, preview = compare_config(run_id, [], with_stems)
            extras = gr.Dropdown(choices=other_masters(paths, result), value=[])
            text = streaming_markdown(preview) if result.manifest.outputs else ""
            return config_json, extras, text

        def on_compare_extra(run_id: str | None, extra_ids: list[str] | None, with_stems: bool):
            return compare_config(run_id, extra_ids, with_stems)[0]

        def on_render(run_id: str | None, extra_ids: list[str] | None, with_stems: bool):
            result = load(run_id)
            if result is None:
                raise gr.Error("Choose a run first.")
            try:
                preview = streaming.render(result, settings)
            except ENGINE_ERRORS as exc:
                raise gr.Error(str(exc)) from exc
            return compare_config(run_id, extra_ids, with_stems)[0], streaming_markdown(preview)

        def on_analysis_run(run_id: str | None):
            result = load(run_id)
            if result is None:
                return (gr.skip(),) * 5
            return analysis_view(result, settings)

        def refresh_runs(current: str | None):
            return gr.Dropdown(choices=run_choices(paths), value=current)

        def on_preset_pick(name: str | None):
            if not name:
                return gr.skip(), gr.skip()
            return presets.text(paths, name), name

        def on_preset_save(name: str, toml_text: str):
            try:
                saved = presets.save_text(paths, (name or "").strip(), toml_text or "")
            except chain.ChainError as exc:
                raise gr.Error(str(exc)) from exc
            listed = preset_choices(paths)
            steps = " > ".join(module.type for module in saved.modules) or "empty"
            return (
                f"Saved **{saved.name}**: {steps}.",
                gr.Dropdown(choices=presets.names(paths), value=saved.name),
                gr.Dropdown(choices=listed),
                gr.Dropdown(choices=listed),
            )

        def on_plugin_scan():
            return [[p.format, str(p.path), ", ".join(p.names)] for p in chain.scan_plugins()]

        def on_plugin_add(toml_text: str, path: str, inner: str):
            if not path:
                raise gr.Error("Enter a plugin path (copy one from the scan).")
            try:
                current = presets.from_toml(toml_text or "", "draft")
                module = chain.capture_plugin(Path(path.strip()), inner.strip() or None)
            except chain.ChainError as exc:
                raise gr.Error(str(exc)) from exc
            draft = chain.ChainPreset("draft", current.description, (*current.modules, module))
            return presets.to_toml(draft)

        def stems_view(stem_set, diagnosis):
            return (
                summary.stems_markdown(diagnosis),
                gain_rows(stem_set, diagnosis),
                "",
            )

        def set_choices(result: RunResult):
            return [
                (f"{s.name} ({'separated' if s.separated else 'from Logic'})", s.name)
                for s in stem_workflow.sets_for_run(result)
            ]

        def run_reference(result: RunResult, wanted: bool):
            genre_name = result.manifest.track.genre
            if not wanted or not genre_name:
                return None
            try:
                return library.get(paths, genre_name, "default")
            except library.ReferenceError:
                return None

        def on_stems_run(run_id: str | None):
            result = load(run_id)
            if result is None:
                return gr.Dropdown(choices=[], value=None)
            choices = set_choices(result)
            return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

        def on_separate(run_id, model_key, compare_wanted, progress=gr.Progress()):
            result = load(run_id)
            if result is None:
                raise gr.Error("Choose a run first.")
            try:
                progress(0.1, desc="Separating (the first use of a model downloads it)")
                project = projects.open_project(paths, result.mix_path, result.manifest.track.title)
                stem_set = stem_workflow.separate_mix(project, model_key)
                progress(0.7, desc="Measuring the stem balance")
                diagnosis = stem_workflow.diagnose(
                    stem_set, result.mix_path, settings, run_reference(result, compare_wanted)
                )
            except ENGINE_ERRORS as exc:
                raise gr.Error(str(exc)) from exc
            picker = gr.Dropdown(choices=set_choices(result), value=stem_set.name)
            return (picker, *stems_view(stem_set, diagnosis))

        def on_stem_set(run_id, set_name, compare_wanted):
            result = load(run_id)
            if result is None or not set_name:
                return (gr.skip(),) * 3
            try:
                stem_set = stem_workflow.find_set(paths, result.manifest.track.slug, set_name)
                diagnosis = stem_workflow.diagnose(
                    stem_set, result.mix_path, settings, run_reference(result, compare_wanted)
                )
            except ENGINE_ERRORS as exc:
                raise gr.Error(str(exc)) from exc
            return stems_view(stem_set, diagnosis)

        def on_stems_import(folder, title, genre_name):
            if not folder or not title:
                raise gr.Error("Enter the folder and a track title.")
            try:
                project, stem_set = stem_workflow.import_logic_stems(
                    paths, Path(folder.strip()).expanduser(), title.strip(), genre_name or None
                )
                result = pipeline.analyze(
                    project.input_path, paths, settings, project.track.genre, project.track.title
                )
                diagnosis = stem_workflow.diagnose(stem_set, result.mix_path, settings)
            except ENGINE_ERRORS as exc:
                raise gr.Error(str(exc)) from exc
            except OSError as exc:
                raise gr.Error(f"Cannot read that folder: {exc}") from exc
            run_picker = gr.Dropdown(choices=run_choices(paths), value=result.manifest.id)
            set_picker = gr.Dropdown(choices=set_choices(result), value=stem_set.name)
            return (run_picker, set_picker, *stems_view(stem_set, diagnosis))

        def on_rebalance(run_id, set_name, rows, spec, lufs, peak, progress=gr.Progress()):
            result = load(run_id)
            if result is None or not set_name:
                raise gr.Error("Choose a run and a stem set first.")
            if not spec:
                raise gr.Error("Add a reference in the Library tab first.")
            gains = {row[0]: float(row[3] or 0) for row in rows if float(row[3] or 0) != 0}
            track = result.manifest.track
            options = MasterOptions(spec, lufs, peak, genre=track.genre, title=track.title)
            try:
                stem_set = stem_workflow.find_set(paths, track.slug, set_name)
                rebalanced, mastered = stem_workflow.rebalance_and_master(
                    paths,
                    settings,
                    stem_set,
                    gains,
                    options,
                    progress=lambda stage, fraction: progress(fraction, desc=stage),
                )
            except ENGINE_ERRORS as exc:
                raise gr.Error(str(exc)) from exc
            notes = [summary.as_markdown(mastered), "", summary.flags_markdown(mastered), ""]
            if rebalanced.trim_db < 0:
                trim = -rebalanced.trim_db
                notes.append(f"The new balance was turned down {trim:.1f} dB to keep headroom.")
            if stem_set.separated:
                notes.append("Built from separated stems: listen for artifacts before trusting it.")
            notes.append("Open the new run in Compare to hear it against the original master.")
            return "\n".join(notes)

        def notes_quote_text(quote: mix_notes.Estimate) -> str:
            text = (
                f"{quote.model}: about {quote.input_tokens} tokens in, roughly "
                f"**${quote.cost_usd:.3f}**. Spent this month: ${quote.spent_this_month_usd:.2f} "
                f"of ${quote.cap_usd:.2f}. API key: {mix_notes.key_status()}."
            )
            return text + (" **This would pass the cap.**" if quote.over_cap else "")

        def on_notes_preview(run_id: str | None, intent: str, model: str):
            result = load(run_id)
            if result is None:
                raise gr.Error("Choose a run first.")
            payload = mix_notes.build_payload(result, intent or "")
            try:
                quote = mix_notes.estimate(payload, model, paths, settings)
            except ENGINE_ERRORS as exc:
                raise gr.Error(str(exc)) from exc
            return notes_quote_text(quote), mix_notes.payload_json(payload)

        def on_notes_send(run_id: str | None, intent: str, model: str):
            result = load(run_id)
            if result is None:
                raise gr.Error("Choose a run first.")
            try:
                written = mix_notes.generate(result, paths, settings, intent or "", model)
            except ENGINE_ERRORS as exc:
                raise gr.Error(str(exc)) from exc
            with index.open_index(paths) as conn:
                index.sync_notes(conn, paths)
            payload = mix_notes.build_payload(result, intent or "")
            return mix_notes.render(written), mix_notes.payload_json(payload)

        def on_notes_run(run_id: str | None):
            result = load(run_id)
            stored = mix_notes.load(result) if result else None
            return mix_notes.render(stored) if stored else ""

        def on_batch(
            files, name, selected_genre, spec, lufs, peak, pre, post, progress=gr.Progress()
        ):
            if not files:
                raise gr.Error("Add at least two mixes.")
            if not spec:
                raise gr.Error("Add a reference in the Library tab first.")
            options = MasterOptions(
                spec,
                lufs,
                peak,
                genre=selected_genre,
                pre_chain=pre or None,
                post_chain=post or None,
            )
            done = batch.run_batch(
                [Path(f) for f in files],
                options,
                paths,
                settings,
                (name or "").strip(),
                progress=lambda stage, fraction: progress(fraction, desc=stage),
            )
            return summary.batch_markdown(done)

        genre.change(on_genre, genre, reference)
        notes_preview.click(
            on_notes_preview, [an_run, notes_intent, notes_model], [notes_quote, notes_payload]
        )
        notes_send.click(
            on_notes_send, [an_run, notes_intent, notes_model], [notes_md, notes_payload]
        )
        an_run.change(on_notes_run, an_run, notes_md)
        batch_inputs = [
            batch_files,
            batch_name,
            genre,
            reference,
            target,
            ceiling,
            pre_chain,
            post_chain,
        ]
        batch_run.click(on_batch, batch_inputs, batch_md)
        st_view = [st_report, st_gains, st_result]
        st_run.change(on_stems_run, st_run, st_set)
        st_set.change(on_stem_set, [st_run, st_set, st_compare], st_view)
        st_separate.click(on_separate, [st_run, st_model, st_compare], [st_set, *st_view])
        st_import.click(
            on_stems_import, [st_folder, st_title, st_genre], [st_run, st_set, *st_view]
        )
        st_master.click(
            on_rebalance, [st_run, st_set, st_gains, reference, target, ceiling], st_result
        )
        stems_tab.select(refresh_runs, st_run, st_run)
        preset_pick.input(on_preset_pick, preset_pick, [preset_text, preset_name])
        preset_save.click(
            on_preset_save,
            [preset_name, preset_text],
            [preset_status, preset_pick, pre_chain, post_chain],
        )
        plugin_scan.click(on_plugin_scan, None, plugin_table)
        plugin_add.click(on_plugin_add, [preset_text, plugin_path, plugin_inner], preset_text)
        after_run = [*view, previous, cmp_run, an_run]
        master_inputs = [mix, previous, genre, reference, target, ceiling, pre_chain, post_chain]
        run.click(on_master, master_inputs, after_run)
        analyze.click(on_analyze, [mix, previous, genre], after_run)
        cmp_run.change(on_compare_run, [cmp_run, cmp_stems], [ab_player, cmp_extra, stream_md])
        cmp_extra.input(on_compare_extra, [cmp_run, cmp_extra, cmp_stems], ab_player)
        cmp_stems.input(on_compare_extra, [cmp_run, cmp_extra, cmp_stems], ab_player)
        cmp_render.click(on_render, [cmp_run, cmp_extra, cmp_stems], [ab_player, stream_md])
        an_run.change(
            on_analysis_run, an_run, [spectrum_plot, band_plot, loudness_plot, stereo_md, an_flags]
        )
        compare_tab.select(refresh_runs, cmp_run, cmp_run)
        analyze_tab.select(refresh_runs, an_run, an_run)
        progress_tab.select(
            lambda: progress_view(paths),
            None,
            [trend_plot, loudness_trend_plot, flag_plot, blind_table],
        )
        previous.input(on_open, previous, view)
        app.load(on_load, None, after_run)
        add.click(
            on_add,
            [new_file, new_genre, new_notes, new_default, genre],
            [library_status, table, genre, reference, new_genre],
        )
    return app


def launch(paths: Paths, settings: Settings, port: int | None = None, browser: bool = True) -> None:
    """Serve on localhost only. Audio is read straight from the data root."""
    build(paths, settings).queue().launch(
        server_name="127.0.0.1",
        server_port=port or settings.ui_port,
        inbrowser=browser,
        allowed_paths=[str(paths.root)],
    )
