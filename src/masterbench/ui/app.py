"""Gradio app. Thin by design: every handler calls an engine function and formats the result."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from masterbench import audio_io, index, summary
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.matchering_engine import MasteringError
from masterbench.master.pipeline import MasterOptions, RunResult
from masterbench.references import library

UPLOAD_TYPES = sorted(audio_io.LOSSLESS_EXTENSIONS)
REFERENCE_TYPES = sorted(audio_io.AUDIO_EXTENSIONS)
ENGINE_ERRORS = (MasteringError, library.ReferenceError, audio_io.AudioError, LookupError)
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
                run = gr.Button("Master", variant="primary")
                analyze = gr.Button("Analyze only")
            with gr.Column(scale=2):
                result_md = gr.Markdown("" if refs else NO_REFERENCES)
                flags_md = gr.Markdown()
                mix_player = gr.Audio(label="Mix", type="filepath", interactive=False)
                master_player = gr.Audio(label="Master", type="filepath", interactive=False)
                ref_player = gr.Audio(label="Reference", type="filepath", interactive=False)
                downloads = gr.File(label="Files", file_count="multiple", interactive=False)
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

        def on_master(uploaded, run_id, selected_genre, spec, lufs, peak, progress=gr.Progress()):
            if not spec:
                raise gr.Error("Add a reference in the Library tab first.")
            options = MasterOptions(spec, lufs, peak, genre=selected_genre)
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
            return (*run_view(result), gr.Dropdown(choices=run_choices(paths)))

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
            return (*run_view(result), gr.Dropdown(choices=run_choices(paths)))

        def on_open(run_id: str | None):
            if not run_id:
                return (gr.skip(),) * len(view)
            return run_view(pipeline.load_run(paths, run_id))

        def on_load(request: gr.Request):
            """Deep link from a notification: /?run=<id> opens that run."""
            run_id = request.query_params.get("run")
            choices = run_choices(paths)
            if not run_id or run_id not in {value for _, value in choices}:
                return (*(gr.skip(),) * len(view), gr.Dropdown(choices=choices))
            result = pipeline.load_run(paths, run_id)
            return (*run_view(result), gr.Dropdown(choices=choices, value=run_id))

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

        genre.change(on_genre, genre, reference)
        run.click(on_master, [mix, previous, genre, reference, target, ceiling], [*view, previous])
        analyze.click(on_analyze, [mix, previous, genre], [*view, previous])
        previous.input(on_open, previous, view)
        app.load(on_load, None, [*view, previous])
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
