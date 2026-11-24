"""Typer entry point. Commands are added phase by phase; each one calls the engine."""

from __future__ import annotations

import json
import logging
import time
from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer

from masterbench import audio_io, batch, config, fixtures, index, notify, projects, summary
from masterbench.master import chain, pipeline, presets
from masterbench.master.matchering_engine import MasteringError
from masterbench.notes import mix_notes
from masterbench.preview import streaming
from masterbench.references import library
from masterbench.stems import separate as stem_separation
from masterbench.stems import workflow as stem_workflow

app = typer.Typer(no_args_is_help=True, add_completion=False, help="MasterBench")
ref_app = typer.Typer(no_args_is_help=True, help="Manage the genre reference library.")
preset_app = typer.Typer(no_args_is_help=True, help="Manage chain presets.")
app.add_typer(ref_app, name="ref")
stems_app = typer.Typer(no_args_is_help=True, help="Work with stems.")
app.add_typer(preset_app, name="preset")
app.add_typer(stems_app, name="stems")

AudioFile = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)]
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


def _fail(exc: Exception) -> typer.Exit:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    return typer.Exit(1)


def _or(value: float | None, default: float) -> float:
    return default if value is None else value


def _progress(stage: str, fraction: float) -> None:
    typer.echo(f"[{fraction:>4.0%}] {stage}", err=True)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(metadata.version("masterbench"))


@app.command()
def init(
    home: Annotated[
        Path | None, typer.Option(help="Put the data root here and remember it.")
    ] = None,
    icloud: Annotated[
        bool, typer.Option("--icloud", help="Put the data root in iCloud Drive/MasterBench.")
    ] = False,
) -> None:
    """Create the data folders and bring the index database up to date."""
    if icloud:
        if not config.ICLOUD_DRIVE.is_dir():
            raise _fail(
                RuntimeError(f"iCloud Drive is not set up on this Mac ({config.ICLOUD_DRIVE})")
            )
        home = config.ICLOUD_DRIVE / "MasterBench"
    if home is not None:
        config.set_data_root(home)
    paths = config.get_paths()
    paths.ensure()
    with index.open_index(paths) as conn:
        versions = conn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
        runs = conn.execute("SELECT count(*) FROM runs").fetchone()[0]
    typer.echo(f"data root: {paths.root}")
    typer.echo(f"database: {paths.db} (schema version {versions}, {runs} runs indexed)")
    if paths.synced:
        typer.echo(
            "The data root is in a synced folder, so the index is kept on this machine only.\n"
            "Keep the folder downloaded (Finder: right-click > Keep Downloaded), or audio that "
            "iCloud has moved off this Mac will have to download before it can be read."
        )


@app.command(name="fixtures")
def make_fixtures(
    out_dir: Annotated[Path, typer.Argument(help="Folder to write the generated audio into.")],
) -> None:
    """Generate the synthetic test audio (pink noise, sweep, 808 loop, vocal, clipped mix)."""
    for name, path in fixtures.write_all(out_dir).items():
        typer.echo(f"{name}: {path}")


@app.command()
def master(
    mix: AudioFile,
    ref: Annotated[
        str | None,
        typer.Option(help="'genre/default', 'genre/name' or a file. Default: the genre default."),
    ] = None,
    genre: Annotated[str | None, typer.Option(help="Genre of the track.")] = None,
    lufs: Annotated[float | None, typer.Option(help="Integrated loudness target.")] = None,
    ceiling: Annotated[float | None, typer.Option(help="True-peak ceiling in dBTP.")] = None,
    match_loudness: Annotated[
        bool, typer.Option("--match-loudness", help="Keep the reference's loudness; no target.")
    ] = False,
    title: Annotated[str | None, typer.Option(help="Track title; groups takes of a song.")] = None,
    pre: Annotated[str | None, typer.Option(help="Chain preset to run before matching.")] = None,
    post: Annotated[
        str | None, typer.Option("--post", "--preset", help="Chain preset to run after matching.")
    ] = None,
    notify_done: Annotated[
        bool, typer.Option("--notify", help="Show a macOS notification when finished.")
    ] = False,
) -> None:
    """Master a mix against a reference. Writes a new run and prints a summary."""
    paths, settings = config.get_paths(), config.load_settings()
    paths.ensure()
    spec = ref or f"{genre or settings.default_genre}/default"
    options = pipeline.MasterOptions(
        reference=spec,
        target_lufs=None if match_loudness else _or(lufs, settings.targets.lufs_i),
        ceiling_dbtp=_or(ceiling, settings.targets.true_peak_dbtp),
        genre=genre,
        title=title,
        pre_chain=pre,
        post_chain=post,
    )
    try:
        result = pipeline.master(mix, options, paths, settings, _progress)
    except ENGINE_ERRORS as exc:
        if notify_done:
            notify.notify(f"MasterBench: {mix.name}", f"Mastering failed: {exc}")
        raise _fail(exc) from exc
    if notify_done:
        url = f"http://127.0.0.1:{settings.ui_port}/?run={result.manifest.id}"
        title = f"Mastered: {result.manifest.track.title}"
        notify.notify(title, summary.headline(result), url)
    typer.echo(summary.as_text(result))


@app.command()
def analyze(
    mix: AudioFile,
    genre: Annotated[str | None, typer.Option(help="Genre of the track.")] = None,
    title: Annotated[str | None, typer.Option(help="Track title; groups takes of a song.")] = None,
) -> None:
    """Validate and analyze a mix without mastering it."""
    paths, settings = config.get_paths(), config.load_settings()
    paths.ensure()
    try:
        result = pipeline.analyze(mix, paths, settings, genre, title, _progress)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(summary.as_text(result))


@app.command()
def preview(
    run_id: Annotated[str, typer.Argument(help="Id of a master run, as printed by 'master'.")],
    force: Annotated[bool, typer.Option("--force", help="Render again even if cached.")] = False,
) -> None:
    """Render codec previews of a master and show what streaming platforms will do to it."""
    paths, settings = config.get_paths(), config.load_settings()
    try:
        run = pipeline.load_run(paths, run_id)
        result = streaming.render(run, settings, force)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(f"master true peak {result.source_true_peak_dbtp:+.2f} dBTP")
    for codec in result.codecs:
        typer.echo(
            f"{codec.label:<34} {codec.true_peak_dbtp:+.2f} dBTP ({codec.peak_change_db:+.2f} dB)"
            f"  {run.run_dir / codec.decoded_file}"
        )
    for reason in result.skipped:
        typer.echo(f"skipped {reason}")
    for flag in result.flags:
        typer.echo(summary.flag_line(flag))
    for gain in result.platforms:
        typer.echo(gain.message)


@app.command()
def reindex() -> None:
    """Rebuild the SQLite index from the run manifests and reference library on disk."""
    paths = config.get_paths()
    paths.ensure()
    with index.open_index(paths) as conn:
        counts = index.reindex(conn, paths)
    typer.echo(", ".join(f"{count} {name}" for name, count in counts.items()))


@app.command()
def watch(
    extra: Annotated[
        list[Path] | None,
        typer.Option("--dir", exists=True, file_okay=False, help="Extra folder to watch."),
    ] = None,
    once: Annotated[
        bool, typer.Option("--once", help="Process what is already there, then exit.")
    ] = False,
) -> None:
    """Analyze bounces as they land in the Inbox. The subfolder names the genre."""
    from masterbench.watch import InboxWatcher, WatchEvent, describe

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    paths, settings = config.get_paths(), config.load_settings()
    paths.ensure()

    def report(event: WatchEvent) -> None:
        title, body, _ = describe(event, settings)
        typer.echo(f"{title}: {body}")

    watcher = InboxWatcher(paths, settings, tuple(extra or ()), on_event=report)
    watcher.start()
    typer.echo(f"watching {', '.join(str(root) for root in watcher.roots)}")
    try:
        if once:
            watcher.drain()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()


@app.command()
def ui(
    port: Annotated[int | None, typer.Option(help="Port on 127.0.0.1.")] = None,
    browser: Annotated[bool, typer.Option(help="Open the browser.")] = True,
) -> None:
    """Launch the local web UI."""
    from masterbench.ui.app import launch

    launch(config.get_paths(), config.load_settings(), port, browser)


@ref_app.command("add")
def ref_add(
    file: AudioFile,
    genre: Annotated[str, typer.Option(help="Genre folder, e.g. hip-hop.")],
    notes: Annotated[str, typer.Option(help="Why this reference.")] = "",
    default: Annotated[bool, typer.Option("--default", help="Make it the genre default.")] = False,
) -> None:
    """Copy a track into the reference library and cache its analysis."""
    paths, settings = config.get_paths(), config.load_settings()
    paths.ensure()
    try:
        ref = library.add(paths, file, genre, settings, notes, default)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    with index.open_index(paths) as conn:
        index.sync_references(conn, paths)
    marker = " (default)" if ref.is_default else ""
    typer.echo(f"{ref.spec}{marker}: {ref.report.headline()}")


@ref_app.command("list")
def ref_list(genre: Annotated[str | None, typer.Option(help="Only this genre.")] = None) -> None:
    """List references with their loudness."""
    refs = library.list_references(config.get_paths(), genre)
    if not refs:
        typer.echo("no references; add one with: masterbench ref add FILE --genre GENRE")
    for ref in refs:
        marker = "*" if ref.is_default else " "
        typer.echo(f"{marker} {ref.spec:<40} {ref.report.headline()}")


@ref_app.command("default")
def ref_default(genre: str, name: str) -> None:
    """Choose the default reference for a genre."""
    paths = config.get_paths()
    try:
        library.set_default(paths, genre, name)
    except library.ReferenceError as exc:
        raise _fail(exc) from exc
    with index.open_index(paths) as conn:
        index.sync_references(conn, paths)
    typer.echo(f"{library.get(paths, genre, 'default').spec} is now the default")


@ref_app.command("remove")
def ref_remove(genre: str, name: str) -> None:
    """Remove a reference from the library."""
    paths = config.get_paths()
    try:
        library.remove(paths, genre, name)
    except library.ReferenceError as exc:
        raise _fail(exc) from exc
    with index.open_index(paths) as conn:
        index.sync_references(conn, paths)
    typer.echo(f"removed {genre}/{name}")


@preset_app.command("list")
def preset_list() -> None:
    """List chain presets (yours and the packaged ones)."""
    paths = config.get_paths()
    for name in presets.names(paths):
        preset = presets.get(paths, name)
        steps = " > ".join(module.type for module in preset.modules)
        typer.echo(f"{name:<18} {steps}")
        if preset.description:
            typer.echo(f"{'':<18} {preset.description}")


@preset_app.command("show")
def preset_show(name: str) -> None:
    """Print a preset's TOML."""
    try:
        typer.echo(presets.text(config.get_paths(), name).rstrip())
    except chain.ChainError as exc:
        raise _fail(exc) from exc


@preset_app.command("add-plugin")
def preset_add_plugin(
    name: Annotated[str, typer.Argument(help="Preset to append to; created if missing.")],
    plugin: Annotated[Path, typer.Argument(exists=True, help="A .vst3 or .component bundle.")],
    plugin_name: Annotated[
        str | None, typer.Option(help="Plugin inside a shell bundle, e.g. a Waves plugin.")
    ] = None,
) -> None:
    """Append a plugin, with its current parameters and state, to a preset."""
    paths = config.get_paths()
    try:
        modules = presets.get(paths, name).modules if name in presets.names(paths) else ()
        description = presets.get(paths, name).description if modules else ""
        module = chain.capture_plugin(plugin, plugin_name)
        saved = presets.save(paths, chain.ChainPreset(name, description, (*modules, module)))
    except chain.ChainError as exc:
        raise _fail(exc) from exc
    typer.echo(f"{saved}: {len(module.params['parameters'])} parameters captured")


@app.command()
def plugins(
    params: Annotated[
        Path | None, typer.Option(exists=True, help="Show this plugin's parameters instead.")
    ] = None,
    plugin_name: Annotated[str | None, typer.Option(help="Plugin inside a shell bundle.")] = None,
) -> None:
    """List the VST3 and AU plugins found in the standard folders."""
    if params is not None:
        try:
            listed = chain.plugin_parameters(params, plugin_name)
        except chain.ChainError as exc:
            raise _fail(exc) from exc
        for p in listed:
            bounds = f"{p['min']:g} to {p['max']:g}" if p["min"] is not None else ""
            typer.echo(f"{p['name']:<28} {p['value']!s:<14} {bounds} {p['units'] or ''}".rstrip())
        return
    found = chain.scan_plugins()
    if not found:
        typer.echo(f"no plugins in {', '.join(str(d) for d in chain.plugin_dirs())}")
    for info in found:
        inside = f"  [{', '.join(info.names)}]" if len(info.names) > 1 else ""
        typer.echo(f"{info.format:<5} {info.path}{inside}")


def _genre_reference(paths, genre: str | None, compare: bool) -> library.Reference | None:
    if not compare or not genre:
        return None
    try:
        return library.get(paths, genre, "default")
    except library.ReferenceError:
        return None


@app.command()
def separate(
    mix: AudioFile,
    model: Annotated[
        str, typer.Option(help=f"Stem model: {', '.join(stem_separation.MODELS)}.")
    ] = stem_separation.DEFAULT_MODEL,
    genre: Annotated[str | None, typer.Option(help="Genre of the track.")] = None,
    title: Annotated[str | None, typer.Option(help="Track title; groups takes of a song.")] = None,
    compare: Annotated[
        bool, typer.Option(help="Also separate the genre's default reference and compare.")
    ] = True,
) -> None:
    """Separate a mix into stems (cached) and report the stem balance."""
    paths, settings = config.get_paths(), config.load_settings()
    paths.ensure()
    try:
        project = projects.open_project(paths, mix, title, genre)
        typer.echo(f"separating with {model} (the first use downloads the model)...", err=True)
        stem_set = stem_workflow.separate_mix(project, model)
        reference = _genre_reference(paths, project.track.genre, compare)
        diagnosis = stem_workflow.diagnose(stem_set, project.input_path, settings, reference)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    if stem_set.seconds is not None:
        typer.echo(f"separated in {stem_set.seconds:.0f} s")
    typer.echo(summary.stems_text(diagnosis))
    typer.echo(f"\n{stem_set.dir}")


@stems_app.command("import")
def stems_import(
    folder: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    title: Annotated[str, typer.Option(help="Track title.")],
    genre: Annotated[str | None, typer.Option(help="Genre of the track.")] = None,
    mix: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="The bounce of the full mix.")
    ] = None,
    compare: Annotated[
        bool, typer.Option(help="Compare with the genre default reference's separated stems.")
    ] = False,
) -> None:
    """Import a folder of stems exported from Logic. Roles are guessed from the file names."""
    paths, settings = config.get_paths(), config.load_settings()
    paths.ensure()
    try:
        project, stem_set = stem_workflow.import_logic_stems(paths, folder, title, genre, mix)
        reference = _genre_reference(paths, project.track.genre, compare)
        diagnosis = stem_workflow.diagnose(stem_set, project.input_path, settings, reference)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    for name, role in stem_set.roles.items():
        typer.echo(f"{name:<24} {role}")
    typer.echo("")
    typer.echo(summary.stems_text(diagnosis))


@stems_app.command("list")
def stems_list(track: Annotated[str, typer.Argument(help="Track slug, e.g. night-drive.")]) -> None:
    """List a track's stem sets."""
    sets = stem_separation.list_sets(config.get_paths().projects / track)
    if not sets:
        typer.echo("no stem sets")
    for stem_set in sets:
        kind = "separated" if stem_set.separated else "from Logic"
        typer.echo(f"{stem_set.name:<24} {kind:<11} {', '.join(stem_set.stems)}")


def _parse_gains(pairs: list[str]) -> dict[str, float]:
    gains = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        try:
            gains[name.strip()] = float(value)
        except ValueError:
            raise typer.BadParameter(f"use NAME=DB, e.g. vocals=+2: {pair!r}") from None
    return gains


@app.command()
def rebalance(
    track: Annotated[str, typer.Argument(help="Track slug, e.g. night-drive.")],
    gain: Annotated[
        list[str], typer.Option("--gain", help="STEM=DB, repeatable; limited to ±6 dB.")
    ],
    stem_set: Annotated[
        str | None, typer.Option("--set", help="Stem set name; default: Logic stems, else newest.")
    ] = None,
    ref: Annotated[str | None, typer.Option(help="Reference for mastering the result.")] = None,
    post: Annotated[str | None, typer.Option(help="Post-chain preset.")] = None,
) -> None:
    """Change stem levels, re-sum, and master the result as a new run."""
    paths, settings = config.get_paths(), config.load_settings()
    try:
        chosen = stem_workflow.find_set(paths, track, stem_set)
        data = json.loads((paths.projects / track / "track.json").read_text(encoding="utf-8"))
        spec = ref or f"{data.get('genre') or settings.default_genre}/default"
        options = pipeline.MasterOptions.from_settings(
            settings, spec, title=data["title"], genre=data.get("genre"), post_chain=post
        )
        rebalanced, result = stem_workflow.rebalance_and_master(
            paths, settings, chosen, _parse_gains(gain), options, progress=_progress
        )
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    if rebalanced.trim_db < 0:
        typer.echo(
            f"The new balance was turned down {-rebalanced.trim_db:.1f} dB to keep headroom."
        )
    if chosen.separated:
        typer.echo("Built from separated stems: listen for artifacts before trusting it.")
    typer.echo(summary.as_text(result))


@app.command(name="batch")
def batch_command(
    mixes: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False, readable=True)],
    ref: Annotated[str | None, typer.Option(help="Reference for every track.")] = None,
    genre: Annotated[str | None, typer.Option(help="Genre of the tracks.")] = None,
    name: Annotated[str, typer.Option(help="A name for the set, e.g. the EP title.")] = "",
    lufs: Annotated[float | None, typer.Option(help="Integrated loudness target.")] = None,
    ceiling: Annotated[float | None, typer.Option(help="True-peak ceiling in dBTP.")] = None,
    pre: Annotated[str | None, typer.Option(help="Chain preset to run before matching.")] = None,
    post: Annotated[
        str | None, typer.Option("--post", "--preset", help="Chain preset to run after matching.")
    ] = None,
    notify_done: Annotated[
        bool, typer.Option("--notify", help="Show a macOS notification when finished.")
    ] = False,
) -> None:
    """Master several tracks the same way and report how consistent the set is."""
    paths, settings = config.get_paths(), config.load_settings()
    paths.ensure()
    options = pipeline.MasterOptions(
        reference=ref or f"{genre or settings.default_genre}/default",
        target_lufs=_or(lufs, settings.targets.lufs_i),
        ceiling_dbtp=_or(ceiling, settings.targets.true_peak_dbtp),
        genre=genre,
        pre_chain=pre,
        post_chain=post,
    )
    result = batch.run_batch(mixes, options, paths, settings, name, progress=_progress)
    typer.echo(summary.batch_text(result))
    if notify_done:
        done = f"{len(result.runs)} mastered, {len(result.failures)} failed"
        notify.notify(f"MasterBench batch{': ' + name if name else ''}", done)
    if result.failures:
        raise typer.Exit(1)


@app.command()
def notes(
    run_id: Annotated[str, typer.Argument(help="Id of the run to review.")],
    intent: Annotated[str, typer.Option(help="What you were going for, in your own words.")] = "",
    model: Annotated[
        str | None, typer.Option(help=f"One of {', '.join(mix_notes.PRICES)}.")
    ] = None,
    show_payload: Annotated[
        bool, typer.Option("--payload", help="Print exactly what would be sent, then stop.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Send even past the monthly cap.")] = False,
) -> None:
    """Ask Claude for mix notes from a run's measurements. Sends numbers, never audio."""
    paths, settings = config.get_paths(), config.load_settings()
    try:
        run = pipeline.load_run(paths, run_id)
        payload = mix_notes.build_payload(run, intent)
        quote = mix_notes.estimate(payload, model or settings.notes_model, paths, settings)
        if show_payload:
            typer.echo(mix_notes.payload_json(payload))
            typer.echo(
                f"\n{quote.model}: about {quote.input_tokens} tokens in, roughly "
                f"${quote.cost_usd:.3f}. Spent this month: ${quote.spent_this_month_usd:.2f} "
                f"of ${quote.cap_usd:.2f}. API key: {mix_notes.key_status()}.",
                err=True,
            )
            return
        result = mix_notes.generate(run, paths, settings, intent, model, force=force)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    with index.open_index(paths) as conn:
        index.sync_notes(conn, paths)
    typer.echo(mix_notes.render(result))
