"""Typer entry point. Commands are added phase by phase; each one calls the engine."""

from __future__ import annotations

import json
import logging
import time
from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer

from masterbench import (
    audio_io,
    batch,
    config,
    fixtures,
    index,
    monitoring,
    notify,
    progress,
    projects,
    summary,
)
from masterbench import track as track_tools
from masterbench.analysis.report import analyze_audio
from masterbench.master import chain, pipeline, presets
from masterbench.master.matchering_engine import MasteringError
from masterbench.notes import mix_notes, trends
from masterbench.preview import speakers, streaming
from masterbench.references import library, suggest, target_curve
from masterbench.release import versions as alternate_versions
from masterbench.stems import separate as stem_separation
from masterbench.stems import workflow as stem_workflow
from masterbench.vocal import engines as vocal_engines
from masterbench.vocal import mix_assistant
from masterbench.vocal import presets as tuning_presets
from masterbench.vocal import scales as vocal_scales
from masterbench.vocal import tuning as vocal_tuning
from masterbench.vocal import workflow as vocal_workflow

app = typer.Typer(no_args_is_help=True, add_completion=False, help="MasterBench")
ref_app = typer.Typer(no_args_is_help=True, help="Manage the genre reference library.")
preset_app = typer.Typer(no_args_is_help=True, help="Manage chain presets.")
app.add_typer(ref_app, name="ref")
stems_app = typer.Typer(no_args_is_help=True, help="Work with stems.")
curve_app = typer.Typer(no_args_is_help=True, help="Genre target curves built from references.")
app.add_typer(preset_app, name="preset")
app.add_typer(stems_app, name="stems")
app.add_typer(curve_app, name="curve")
vocal_app = typer.Typer(no_args_is_help=True, help="Pitch report and tuning for a solo vocal.")
app.add_typer(vocal_app, name="vocal")

AudioFile = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)]
ENGINE_ERRORS = (
    MasteringError,
    library.ReferenceError,
    audio_io.AudioError,
    streaming.PreviewError,
    chain.ChainError,
    stem_separation.StemError,
    mix_notes.NotesError,
    track_tools.TrackError,
    alternate_versions.VersionError,
    vocal_workflow.VocalError,
    vocal_tuning.TuningError,
    vocal_scales.ScaleError,
    vocal_engines.EngineError,
    mix_assistant.MixError,
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
    curve: Annotated[
        bool,
        typer.Option("--curve", help="Match the genre's target curve instead of one reference."),
    ] = False,
) -> None:
    """Master a mix against a reference or a genre target curve. Writes a new run."""
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
        mode="curve" if curve else "matchering",
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
    if ref.quality is not None:
        typer.echo(f"quality: {ref.quality.badge}")
        for issue in ref.quality.issues:
            typer.echo(f"  {summary.flag_line(issue)}")
        if not ref.quality.usable_for_curves:
            typer.echo("  It is left out of target curves unless you include it by name.")


@ref_app.command("list")
def ref_list(genre: Annotated[str | None, typer.Option(help="Only this genre.")] = None) -> None:
    """List references with their loudness and quality badge."""
    refs = library.list_references(config.get_paths(), genre)
    if not refs:
        typer.echo("no references; add one with: masterbench ref add FILE --genre GENRE")
    for ref in refs:
        marker = "*" if ref.is_default else " "
        badge = ref.quality.badge if ref.quality else "unchecked"
        typer.echo(f"{marker} {ref.spec:<36} {badge:<9} {ref.report.headline()}")
        for issue in ref.quality.issues if ref.quality else []:
            typer.echo(f"    {summary.flag_line(issue)}")


@ref_app.command("suggest")
def ref_suggest(
    mix: AudioFile,
    genre: Annotated[
        str | None, typer.Option(help="Genre to search; default: your default.")
    ] = None,
) -> None:
    """Rank a genre's references by how close they sit to a mix."""
    paths, settings = config.get_paths(), config.load_settings()
    try:
        audio, info = audio_io.load(mix)
        report = analyze_audio(audio, info, mix, "reference", settings)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    wanted = genre or settings.default_genre
    found = suggest.suggest(report, library.list_references(paths, wanted))
    if not found:
        typer.echo(f"no references in '{wanted}' to compare with")
    for item in found:
        typer.echo(f"{item.reference.spec:<36} {item.similarity:+.2f}  {item.why}")


@ref_app.command("refresh")
def ref_refresh() -> None:
    """Re-analyze references added before the current measurements existed."""
    paths, settings = config.get_paths(), config.load_settings()
    count = library.refresh(paths, settings)
    with index.open_index(paths) as conn:
        index.sync_references(conn, paths)
    typer.echo(f"{count} reference(s) re-analyzed")


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


@curve_app.command("build")
def curve_build(
    genre: str,
    ref: Annotated[
        list[str] | None, typer.Option("--ref", help="Use only these references (repeatable).")
    ] = None,
    include_low_quality: Annotated[
        bool, typer.Option("--include-low-quality", help="Also use references badged poor.")
    ] = False,
) -> None:
    """Build a genre's target curve: the median spectrum of its references and their range."""
    paths = config.get_paths()
    try:
        curve = target_curve.build(paths, genre, ref, include_low_quality)
    except (target_curve.CurveError, library.ReferenceError) as exc:
        raise _fail(exc) from exc
    typer.echo(f"{curve.genre}: built from {len(curve.references)} references")
    typer.echo(", ".join(curve.references))
    _print_curve(curve)


def _print_curve(curve: target_curve.TargetCurve) -> None:
    typer.echo(f"{'Band':<12} {'25%':>7} {'Median':>7} {'75%':>7}   (dB relative to loudness)")
    for band, value in curve.bands.items():
        typer.echo(f"{band:<12} {value.p25:>7.1f} {value.median:>7.1f} {value.p75:>7.1f}")
    if curve.contrast_lu is not None:
        typer.echo(f"Hook minus verse in these references: {curve.contrast_lu:.1f} LU (median)")


@curve_app.command("show")
def curve_show(genre: str) -> None:
    """Print a genre's target curve."""
    try:
        curve = target_curve.get(config.get_paths(), genre)
    except target_curve.CurveError as exc:
        raise _fail(exc) from exc
    typer.echo(f"{curve.genre}: {len(curve.references)} references, built {curve.created_at[:10]}")
    typer.echo(", ".join(curve.references))
    _print_curve(curve)


@app.command()
def sections(
    run_id: Annotated[str, typer.Argument(help="Id of a run of the track.")],
    markers: Annotated[
        str | None,
        typer.Option("--set", help="Hand-set sections, e.g. '0 intro, 0:14 verse, 0:41 hook'."),
    ] = None,
    clear: Annotated[bool, typer.Option("--clear", help="Go back to automatic sections.")] = False,
) -> None:
    """Show a run's sections, or set the track's sections by hand for later analyses."""
    paths = config.get_paths()
    try:
        run = pipeline.load_run(paths, run_id)
        if clear:
            track_tools.clear_sections(run)
            typer.echo("Hand-set sections cleared; the next analysis detects them again.")
            return
        if markers is not None:
            found = track_tools.set_sections(run, markers)
            typer.echo(f"Saved: {track_tools.markers_text(found)}")
            typer.echo("They apply the next time this track is analyzed or mastered.")
            return
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(summary.sections_text(run.reports["mix"]))
    if run.reports["mix"].sections is not None:
        typer.echo(f"\nTo edit: --set '{track_tools.markers_text(run.reports['mix'].sections)}'")


@app.command()
def key(
    track: Annotated[str, typer.Argument(help="Track slug, e.g. night-drive.")],
    value: Annotated[str | None, typer.Argument(help="E.g. 'F minor'. Omit with --clear.")] = None,
    clear: Annotated[bool, typer.Option("--clear", help="Go back to the detected key.")] = False,
) -> None:
    """Set a track's key by hand, for when detection picks the relative major or minor."""
    if value is None and not clear:
        raise typer.BadParameter("give a key such as 'F minor', or --clear")
    try:
        saved = track_tools.set_key(config.get_paths(), track, None if clear else value)
    except track_tools.TrackError as exc:
        raise _fail(exc) from exc
    typer.echo(
        f"{track}: key {'set to ' + saved if saved else 'cleared'}; applies from the next analysis"
    )


@app.command()
def lowend(run_id: Annotated[str, typer.Argument(help="Id of a run of the track.")]) -> None:
    """808 tuning against the key, what carries on small speakers, and the kick/808 clash."""
    paths, settings = config.get_paths(), config.load_settings()
    try:
        run = pipeline.load_run(paths, run_id)
        found = track_tools.low_end(run, paths, settings)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    mix = run.reports["mix"]
    if mix.musical is not None:
        typer.echo(summary.musical_line(mix))
        typer.echo("")
    typer.echo(summary.lowend_text(found))
    from masterbench.analysis import lowend as lowend_tools

    for flag in lowend_tools.flags(found, settings.thresholds):
        typer.echo(summary.flag_line(flag))


@app.command(name="speakers")
def speakers_command(
    run_id: Annotated[str, typer.Argument(help="Id of the run to render.")],
    force: Annotated[bool, typer.Option("--force", help="Render again even if cached.")] = False,
) -> None:
    """Render phone, earbud, car, laptop, club and mono simulations of a run."""
    paths = config.get_paths()
    try:
        run = pipeline.load_run(paths, run_id)
        previews = speakers.render(run, paths, force)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    for item in previews:
        typer.echo(f"{item.label}: {item.message}")
        typer.echo(f"  {run.run_dir / item.file}")
    typer.echo(speakers.NOTICE)


@app.command()
def versions(
    run_id: Annotated[str, typer.Argument(help="Id of a master run.")],
    stem_set: Annotated[
        str | None, typer.Option("--set", help="Stem set name; default: Logic stems, else newest.")
    ] = None,
) -> None:
    """Make an instrumental and an a cappella that got the same processing as the master."""
    paths, settings = config.get_paths(), config.load_settings()
    try:
        run = pipeline.load_run(paths, run_id)
        found = alternate_versions.make(run, paths, settings, stem_set)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(summary.versions_text(found, alternate_versions.versions_dir(run)))


@app.command()
def habits() -> None:
    """Recurring habits across your recent tracks, from the index."""
    paths, settings = config.get_paths(), config.load_settings()
    with index.open_index(paths) as conn:
        points = progress.track_points(conn)
    found = progress.habits(points, settings.thresholds)
    typer.echo(summary.habits_text(found, min(len(points), progress.HABIT_WINDOW)))


@app.command(name="trends")
def trends_command(
    model: Annotated[
        str | None, typer.Option(help=f"One of {', '.join(mix_notes.PRICES)}.")
    ] = None,
    show_payload: Annotated[
        bool, typer.Option("--payload", help="Print exactly what would be sent, then stop.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Send even past the monthly cap.")] = False,
) -> None:
    """Ask Claude for your top habits and a practice exercise for each. Sends numbers only."""
    paths, settings = config.get_paths(), config.load_settings()
    with index.open_index(paths) as conn:
        payload = trends.gather(conn, settings)
    try:
        if show_payload:
            prompt, _ = mix_notes.load_prompt(trends.PROMPT_FILE)
            used = model or settings.notes_model
            quote = mix_notes.estimate(payload, used, paths, settings, prompt)
            typer.echo(mix_notes.payload_json(payload))
            typer.echo(
                f"\n{quote.model}: about {quote.input_tokens} tokens in, roughly "
                f"${quote.cost_usd:.3f}. Spent this month: ${quote.spent_this_month_usd:.2f} "
                f"of ${quote.cap_usd:.2f}. API key: {mix_notes.key_status()}.",
                err=True,
            )
            return
        result = trends.generate(payload, paths, settings, model, force=force)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(trends.render(result))


@app.command()
def calibrate() -> None:
    """Write the level-calibration noise and print how to set your monitors with it."""
    paths, settings = config.get_paths(), config.load_settings()
    paths.ensure()
    typer.echo(f"Noise file: {monitoring.calibration_file(paths)}")
    typer.echo("")
    typer.echo(monitoring.calibration_steps(settings))


@app.command()
def mcp() -> None:
    """Run the MCP server on stdio, for Claude Code or the Claude app."""
    from masterbench import mcp_server

    mcp_server.serve(config.get_paths(), config.load_settings())


VocalArg = Annotated[
    str,
    typer.Argument(help="A dry vocal file, or a track slug to use that track's vocal stem."),
]
KeyOption = Annotated[
    str | None, typer.Option("--key", help="E.g. 'F minor'. Default: the track's key.")
]
ScaleOption = Annotated[
    str | None,
    typer.Option(
        "--scale", help="Scale on the key's tonic: dorian, minor pentatonic, chromatic..."
    ),
]
NotesOption = Annotated[
    str | None, typer.Option("--notes", help="A custom scale, e.g. 'F G# A# C D#'.")
]


def _vocal_source(paths, source: str, track: str | None) -> vocal_workflow.VocalSource:
    file = Path(source).expanduser()
    if file.is_file():
        return vocal_workflow.open_vocal(paths, file, track)
    if (paths.projects / source / "track.json").is_file():
        return vocal_workflow.from_stems(paths, source)
    raise vocal_workflow.VocalError(f"'{source}' is neither an audio file nor a track")


def _custom(notes: str | None) -> tuple[str, ...]:
    return tuple(notes.replace(",", " ").split()) if notes else ()


def _vocal_key(source: vocal_workflow.VocalSource, paths, key: str | None) -> str:
    """The key to tune in: given, else the stored report's, else a fresh report's."""
    if key:
        return key
    found = vocal_workflow.stored_report(source) or vocal_workflow.report(source, paths)
    return found.key


def _parse_edits(bypass: str | None, targets: list[str]) -> vocal_tuning.Edits:
    edits: vocal_tuning.Edits = {}
    try:
        for number in (bypass or "").replace(",", " ").split():
            edits[int(number) - 1] = vocal_tuning.NoteEdit(bypass=True)
        for pair in targets:
            number, _, note = pair.partition("=")
            edits[int(number) - 1] = vocal_tuning.NoteEdit(target=vocal_scales.parse_note(note))
    except ValueError as exc:
        raise typer.BadParameter(f"edits look like --bypass 3,7 and --set 5=G#3 ({exc})") from exc
    return edits


@vocal_app.command("report")
def vocal_report(
    source: VocalArg,
    track: Annotated[str | None, typer.Option(help="Track this vocal belongs to.")] = None,
    key: KeyOption = None,
    scale: ScaleOption = None,
    notes: NotesOption = None,
) -> None:
    """How in tune a vocal is, note by note. Reads only; nothing is changed."""
    paths = config.get_paths()
    try:
        found = _vocal_source(paths, source, track)
        report = vocal_workflow.report(found, paths, key, scale, _custom(notes))
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(f"{found.label} [{found.id}]")
    typer.echo(summary.pitch_text(report))


@vocal_app.command("tune")
def vocal_tune(
    source: VocalArg,
    preset: Annotated[str, typer.Option(help="natural, tight, hard-tune or one of yours.")] = (
        tuning_presets.DEFAULT
    ),
    track: Annotated[str | None, typer.Option(help="Track this vocal belongs to.")] = None,
    key: KeyOption = None,
    scale: ScaleOption = None,
    notes: NotesOption = None,
    amount: Annotated[
        float | None, typer.Option(min=0, max=100, help="How much to correct, 0 to 100.")
    ] = None,
    retune: Annotated[
        float | None, typer.Option(min=0, max=1000, help="Retune speed in ms; 0 is instant.")
    ] = None,
    humanize: Annotated[
        float | None, typer.Option(min=0, max=100, help="100 keeps vibrato; 0 flattens.")
    ] = None,
    engine: Annotated[str | None, typer.Option(help="rubberband or world.")] = None,
    formant_shift: Annotated[
        float | None, typer.Option(help="Move the formants, in semitones (WORLD only).")
    ] = None,
    note_hold: Annotated[
        bool | None, typer.Option("--note-hold/--no-note-hold", help="One step per slide.")
    ] = None,
    bypass: Annotated[str | None, typer.Option(help="Notes to leave alone, by number: 3,7")] = None,
    set_target: Annotated[
        list[str] | None, typer.Option("--set", help="Send a note somewhere else: 5=G#3")
    ] = None,
    clear_edits: Annotated[bool, typer.Option(help="Forget stored bypasses and targets.")] = False,
    save_preset: Annotated[
        str | None, typer.Option(help="Also save these settings as a preset of this name.")
    ] = None,
) -> None:
    """Tune a vocal to its key. Writes tuned_<settings>.wav; the original is untouched."""
    paths = config.get_paths()
    try:
        found = _vocal_source(paths, source, track)
        chosen = tuning_presets.get(paths, preset)
        settings = chosen.settings(
            _vocal_key(found, paths, key),
            scale,
            _custom(notes),
            amount=None if amount is None else amount / 100,
            retune_ms=retune,
            humanize=None if humanize is None else humanize / 100,
            engine=engine,
            formant_shift=formant_shift,
            note_hold=note_hold,
        )
        stored = {} if clear_edits else vocal_workflow.read_edits(found)
        edits = vocal_workflow.write_edits(
            found, {**stored, **_parse_edits(bypass, set_target or [])}
        )
        if save_preset:
            saved = tuning_presets.from_settings(save_preset, settings, f"Saved from {preset}.")
            typer.echo(f"saved preset: {tuning_presets.save(paths, saved)}")
        version = vocal_workflow.tune(found, settings, edits, label=preset.capitalize())
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(summary.tuned_text(version, found.folder))


@vocal_app.command("tune-mix")
def vocal_tune_mix(
    mix: AudioFile,
    preset: Annotated[str, typer.Option(help="natural, tight, hard-tune or one of yours.")] = (
        tuning_presets.DEFAULT
    ),
    key: KeyOption = None,
    scale: ScaleOption = None,
    title: Annotated[str | None, typer.Option(help="Track title.")] = None,
    genre: Annotated[str | None, typer.Option(help="Genre of the track.")] = None,
    model: Annotated[str, typer.Option(help="Separation model.")] = stem_separation.DEFAULT_MODEL,
) -> None:
    """Tune the vocal inside a full mix: separate it, tune it, put it back. Preview quality."""
    paths, app_settings = config.get_paths(), config.load_settings()
    try:
        if key is None:
            analyzed = pipeline.analyze(mix, paths, app_settings, genre, title)
            musical = analyzed.reports["mix"].musical
            if musical is None:
                raise vocal_workflow.VocalError("no key could be detected; pass --key")
            key = musical.key
        settings = tuning_presets.get(paths, preset).settings(key, scale)
        found, version = vocal_workflow.tune_mix(paths, mix, settings, title, genre, model)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(summary.tuned_text(version, found.folder))


@vocal_app.command("benchmark")
def vocal_benchmark(
    source: VocalArg,
    track: Annotated[str | None, typer.Option(help="Track this vocal belongs to.")] = None,
    key: KeyOption = None,
    scale: ScaleOption = None,
    add: Annotated[
        list[str] | None,
        typer.Option(help="The same vocal tuned elsewhere, as Label=file: 'Waves=waves.wav'"),
    ] = None,
) -> None:
    """Tune one vocal with both engines, measure them, and line them up for a blind test."""
    paths = config.get_paths()
    try:
        found = _vocal_source(paths, source, track)
        for pair in add or []:
            label, _, file = pair.partition("=")
            if not file:
                raise typer.BadParameter("--add takes Label=file, e.g. 'Waves=waves.wav'")
            vocal_workflow.add_external(found, Path(file).expanduser(), label)
        rows = vocal_workflow.benchmark(found, paths, _vocal_key(found, paths, key), scale)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(summary.benchmark_text(rows))
    typer.echo("")
    listed = [c["label"] for c in vocal_workflow.candidates(found)]
    typer.echo(f"Ready to compare in the Vocal tab [{found.id}]: {', '.join(listed)}")


@vocal_app.command("mix")
def vocal_mix(
    vocal: Annotated[
        Path | None,
        typer.Argument(exists=True, dir_okay=False, help="The dry lead vocal, tuned or not."),
    ] = None,
    instrumental: Annotated[
        Path | None,
        typer.Argument(exists=True, dir_okay=False, help="The beat, bounced from the same start."),
    ] = None,
    from_stems: Annotated[
        str | None, typer.Option(help="Use a track's stems instead of two files.")
    ] = None,
    adlib: Annotated[
        list[Path] | None, typer.Option(exists=True, dir_okay=False, help="Ad-lib or backing stem.")
    ] = None,
    title: Annotated[str | None, typer.Option(help="Track title.")] = None,
    genre: Annotated[str | None, typer.Option(help="Genre of the track.")] = None,
    preset: Annotated[
        str, typer.Option("--chain", help="Vocal chain: gentle, modern-rap, aggressive or yours.")
    ] = mix_assistant.DEFAULT_CHAIN,
    target: Annotated[
        float | None,
        typer.Option(help="Vocal loudness against the beat, in LU. Default: by genre."),
    ] = None,
    target_from_reference: Annotated[
        bool, typer.Option(help="Take the target from the reference's stems (separates it once).")
    ] = False,
    ride: Annotated[
        float, typer.Option(min=0, max=12, help="How far the rider may move, dB.")
    ] = 6.0,
    carve: Annotated[
        float, typer.Option(min=0, max=6, help="Dip in the beat's 1-4 kHz, dB.")
    ] = 3.0,
    duck: Annotated[float, typer.Option(min=0, max=6, help="Duck the beat in hooks, dB.")] = 0.0,
    adlib_preset: Annotated[str, typer.Option(help="tucked, wide, left or right.")] = "tucked",
    ref: Annotated[str | None, typer.Option(help="Reference to master against.")] = None,
    curve: Annotated[bool, typer.Option("--curve", help="Master toward the genre curve.")] = False,
    lufs: Annotated[float | None, typer.Option(help="Loudness target for the master.")] = None,
    notes: Annotated[
        bool, typer.Option("--notes", help="Ask for AI mix notes too (numbers only are sent).")
    ] = False,
    intent: Annotated[str, typer.Option(help="What you are going for, for the notes.")] = "",
) -> None:
    """Mix a dry vocal into an instrumental and master it: rough to release candidate."""
    paths, app_settings = config.get_paths(), config.load_settings()
    if (vocal is None or instrumental is None) and not from_stems:
        raise typer.BadParameter("give a vocal and an instrumental, or --from-stems TRACK")
    try:
        options = mix_assistant.MixOptions(
            chain=preset,
            title=title,
            genre=genre,
            target_db=target,
            target_from_reference=target_from_reference,
            max_ride_db=ride,
            carve_db=carve,
            duck_db=duck,
            adlib_preset=adlib_preset,
            reference=ref,
            by_curve=curve,
            target_lufs=lufs,
        )
        if from_stems:
            result = mix_assistant.from_stems(
                paths,
                app_settings,
                from_stems,
                options,
                progress=_progress,
                notes=notes,
                intent=intent,
            )
        else:
            assert vocal is not None and instrumental is not None
            result = mix_assistant.mix(
                paths, app_settings, vocal, instrumental, options, adlib, _progress, notes, intent
            )
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(summary.vocal_mix_text(result))
    if result.notes is not None:
        typer.echo("")
        typer.echo(mix_notes.render(result.notes))


@vocal_app.command("chains")
def vocal_chains() -> None:
    """List the vocal chain presets."""
    paths = config.get_paths()
    for name in mix_assistant.chain_names(paths):
        found = mix_assistant.get_chain(paths, name)
        typer.echo(f"{name:<12} {found.description}")


@vocal_app.command("presets")
def vocal_presets() -> None:
    """List the tuning presets."""
    paths = config.get_paths()
    for name in tuning_presets.names(paths):
        preset = tuning_presets.get(paths, name)
        values = preset.settings("C major")
        typer.echo(
            f"{name:<12} retune {values.retune_ms:g} ms, humanize {values.humanize:.0%}, "
            f"amount {values.amount:.0%}, {values.engine}"
        )
        if preset.description:
            typer.echo(f"             {preset.description}")


if __name__ == "__main__":
    app()
