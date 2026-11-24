"""Typer entry point. Commands are added phase by phase; each one calls the engine."""

from __future__ import annotations

import logging
import time
from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer

from masterbench import audio_io, config, fixtures, index, summary
from masterbench.master import pipeline
from masterbench.master.matchering_engine import MasteringError
from masterbench.references import library

app = typer.Typer(no_args_is_help=True, add_completion=False, help="MasterBench")
ref_app = typer.Typer(no_args_is_help=True, help="Manage the genre reference library.")
app.add_typer(ref_app, name="ref")

AudioFile = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)]
ENGINE_ERRORS = (MasteringError, library.ReferenceError, audio_io.AudioError)


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
def init() -> None:
    """Create the data folders and bring the index database up to date."""
    paths = config.get_paths()
    paths.ensure()
    with index.open_index(paths) as conn:
        versions = conn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
    typer.echo(f"data root: {paths.root}")
    typer.echo(f"database: {paths.db} (schema version {versions})")


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
    )
    try:
        result = pipeline.master(mix, options, paths, settings, _progress)
    except ENGINE_ERRORS as exc:
        raise _fail(exc) from exc
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
