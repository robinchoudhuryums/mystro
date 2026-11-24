"""Typer entry point. Commands are added phase by phase; each one calls the engine."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer

from masterbench import config, db, fixtures

app = typer.Typer(no_args_is_help=True, add_completion=False, help="MasterBench")


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(metadata.version("masterbench"))


@app.command()
def init() -> None:
    """Create the data folders and bring the index database up to date."""
    paths = config.get_paths()
    paths.ensure()
    conn = db.connect(paths.db)
    try:
        applied = db.migrate(conn)
    finally:
        conn.close()
    typer.echo(f"data root: {paths.root}")
    typer.echo(f"migrations applied: {', '.join(m.name for m in applied) or 'none (up to date)'}")


@app.command(name="fixtures")
def make_fixtures(
    out_dir: Annotated[Path, typer.Argument(help="Folder to write the generated audio into.")],
) -> None:
    """Generate the synthetic test audio (pink noise, sweep, 808 loop, vocal, clipped mix)."""
    for name, path in fixtures.write_all(out_dir).items():
        typer.echo(f"{name}: {path}")
