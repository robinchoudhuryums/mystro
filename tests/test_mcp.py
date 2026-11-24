from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters
from mcp.types import TextContent

from masterbench import fixtures, mcp_server
from masterbench.config import Paths, Settings
from masterbench.master import chain, pipeline
from masterbench.master.chain import ChainPreset, Module
from masterbench.master.pipeline import MasterOptions
from masterbench.references import library

TOOLS = {
    "list_tracks",
    "list_runs",
    "get_analysis",
    "compare_runs",
    "analyze_file",
    "suggest_reference",
    "get_progress_trends",
    "get_habits",
    "master_file",
    "run_batch",
    "pitch_report",
}
THIN = ChainPreset("thin", "", (Module("low_shelf", {"freq_hz": 70.0, "gain_db": -8.0}),))


def call(paths: Paths, settings: Settings, name: str, arguments: dict[str, Any] | None = None):
    """Call one tool through a real MCP client connected in-process."""

    async def go():
        async with Client(mcp_server.build_server(paths, settings)) as client:
            return await client.call_tool(name, arguments or {})

    return asyncio.run(go())


def said(result) -> str:
    first = result.content[0]
    assert isinstance(first, TextContent)
    return first.text


def ok(
    paths: Paths, settings: Settings, name: str, arguments: dict[str, Any] | None = None
) -> dict:
    result = call(paths, settings, name, arguments)
    assert not result.is_error, said(result)
    assert result.structured_content is not None
    return result.structured_content


def error(paths: Paths, settings: Settings, name: str, arguments: dict[str, Any]) -> str:
    result = call(paths, settings, name, arguments)
    assert result.is_error
    return said(result)


@pytest.fixture
def library_with_runs(paths: Paths, settings: Settings, tmp_path: Path, fx: dict[str, Path]):
    """Three thin-sounding tracks analyzed, and one of them mastered twice."""
    library.add(paths, fx["reference"], "hip-hop", settings)
    files = []
    for number in range(3):
        loop, _ = fixtures.drum_808_loop(seed=30 + number, detune=False)
        thin = chain.run(THIN, loop, fixtures.SR)
        files.append(
            fixtures.write_wav(paths.root / "Inbox" / "hip-hop" / f"Song {number}.wav", thin)
        )
        pipeline.analyze(files[-1], paths, settings, genre="hip-hop")
    quiet = pipeline.master(
        files[0], MasterOptions.from_settings(settings, "hip-hop/default"), paths, settings
    )
    loud = pipeline.master(files[0], MasterOptions("hip-hop/default", -10.0, -1.0), paths, settings)
    return files, quiet, loud


def test_tools_are_listed_with_schemas(paths: Paths, settings: Settings) -> None:
    async def go():
        async with Client(mcp_server.build_server(paths, settings)) as client:
            return (await client.list_tools()).tools

    tools = {tool.name: tool for tool in asyncio.run(go())}
    assert set(tools) == TOOLS
    for tool in tools.values():
        assert tool.description and tool.output_schema is not None, tool.name
    compare = tools["compare_runs"].input_schema
    assert compare["required"] == ["run_a", "run_b"] and set(compare["properties"]) == {
        "run_a",
        "run_b",
        "role",
    }
    assert tools["master_file"].input_schema["required"] == ["path"]
    assert "Needs writes enabled" in (tools["master_file"].description or "")


def test_compare_runs_and_get_habits_through_the_client(
    paths: Paths, settings: Settings, library_with_runs
) -> None:
    """Phase 7 criterion: an MCP client can call compare_runs and get_habits."""
    _, quiet, loud = library_with_runs
    diff = ok(
        paths, settings, "compare_runs", {"run_a": quiet.manifest.id, "run_b": loud.manifest.id}
    )
    assert diff["role"] == "master" and diff["b"]["run_id"] == loud.manifest.id
    assert 3.0 < diff["b_minus_a"]["lufs_i"] <= 4.1, "the limiter stops short of -10 here"
    assert diff["b_minus_a"]["crest_db"] < -2, "the louder master is more squashed"
    assert set(diff["band_level_b_minus_a_db"]) == {
        "sub",
        "bass",
        "low_mids",
        "mids",
        "upper_mids",
        "presence",
        "air",
    }
    assert "heavy_limiting" in diff["flags"]["only_in_b"]
    assert Path(diff["a"]["file"]).is_file() and len(json.dumps(diff)) < 3000, "compact, with paths"

    mixes = ok(
        paths,
        settings,
        "compare_runs",
        {"run_a": quiet.manifest.id, "run_b": loud.manifest.id, "role": "mix"},
    )
    assert mixes["b_minus_a"]["lufs_i"] == 0.0, "both masters came from the same mix"

    habits = ok(paths, settings, "get_habits")
    assert habits["tracks_considered"] == 3
    thin = next(h for h in habits["habits"] if h["code"] == "band_sub_under")
    assert thin["tracks"] == 3 and "on 3 of the last 3 tracks" in thin["statement"]
    assert 0 < thin["confidence"] < 0.5, "three tracks is a hint, not a certainty"


def test_reading_tools(paths: Paths, settings: Settings, library_with_runs) -> None:
    files, quiet, loud = library_with_runs
    tracks = ok(paths, settings, "list_tracks")["tracks"]
    assert {t["slug"]: t["runs"] for t in tracks} == {"song-0": 3, "song-1": 1, "song-2": 1}
    masters = ok(paths, settings, "list_runs", {"kind": "master"})["runs"]
    assert [r["run_id"] for r in masters] == [loud.manifest.id, quiet.manifest.id]
    assert len(ok(paths, settings, "list_runs", {"track": "song-1"})["runs"]) == 1
    assert len(ok(paths, settings, "list_runs", {"limit": 2})["runs"]) == 2

    full = ok(paths, settings, "get_analysis", {"run_id": quiet.manifest.id})
    assert full["kind"] == "master" and Path(full["files"]["master_24"]).is_file()
    master = full["analyses"]["master"]
    assert master["lufs_i"] == pytest.approx(-14.0, abs=0.2) and master["key"] == "F minor"
    assert "spectrum" not in master and len(json.dumps(full)) < 12000

    trends = ok(paths, settings, "get_progress_trends", {"limit": 2})
    assert [t["slug"] for t in trends["tracks"]] == ["song-1", "song-2"]
    assert trends["tracks"][0]["band_deltas"]["sub"] < -3

    suggested = ok(
        paths, settings, "suggest_reference", {"path": str(files[1]), "genre": "hip-hop"}
    )
    assert suggested["suggestions"][0]["reference"] == "hip-hop/reference"

    measured = ok(paths, settings, "analyze_file", {"path": str(files[2]), "genre": "hip-hop"})
    assert (
        measured["saved"] is False
        and measured["analyses"]["mix"]["band_deltas_vs_reference_db"]["sub"] < -3
    )
    assert len(ok(paths, settings, "list_tracks")["tracks"]) == 3, "nothing was stored"


def test_read_only_by_default_and_limited_to_allowed_folders(
    paths: Paths, settings: Settings, library_with_runs, tmp_path: Path, fx: dict[str, Path]
) -> None:
    files, _, _ = library_with_runs
    for name, arguments in (
        ("master_file", {"path": str(files[0])}),
        ("run_batch", {"file_paths": [str(files[0]), str(files[1])]}),
    ):
        message = error(paths, settings, name, arguments)
        assert "read-only" in message and "mcp_allow_writes = true" in message
    assert len(list(paths.projects.glob("*/runs/*"))) == 5, "no run was created"

    message = error(paths, settings, "analyze_file", {"path": str(fx["loop"])})
    assert "outside the folders this server may read" in message and "mcp_allowed_dirs" in message
    assert "outside the folders" in error(
        paths, settings, "suggest_reference", {"path": "/etc/hostname"}
    )
    assert "no such file" in error(
        paths, settings, "analyze_file", {"path": str(paths.root / "nope.wav")}
    )
    assert "no run with id nope" in error(
        paths, settings, "compare_runs", {"run_a": "nope", "run_b": "x"}
    )
    assert "no run with id" in error(paths, settings, "get_analysis", {"run_id": "../../etc"})

    opened = replace(settings, mcp_allowed_dirs=(str(fx["loop"].parent),))
    assert ok(paths, opened, "analyze_file", {"path": str(fx["loop"])})["saved"] is False


def test_writing_tools_when_allowed(paths: Paths, settings: Settings, library_with_runs) -> None:
    files, _, _ = library_with_runs
    writable = replace(settings, mcp_allow_writes=True)
    mastered = ok(
        paths,
        writable,
        "master_file",
        {"path": str(files[1]), "genre": "hip-hop", "target_lufs": -12.0},
    )
    assert mastered["kind"] == "master" and mastered["target_lufs"] == -12.0
    assert mastered["analyses"]["master"]["lufs_i"] == pytest.approx(-12.0, abs=0.2)
    assert Path(mastered["files"]["master_24"]).is_file()
    assert "match must be" in error(
        paths, writable, "master_file", {"path": str(files[1]), "match": "magic"}
    )

    stored = ok(paths, writable, "analyze_file", {"path": str(files[2]), "genre": "hip-hop"})
    assert stored["saved"] is True and stored["kind"] == "analysis_only"

    done = ok(
        paths,
        writable,
        "run_batch",
        {"file_paths": [str(f) for f in files[:2]], "genre": "hip-hop", "name": "EP"},
    )
    assert len(done["run_ids"]) == 2 and done["failures"] == []
    assert done["consistency"]["loudness_spread_lu"] < 0.5


@pytest.mark.slow
def test_the_real_server_over_stdio(
    data_root: Path, isolated_state: Path, fx: dict[str, Path]
) -> None:
    """`masterbench mcp` as a subprocess, spoken to the way Claude Code would."""
    from masterbench import config

    paths = config.get_paths()
    paths.ensure()
    first = pipeline.analyze(fx["loop"], paths, Settings(), genre="hip-hop")
    second = pipeline.analyze(fx["loop"], paths, Settings(), genre="hip-hop")
    ids = {"run_a": first.manifest.id, "run_b": second.manifest.id, "role": "mix"}
    env = {
        "MASTERBENCH_HOME": str(data_root),
        "MASTERBENCH_STATE": str(isolated_state),
        "PATH": "/usr/bin:/bin",
    }
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "masterbench.cli", "mcp"], env=env
    )

    async def go():
        async with Client(server) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            tracks = await client.call_tool("list_tracks", {})
            habits = await client.call_tool("get_habits", {})
            diff = await client.call_tool("compare_runs", ids)
            return names, *(r.structured_content for r in (tracks, habits, diff))

    names, tracks, habits, diff = asyncio.run(go())
    assert names == TOOLS
    assert tracks is not None and [t["slug"] for t in tracks["tracks"]] == ["loop"]
    assert habits == {"tracks_considered": 1, "habits": []}
    assert diff is not None and diff["b_minus_a"]["lufs_i"] == 0.0
    assert diff["flags"]["in_both"] == ["bass_detuned"]


def test_pitch_report_tool(paths: Paths, settings: Settings, fx: dict[str, Path]) -> None:
    vocal = paths.root / "Inbox" / "Lead Vox.wav"
    vocal.write_bytes(fx["vocal"].read_bytes())
    found = ok(paths, settings, "pitch_report", {"path": str(vocal)})
    assert (found["key"], found["key_source"], found["notes"]) == ("F minor", "detected", 8)
    assert found["in_tune_share"] == pytest.approx(0.75, abs=0.01)
    assert [(n["number"], n["sung"], round(n["cents"])) for n in found["furthest_notes"]] == [
        (3, "C4", 30),
        (6, "G3", 30),
    ]
    assert found["flags"] == [] and len(json.dumps(found)) < 1500
    assert not list(paths.projects.glob("*/vocal")), "the tool stores nothing"

    given = ok(
        paths,
        settings,
        "pitch_report",
        {"path": str(vocal), "key": "F minor", "scale": "chromatic"},
    )
    assert given["key_source"] == "given" and given["scale"] == "chromatic"
    assert "outside the folders" in error(
        paths, settings, "pitch_report", {"path": str(fx["vocal"])}
    )
    assert "not a note name" in error(
        paths, settings, "pitch_report", {"path": str(vocal), "key": "H minor"}
    )
