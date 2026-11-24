from __future__ import annotations

import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from masterbench import index, watch
from masterbench.config import Paths, Settings
from masterbench.references.library import Reference
from masterbench.watch import InboxWatcher, WatchEvent


@pytest.fixture
def fast(settings: Settings) -> Settings:
    return replace(settings, watch_stable_s=0.3)


@pytest.fixture
def notifications(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str | None]]:
    sent: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(watch.notify, "notify", lambda *args: sent.append(args) or True)
    return sent


def test_genre_comes_from_the_first_subfolder(paths: Paths) -> None:
    assert watch.genre_for(paths.inbox / "house" / "a.wav", paths.inbox) == "house"
    assert watch.genre_for(paths.inbox / "house" / "ep" / "a.wav", paths.inbox) == "house"
    assert watch.genre_for(paths.inbox / "a.wav", paths.inbox) is None


def test_wait_until_stable_outlasts_a_chunked_write(tmp_path: Path) -> None:
    target = tmp_path / "bounce.wav"
    target.write_bytes(b"\0" * 1000)

    def keep_writing() -> None:
        for _ in range(4):
            time.sleep(0.15)
            with target.open("ab") as handle:
                handle.write(b"\0" * 1000)

    writer = threading.Thread(target=keep_writing)
    writer.start()
    started = time.monotonic()
    assert watch.wait_until_stable(target, stable_s=0.4, poll_s=0.05)
    writer.join()
    assert time.monotonic() - started >= 0.9
    assert target.stat().st_size == 5000


def test_wait_until_stable_gives_up_on_missing_or_empty_files(tmp_path: Path) -> None:
    assert not watch.wait_until_stable(tmp_path / "gone.wav", stable_s=0.1)
    empty = tmp_path / "empty.wav"
    empty.touch()
    assert not watch.wait_until_stable(empty, stable_s=0.1, poll_s=0.05, timeout_s=0.5)


def test_process_file_analyzes_once(
    paths: Paths, settings: Settings, fx: dict[str, Path], notifications
) -> None:
    dropped = paths.inbox / "house" / "Night Drive v2.wav"
    shutil.copy(fx["loop"], dropped)
    event = watch.process_file(dropped, paths.inbox, paths, settings)
    assert event.analysis is not None and event.mastered is None and event.error is None
    assert event.analysis.manifest.kind == "analysis_only"
    assert event.analysis.manifest.track.title == "Night Drive"
    assert event.analysis.manifest.track.genre == "house"

    title, body, url = watch.describe(event, settings)
    assert title == "Analyzed: Night Drive" and "LUFS" in body
    assert url == f"http://127.0.0.1:7860/?run={event.analysis.manifest.id}"

    again = watch.process_file(dropped, paths.inbox, paths, settings)
    assert again.skipped == "already analyzed" and again.analysis is None


def test_auto_master_is_opt_in_and_needs_a_default_reference(
    paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path], notifications
) -> None:
    auto = replace(settings, watch_auto_master=True)
    hip_hop = paths.inbox / "hip-hop" / "loop.wav"
    shutil.copy(fx["loop"], hip_hop)
    event = watch.process_file(hip_hop, paths.inbox, paths, auto)
    assert event.mastered is not None and event.mastered.manifest.kind == "master"
    assert watch.describe(event, auto)[0] == "Mastered: loop"

    house = paths.inbox / "house" / "sweep.wav"
    shutil.copy(fx["sine_sweep"], house)
    event = watch.process_file(house, paths.inbox, paths, auto)
    assert event.analysis is not None and event.mastered is None, "house has no reference"


def test_unreadable_file_is_reported_not_raised(paths: Paths, settings: Settings) -> None:
    broken = paths.inbox / "hip-hop" / "broken.wav"
    broken.write_bytes(b"RIFF nonsense")
    event = watch.process_file(broken, paths.inbox, paths, settings)
    assert event.error and event.analysis is None
    assert watch.describe(event, settings)[0] == "MasterBench: broken.wav"


def test_watcher_picks_up_new_and_existing_files(
    paths: Paths, fast: Settings, fx: dict[str, Path], notifications
) -> None:
    shutil.copy(fx["vocal"], paths.inbox / "house" / "before.wav")
    (paths.inbox / "house" / "notes.txt").write_text("not audio")
    events: list[WatchEvent] = []
    watcher = InboxWatcher(paths, fast, on_event=events.append)
    watcher.start()
    try:
        shutil.copy(fx["loop"], paths.inbox / "hip-hop" / "after.wav")
        deadline = time.monotonic() + 20
        while len(events) < 2 and time.monotonic() < deadline:
            time.sleep(0.1)
        watcher.drain()
    finally:
        watcher.stop()

    assert {(e.path.name, e.genre) for e in events} == {
        ("before.wav", "house"),
        ("after.wav", "hip-hop"),
    }
    assert all(e.analysis is not None for e in events)
    assert {title for title, _, _ in notifications} == {"Analyzed: before", "Analyzed: after"}
    with index.open_index(paths) as conn:
        assert len(index.recent_runs(conn, kind="analysis_only")) == 2
