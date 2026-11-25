"""Watch folder: analyze bounces as they land in the Inbox (spec 5.19)."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from masterbench import audio_io, index, notify, summary
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.pipeline import MasterOptions, RunResult
from masterbench.references import library

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatchEvent:
    path: Path
    genre: str | None
    analysis: RunResult | None = None
    mastered: RunResult | None = None
    skipped: str | None = None
    error: str | None = None


def genre_for(path: Path, root: Path) -> str | None:
    """The first folder under the watched root names the genre: Inbox/house/x.wav -> house."""
    relative = path.resolve().relative_to(root.resolve())
    return relative.parts[0] if len(relative.parts) > 1 else None


def wait_until_stable(
    path: Path, stable_s: float, poll_s: float = 0.25, timeout_s: float = 900.0
) -> bool:
    """Block until the file size stops changing. Logic writes a bounce in chunks."""
    deadline = time.monotonic() + timeout_s
    last_size, unchanged_since = -1, time.monotonic()
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size != last_size:
            last_size, unchanged_since = size, time.monotonic()
        elif size > 0 and time.monotonic() - unchanged_since >= stable_s:
            return True
        time.sleep(poll_s)
    return False


def process_file(path: Path, root: Path, paths: Paths, settings: Settings) -> WatchEvent:
    """Analyze one new bounce, and master it too when auto-mastering is on."""
    genre = genre_for(path, root)
    try:
        with index.open_index(paths) as conn:
            if index.input_already_run(conn, audio_io.file_sha256(path), "analysis_only"):
                return WatchEvent(path, genre, skipped="already analyzed")
        analysis = pipeline.analyze(path, paths, settings, genre=genre)
        mastered = None
        if settings.watch_auto_master and genre:
            if not library.list_references(paths, genre):
                log.info("no reference for %s; not auto-mastering %s", genre, path.name)
            else:
                spec = library.default_spec(settings, genre)
                options = MasterOptions.from_settings(settings, spec, genre=genre)
                mastered = pipeline.master(path, options, paths, settings)
        return WatchEvent(path, genre, analysis, mastered)
    except Exception as exc:  # one bad file must not stop the watcher
        log.exception("failed to process %s", path)
        return WatchEvent(path, genre, error=str(exc))


def describe(event: WatchEvent, settings: Settings) -> tuple[str, str, str | None]:
    """Notification title, body and the link that opens the run in the UI."""
    result = event.mastered or event.analysis
    if result is None:
        return f"MasterBench: {event.path.name}", event.error or event.skipped or "", None
    verb = "Mastered" if event.mastered else "Analyzed"
    url = f"http://127.0.0.1:{settings.ui_port}/?run={result.manifest.id}"
    return f"{verb}: {result.manifest.track.title}", summary.headline(result), url


def announce(event: WatchEvent, settings: Settings) -> None:
    if not event.skipped:
        notify.notify(*describe(event, settings))


class _Handler(FileSystemEventHandler):
    def __init__(self, enqueue: Callable[[Path], None]) -> None:
        self._enqueue = enqueue

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._enqueue(Path(str(event.src_path)))

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._enqueue(Path(str(event.dest_path)))


class InboxWatcher:
    """Watches the Inbox (and any extra folders) and processes new audio files one at a time."""

    def __init__(
        self,
        paths: Paths,
        settings: Settings,
        extra_dirs: tuple[Path, ...] = (),
        on_event: Callable[[WatchEvent], None] | None = None,
    ) -> None:
        self.paths, self.settings = paths, settings
        self.roots = (paths.inbox, *extra_dirs)
        self._on_event = on_event or (lambda event: None)
        self._queue: queue.Queue[Path | None] = queue.Queue()
        self._pending: set[Path] = set()
        self._lock = threading.Lock()
        self._observer = Observer()
        self._worker = threading.Thread(target=self._work, name="masterbench-watch", daemon=True)

    def _root_of(self, path: Path) -> Path | None:
        resolved = path.resolve()
        return next((r for r in self.roots if resolved.is_relative_to(r.resolve())), None)

    def enqueue(self, path: Path) -> None:
        if not audio_io.is_audio(path) or path.name.startswith("."):
            return
        with self._lock:
            if path in self._pending:
                return
            self._pending.add(path)
        self._queue.put(path)

    def _work(self) -> None:
        while (path := self._queue.get()) is not None:
            try:
                root = self._root_of(path)
                if root is None or not wait_until_stable(path, self.settings.watch_stable_s):
                    continue
                event = process_file(path, root, self.paths, self.settings)
                announce(event, self.settings)
                self._on_event(event)
            finally:
                with self._lock:
                    self._pending.discard(path)
                self._queue.task_done()

    def scan_existing(self) -> None:
        """Queue files that arrived while the watcher was not running."""
        for root in self.roots:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    self.enqueue(path)

    def start(self) -> None:
        for root in self.roots:
            root.mkdir(parents=True, exist_ok=True)
            self._observer.schedule(_Handler(self.enqueue), str(root), recursive=True)
        self._observer.start()
        self._worker.start()
        self.scan_existing()

    def drain(self) -> None:
        """Block until every queued file has been handled."""
        self._queue.join()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()
        self._queue.put(None)
        self._worker.join()
