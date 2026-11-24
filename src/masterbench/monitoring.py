"""Listening level and breaks: a calibration tone, session timing, and fresh-ears re-checks.

Judgement drifts with level and with fatigue. Setting the monitors to the same level every
session, stopping regularly, and re-checking yesterday's picks with fresh ears all make
listening decisions more repeatable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from masterbench import audio_io, projects
from masterbench.config import Paths, Settings

CALIBRATION_RMS_DBFS = -20.0
CALIBRATION_SECONDS = 30.0
CALIBRATION_RATE = 48000
QUEUE_FILE = "fresh_ears.json"


@dataclass(frozen=True)
class Session:
    id: str
    started_at: str
    last_seen_at: str
    calibrated: bool = False
    last_break_at: str | None = None

    @property
    def minutes(self) -> float:
        return round((_parse(self.last_seen_at) - _parse(self.started_at)).total_seconds() / 60, 1)

    @property
    def minutes_since_break(self) -> float:
        since = _parse(self.last_break_at or self.started_at)
        return round((_parse(self.last_seen_at) - since).total_seconds() / 60, 1)


@dataclass(frozen=True)
class FreshEarsCheck:
    track_slug: str
    run_ids: list[str]
    queued_at: str
    session_id: str
    note: str = ""


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _now(now: datetime | None = None) -> str:
    moment = now or datetime.now(UTC)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sessions_dir(paths: Paths) -> Path:
    return paths.root / "sessions"


def _save(paths: Paths, session: Session) -> Session:
    projects.write_json(sessions_dir(paths) / f"{session.id}.json", asdict(session))
    return session


def start_session(paths: Paths, now: datetime | None = None) -> Session:
    stamp = _now(now)
    return _save(paths, Session(projects.new_run_id(now), stamp, stamp))


def load_session(paths: Paths, session_id: str) -> Session:
    file = sessions_dir(paths) / f"{session_id}.json"
    if not file.is_file() or not session_id.replace("-", "").isalnum():
        raise LookupError(f"no session {session_id}")
    return Session(**json.loads(file.read_text(encoding="utf-8")))


def all_sessions(paths: Paths) -> list[Session]:
    folder = sessions_dir(paths)
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    return [Session(**json.loads(f.read_text(encoding="utf-8"))) for f in files]


def heartbeat(paths: Paths, session_id: str, now: datetime | None = None) -> Session:
    """Record that the session is still going. Called once a minute by the UI."""
    return _save(paths, replace(load_session(paths, session_id), last_seen_at=_now(now)))


def mark_calibrated(paths: Paths, session_id: str) -> Session:
    return _save(paths, replace(load_session(paths, session_id), calibrated=True))


def take_break(paths: Paths, session_id: str, now: datetime | None = None) -> Session:
    stamp = _now(now)
    session = load_session(paths, session_id)
    return _save(paths, replace(session, last_seen_at=stamp, last_break_at=stamp))


def break_due(session: Session, settings: Settings) -> bool:
    return session.minutes_since_break >= settings.break_interval_min


def status_text(session: Session, settings: Settings) -> str:
    level = "level calibrated" if session.calibrated else "level not calibrated this session"
    text = f"Session: {session.minutes:.0f} min, {level}."
    if break_due(session, settings):
        text += (
            f" **{session.minutes_since_break:.0f} minutes without a break.** Ears adapt to "
            "whatever they hear; ten minutes of quiet resets them."
        )
    return text


def calibration_noise(
    seconds: float = CALIBRATION_SECONDS, sample_rate: int = CALIBRATION_RATE
) -> np.ndarray:
    """Mono-compatible pink noise at -20 dBFS RMS, band-limited to 500 Hz - 2 kHz.

    Band-limited pink noise is the usual signal for setting a monitoring level: it avoids room
    modes in the bass and the directional top end, so a phone's SPL meter reads it steadily.
    """
    rng = np.random.default_rng(20)
    count = int(seconds * sample_rate)
    spectrum = np.fft.rfft(rng.standard_normal(count))
    freqs = np.fft.rfftfreq(count, 1 / sample_rate)
    shape = np.where((freqs >= 500) & (freqs <= 2000), 1 / np.sqrt(np.maximum(freqs, 1.0)), 0.0)
    noise = np.fft.irfft(spectrum * shape, count)
    fade = int(0.5 * sample_rate)
    noise[:fade] *= np.linspace(0, 1, fade)
    noise[-fade:] *= np.linspace(1, 0, fade)
    noise *= 10 ** (CALIBRATION_RMS_DBFS / 20) / np.sqrt(np.mean(noise**2))
    return np.column_stack([noise, noise])


def calibration_file(paths: Paths) -> Path:
    """The calibration tone as a WAV in the data root, written once."""
    file = paths.root / "calibration" / "pink_noise_-20dBFS.wav"
    if not file.exists():
        audio_io.save_wav(file, calibration_noise(), CALIBRATION_RATE, 24)
    return file


def calibration_steps(settings: Settings) -> str:
    target = settings.spl_target_db
    return (
        f"1. Set your interface and system volume where you usually work.\n"
        f"2. Play the noise: pink noise from 500 Hz to 2 kHz at "
        f"{CALIBRATION_RMS_DBFS:.0f} dBFS RMS.\n"
        f"3. Hold a phone with an SPL meter app (C-weighted, slow) where your head is.\n"
        f"4. Turn the monitor knob until it reads about {target:.0f} dB SPL, then leave it "
        f"there. On headphones there is no room to measure: set a level that is comfortable "
        f"for this noise and mark the knob.\n"
        f"5. Mark the session as calibrated. A phone app is only accurate to a few dB; what "
        f"matters is using the same level every session."
    )


def _queue(paths: Paths) -> list[dict[str, Any]]:
    file = paths.root / QUEUE_FILE
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else []


def queue_fresh_ears(
    paths: Paths, session_id: str, track_slug: str, run_ids: list[str], note: str = ""
) -> FreshEarsCheck:
    """Put versions aside for a blind comparison in a later session."""
    if len(run_ids) < 1:
        raise ValueError("choose at least one run to re-check")
    check = FreshEarsCheck(track_slug, list(run_ids), _now(), session_id, note)
    kept = [c for c in _queue(paths) if c["track_slug"] != track_slug]
    projects.write_json(paths.root / QUEUE_FILE, [*kept, asdict(check)])
    return check


def fresh_ears_due(paths: Paths, session_id: str) -> list[FreshEarsCheck]:
    """Checks queued in an earlier session: the ones fresh ears can now judge."""
    return [FreshEarsCheck(**c) for c in _queue(paths) if c["session_id"] != session_id]


def finish_fresh_ears(paths: Paths, track_slug: str) -> None:
    kept = [c for c in _queue(paths) if c["track_slug"] != track_slug]
    projects.write_json(paths.root / QUEUE_FILE, kept)
