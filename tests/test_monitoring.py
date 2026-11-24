from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from masterbench import index, monitoring
from masterbench.config import Paths, Settings

START = datetime(2026, 11, 24, 18, 0, tzinfo=UTC)


def test_calibration_noise_is_at_the_reference_level() -> None:
    noise = monitoring.calibration_noise(seconds=10.0)
    rms = 10 * np.log10(np.mean(noise**2))
    assert rms == pytest.approx(monitoring.CALIBRATION_RMS_DBFS, abs=0.01)
    assert np.array_equal(noise[:, 0], noise[:, 1]), "the same in both speakers"
    freqs, power = signal.welch(noise[:, 0], monitoring.CALIBRATION_RATE, nperseg=8192)
    inside = power[(freqs >= 500) & (freqs <= 2000)].sum()
    assert inside / power.sum() > 0.99, "band-limited to 500 Hz - 2 kHz"
    assert np.abs(noise).max() < 1.0 and abs(noise[0, 0]) < 1e-6, "no clipping, faded in"
    assert np.array_equal(noise, monitoring.calibration_noise(seconds=10.0))


def test_calibration_file_is_written_once(paths: Paths, settings: Settings) -> None:
    file = monitoring.calibration_file(paths)
    info = sf.info(file)
    assert (info.samplerate, info.subtype, info.channels) == (48000, "PCM_24", 2)
    assert info.duration == pytest.approx(30.0)
    stamp = file.stat().st_mtime_ns
    assert monitoring.calibration_file(paths) == file and file.stat().st_mtime_ns == stamp
    steps = monitoring.calibration_steps(settings)
    assert "78 dB SPL" in steps and "-20 dBFS RMS" in steps
    assert "70 dB SPL" in monitoring.calibration_steps(replace(settings, spl_target_db=70.0))


def test_session_timing_and_break_reminders(paths: Paths, settings: Settings) -> None:
    session = monitoring.start_session(paths, START)
    assert session.minutes == 0 and not session.calibrated
    assert not monitoring.break_due(session, settings)

    session = monitoring.heartbeat(paths, session.id, START + timedelta(minutes=49))
    assert session.minutes == 49 and not monitoring.break_due(session, settings)
    session = monitoring.heartbeat(paths, session.id, START + timedelta(minutes=51))
    assert monitoring.break_due(session, settings)
    assert "51 minutes without a break" in monitoring.status_text(session, settings)
    assert monitoring.break_due(session, replace(settings, break_interval_min=60)) is False

    session = monitoring.take_break(paths, session.id, START + timedelta(minutes=62))
    assert session.minutes_since_break == 0 and not monitoring.break_due(session, settings)
    session = monitoring.heartbeat(paths, session.id, START + timedelta(minutes=100))
    assert session.minutes == 100 and session.minutes_since_break == 38

    session = monitoring.mark_calibrated(paths, session.id)
    assert session.calibrated and "level calibrated" in monitoring.status_text(session, settings)
    assert monitoring.load_session(paths, session.id) == session
    for bad in ("nope", "../escape"):
        with pytest.raises(LookupError):
            monitoring.load_session(paths, bad)


def test_sessions_are_indexed_from_their_files(paths: Paths) -> None:
    first = monitoring.start_session(paths, START)
    monitoring.heartbeat(paths, first.id, START + timedelta(minutes=30))
    second = monitoring.start_session(paths, START + timedelta(days=1))
    monitoring.mark_calibrated(paths, second.id)
    assert [s.id for s in monitoring.all_sessions(paths)] == [first.id, second.id]
    with index.open_index(paths) as conn:
        assert index.reindex(conn, paths)["sessions"] == 2
        rows = conn.execute(
            "SELECT calibrated, minutes FROM sessions ORDER BY started_at"
        ).fetchall()
    assert [tuple(r) for r in rows] == [(0, 30.0), (1, 0.0)]


def test_fresh_ears_checks_wait_for_a_later_session(paths: Paths) -> None:
    today = monitoring.start_session(paths, START)
    check = monitoring.queue_fresh_ears(
        paths, today.id, "night-drive", ["run-a", "run-b"], "v3 or v4?"
    )
    assert check.run_ids == ["run-a", "run-b"]
    assert monitoring.fresh_ears_due(paths, today.id) == [], "not with the ears that made the pick"

    tomorrow = monitoring.start_session(paths, START + timedelta(days=1))
    assert monitoring.fresh_ears_due(paths, tomorrow.id) == [check]
    monitoring.queue_fresh_ears(paths, today.id, "night-drive", ["run-c"])
    (latest,) = monitoring.fresh_ears_due(paths, tomorrow.id)
    assert latest.run_ids == ["run-c"], "one pending check per track"
    monitoring.finish_fresh_ears(paths, "night-drive")
    assert monitoring.fresh_ears_due(paths, tomorrow.id) == []
    with pytest.raises(ValueError, match="at least one run"):
        monitoring.queue_fresh_ears(paths, today.id, "night-drive", [])
