"""Desktop notifications. macOS only; elsewhere this is a no-op that reports False."""

from __future__ import annotations

import shutil
import subprocess
import sys


def _applescript_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def notify(title: str, message: str, url: str | None = None) -> bool:
    """Show a notification. With terminal-notifier installed, clicking it opens the URL."""
    if sys.platform != "darwin":
        return False
    if url and shutil.which("terminal-notifier"):
        cmd = ["terminal-notifier", "-title", title, "-message", message, "-open", url]
    else:
        script = (
            f"display notification {_applescript_string(message)}"
            f" with title {_applescript_string(title)}"
        )
        cmd = ["osascript", "-e", script]
    return subprocess.run(cmd, capture_output=True).returncode == 0
