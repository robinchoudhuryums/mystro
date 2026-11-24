"""Runs the Phase 0 verification spike's offline checks, so a dependency bump that breaks
an API we rely on fails the test suite rather than a mastering run."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_stack.py"


@pytest.mark.slow
def test_offline_checks_pass(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("verify_stack", SCRIPT)
    assert spec and spec.loader
    spike = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = spike
    spec.loader.exec_module(spike)

    names = [name for name in spike.CHECKS if name not in spike.NETWORK_CHECKS]
    outcomes = spike.run_checks(names, tmp_path)
    failures = [f"{o.name}: {o.detail}" for o in outcomes if o.status == "FAIL"]
    assert not failures, "\n".join(failures)
    passed = {o.name for o in outcomes if o.status == "PASS"}
    assert {"matchering", "pedalboard", "loudness", "librosa", "mcp", "codecs"} <= passed
