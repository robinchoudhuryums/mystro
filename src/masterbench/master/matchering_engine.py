"""Matchering wrapper: match a mix to a reference and hand back unlimited float audio."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import matchering as mg
import numpy as np
import soundfile as sf

MAX_LENGTH_S = 20 * 60


class MasteringError(RuntimeError):
    pass


def match(
    target: Path, reference: Path, sample_rate: int, log: Callable[[str], None] | None = None
) -> np.ndarray:
    """Match target to reference at the target's sample rate.

    Matchering's own limiter and normalization are off, so the result can exceed full scale.
    It is returned as float (frames, 2) and must go through our limiter before export.
    """
    config = mg.Config(internal_sample_rate=sample_rate, max_length=MAX_LENGTH_S)
    mg.log(info_handler=log, warning_handler=log)
    with tempfile.TemporaryDirectory(prefix="masterbench-") as tmp:
        out = Path(tmp) / "matched.wav"
        result = mg.Result(str(out), subtype="FLOAT", use_limiter=False, normalize=False)
        try:
            mg.process(
                target=str(target), reference=str(reference), results=[result], config=config
            )
        except Exception as exc:  # Matchering raises its own ModuleError for bad inputs
            raise MasteringError(f"Matchering failed: {exc}") from exc
        finally:
            mg.log()
        audio, _ = sf.read(out, dtype="float64", always_2d=True)
    return audio
