"""Rebalance: per-stem gain and optional chain, re-summed with headroom for mastering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from masterbench import audio_io, projects
from masterbench.config import Paths
from masterbench.master import chain, presets
from masterbench.stems.separate import StemError, StemSet

MAX_GAIN_DB = 6.0
HEADROOM_DBFS = -1.0


@dataclass(frozen=True)
class Rebalanced:
    path: Path
    gains_db: dict[str, float]
    chains: dict[str, str]
    trim_db: float
    peak_dbfs: float


def rebalance(
    stem_set: StemSet,
    gains_db: dict[str, float],
    paths: Paths,
    chains: dict[str, str] | None = None,
) -> Rebalanced:
    """Write a new mix from the stems. Gains are limited to ±6 dB per stem.

    If the new sum would peak above -1 dBFS the whole mix is turned down to fit; nothing is
    clipped or limited here, that is the mastering stage's job.
    """
    chains = chains or {}
    unknown = sorted((set(gains_db) | set(chains)) - set(stem_set.stems))
    if unknown:
        raise StemError(f"no such stems: {', '.join(unknown)}; have {', '.join(stem_set.stems)}")
    too_much = {name: gain for name, gain in gains_db.items() if abs(gain) > MAX_GAIN_DB}
    if too_much:
        raise StemError(f"stem gains are limited to ±{MAX_GAIN_DB:g} dB: {too_much}")

    total = np.zeros((stem_set.frames, 2))
    for name in stem_set.stems:
        audio = stem_set.load(name) * 10 ** (gains_db.get(name, 0.0) / 20)
        if name in chains:
            audio = chain.run(presets.get(paths, chains[name]), audio, stem_set.sample_rate)
        total += audio

    peak = float(np.max(np.abs(total)))
    ceiling = 10 ** (HEADROOM_DBFS / 20)
    trim_db = min(0.0, 20 * np.log10(ceiling / peak)) if peak > 0 else 0.0
    total *= 10 ** (trim_db / 20)

    settings = {"gains_db": gains_db, "chains": chains}
    tag = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    path = audio_io.save_wav(
        stem_set.dir / f"rebalanced-{tag}.wav", total, stem_set.sample_rate, 24
    )
    result = Rebalanced(
        path=path,
        gains_db=dict(gains_db),
        chains=dict(chains),
        trim_db=round(float(trim_db), 2),
        peak_dbfs=round(float(20 * np.log10(peak)) + float(trim_db), 2) if peak > 0 else -120.0,
    )
    projects.write_json(
        path.with_suffix(".json"),
        {**settings, "stem_set": stem_set.name, "trim_db": result.trim_db},
    )
    return result
