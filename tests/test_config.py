from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from masterbench import config


def test_defaults_match_streaming_targets(data_root: Path) -> None:
    settings = config.load_settings()
    assert settings.targets.lufs_i == -14.0
    assert settings.targets.true_peak_dbtp == -1.0
    assert settings.watch_auto_master is False
    assert settings.mcp_allow_writes is False


def test_data_root_env_override_and_layout(data_root: Path) -> None:
    paths = config.get_paths()
    assert paths.root == data_root
    paths.ensure()
    assert (data_root / "Inbox" / "hip-hop").is_dir()
    assert (data_root / "references" / "house").is_dir()
    assert paths.chain_presets == data_root / "presets" / "chains"
    assert paths.db == data_root / "masterbench.db"


def test_settings_toml_overrides_nested_values(data_root: Path) -> None:
    data_root.mkdir()
    (data_root / "settings.toml").write_text(
        'notes_model = "claude-haiku-5-5"\n[targets]\nlufs_i = -10\n', encoding="utf-8"
    )
    settings = config.load_settings()
    assert settings.notes_model == "claude-haiku-5-5"
    assert settings.targets.lufs_i == -10.0 and isinstance(settings.targets.lufs_i, float)
    assert settings.targets.true_peak_dbtp == -1.0


@pytest.mark.parametrize(
    "toml",
    ["typo_key = 1\n", "[targets]\nlufs = -9\n", 'watch_auto_master = "yes"\n', "targets = 3\n"],
)
def test_bad_settings_are_rejected(data_root: Path, toml: str) -> None:
    data_root.mkdir()
    (data_root / "settings.toml").write_text(toml, encoding="utf-8")
    with pytest.raises(ValueError, match="setting"):
        config.load_settings()


def test_bands_are_contiguous() -> None:
    edges = list(config.BANDS.values())
    assert edges[0][0] == 20.0 and edges[-1][1] == 20000.0
    assert all(a[1] == b[0] for a, b in itertools.pairwise(edges))
