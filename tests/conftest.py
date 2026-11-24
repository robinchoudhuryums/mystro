from __future__ import annotations

import json
from pathlib import Path

import anthropic
import httpx2
import numpy as np
import pytest

from masterbench import config, fixtures
from masterbench.config import Paths, Settings
from masterbench.references import library


@pytest.fixture(scope="session")
def fx(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generated audio fixtures, written once per test session."""
    return fixtures.write_all(tmp_path_factory.mktemp("fixtures"))


@pytest.fixture(scope="session")
def truth(fx: dict[str, Path]) -> dict:
    return json.loads(fx["truth"].read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def isolated_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Tests never read or write the real machine's state folder or data root pointer."""
    state = tmp_path / "state"
    monkeypatch.setenv("MASTERBENCH_STATE", str(state))
    monkeypatch.delenv("MASTERBENCH_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return state


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "MasterBench"
    monkeypatch.setenv("MASTERBENCH_HOME", str(root))
    return root


@pytest.fixture
def paths(data_root: Path) -> Paths:
    paths = config.get_paths()
    paths.ensure()
    return paths


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def reference(paths: Paths, settings: Settings, fx: dict[str, Path]) -> library.Reference:
    """The fixture reference, added to the library as the hip-hop default."""
    return library.add(paths, fx["reference"], "hip-hop", settings)


def peak_hz(segment: np.ndarray, sr: float, low_hz: float, high_hz: float) -> float:
    """Frequency of the strongest spectral peak in a range (zero-padded, interpolated)."""
    n = 1 << 19
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    window = np.flatnonzero((freqs >= low_hz) & (freqs <= high_hz))
    k = window[np.argmax(spectrum[window])]
    a, b, c = np.log(spectrum[k - 1 : k + 2])
    return float(freqs[k] + 0.5 * (a - c) / (a - 2 * b + c) * (freqs[1] - freqs[0]))


def cents(freq_hz: float, ref_hz: float) -> float:
    return float(1200 * np.log2(freq_hz / ref_hz))


class FakeAPI:
    """An HTTP stand-in behind the real SDK client, so request building is exercised."""

    def __init__(self, reply: dict, status: int = 200, stop_reason: str = "end_turn") -> None:
        self.requests: list[httpx2.Request] = []
        self.reply, self.status, self.stop_reason = reply, status, stop_reason
        self.text: str | None = None

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.status != 200:
            error = {
                "type": "error",
                "error": {"type": "authentication_error", "message": "bad key"},
            }
            return httpx2.Response(self.status, json=error)
        body = json.loads(request.content)
        text = self.text if self.text is not None else json.dumps(self.reply)
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {"type": "text", "text": text},
                ],
                "stop_reason": self.stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 4000, "output_tokens": 1500},
            },
        )

    def client(self) -> anthropic.Anthropic:
        http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(self))
        return anthropic.Anthropic(api_key="test-key", http_client=http, max_retries=0)

    def body(self, number: int = -1) -> dict:
        return json.loads(self.requests[number].content)
