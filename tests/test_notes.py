from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import anthropic
import httpx2
import pytest

from masterbench import index
from masterbench.config import Paths, Settings
from masterbench.master import pipeline
from masterbench.master.pipeline import MasterOptions, RunResult
from masterbench.notes import mix_notes
from masterbench.notes.mix_notes import NotesError
from masterbench.references.library import Reference

REPLY = {
    "summary": "A balanced mix that is light in the sub.",
    "issues": [
        {
            "title": "Sub sits under the reference",
            "impact": "high",
            "what": "The lowest octave is quieter than the reference at equal loudness.",
            "evidence": "band_deltas_db.sub is -0.66 dB.",
            "fix_in_mix": "Raise the 808 or shelve it up in Channel EQ.",
            "fix_in_master": "A low shelf can add a little, at the cost of headroom.",
        }
    ],
    "already_good": ["True peak is under the ceiling."],
    "caveats": ["No stems were available."],
}


class FakeAPI:
    """An HTTP stand-in behind the real SDK client, so request building is exercised."""

    def __init__(
        self, reply: dict | None = None, status: int = 200, stop_reason: str = "end_turn"
    ) -> None:
        self.requests: list[httpx2.Request] = []
        self.reply, self.status, self.stop_reason = reply or REPLY, status, stop_reason
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


@pytest.fixture
def run(paths: Paths, settings: Settings, reference: Reference, fx: dict[str, Path]) -> RunResult:
    options = MasterOptions.from_settings(settings, "hip-hop/default", post_chain="warm")
    return pipeline.master(fx["loop"], options, paths, settings)


def test_payload_is_measurements_and_labels_only(run: RunResult, paths: Paths) -> None:
    payload = mix_notes.build_payload(run, "  Dark and heavy, vocals up front.  ")
    text = mix_notes.payload_json(payload)
    assert str(paths.root) not in text and "/tmp" not in text and ".wav" not in text
    assert "sha256" not in text and "source_path" not in text
    assert payload["track"] == {"title": "loop", "genre": "hip-hop"}
    assert payload["intent"] == "Dark and heavy, vocals up front."
    assert set(payload["analyses"]) == {"master", "mix", "reference"}
    mix = payload["analyses"]["mix"]
    assert set(mix) >= {
        "file",
        "loudness",
        "bands_db",
        "band_deltas_db",
        "stereo",
        "spectrum",
        "flags",
    }
    assert len(mix["spectrum"]["freqs_hz"]) == 30
    assert payload["processing"]["chains"] == {"post_chain": ["low_shelf", "saturation"]}
    assert (
        payload["processing"]["target_lufs"] == -14.0
        and "seconds" not in payload["processing"]["stats"]
    )
    assert "intent" not in mix_notes.build_payload(run)
    assert len(text) < 20_000, "small enough to cost about a cent to send"


def test_request_goes_out_as_structured_output_with_refusal_fallback(
    run: RunResult, paths: Paths, settings: Settings
) -> None:
    api = FakeAPI()
    notes = mix_notes.generate(run, paths, settings, "Dark and heavy.", client=api.client())

    (request,) = api.requests
    assert (
        request.url.path == "/v1/messages" and request.headers["authorization"] == "Bearer test-key"
    )
    assert mix_notes.FALLBACK_BETA in request.headers["anthropic-beta"]
    body = api.body()
    assert body["model"] == "claude-sonnet-5-5" and body["fallbacks"] == "default"
    assert body["max_tokens"] == 16000 and "thinking" not in body and "temperature" not in body
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"] == {
        "type": "json_schema",
        "schema": mix_notes.NOTES_SCHEMA,
    }
    assert "invented or misremembered figure" in body["system"]
    assert not body["system"].startswith("<!--"), "the version marker is not sent"
    (message,) = body["messages"]
    assert json.loads(message["content"]) == mix_notes.build_payload(run, "Dark and heavy.")

    assert notes.body == REPLY and not notes.cached and notes.served_by == "claude-sonnet-5-5"
    assert notes.cost_usd == pytest.approx((4000 * 2.00 + 1500 * 10.00) / 1e6)
    assert mix_notes.load(run) == notes
    assert (
        "### 1. Sub sits under the reference (high impact)"
        in (run.run_dir / "notes.md").read_text()
    )


def test_same_measurements_reuse_the_stored_answer(run: RunResult, paths: Paths, settings) -> None:
    api = FakeAPI()
    first = mix_notes.generate(run, paths, settings, client=api.client())
    again = mix_notes.generate(run, paths, settings, client=api.client())
    assert len(api.requests) == 1 and again.cached and again.cost_usd == 0.0
    assert again.body == first.body and "stored answer, no charge" in mix_notes.render(again)
    mix_notes.generate(run, paths, settings, "Different intent.", client=api.client())
    mix_notes.generate(run, paths, settings, model="claude-haiku-5-5", client=api.client())
    assert len(api.requests) == 3, "a new intent or model is a new question"


def test_haiku_uses_the_plain_endpoint_without_fallbacks(
    run: RunResult, paths: Paths, settings
) -> None:
    api = FakeAPI()
    notes = mix_notes.generate(run, paths, settings, model="claude-haiku-5-5", client=api.client())
    assert "fallbacks" not in api.body() and "anthropic-beta" not in api.requests[0].headers
    assert notes.cost_usd == pytest.approx((4000 * 0.10 + 1500 * 0.50) / 1e6)


def test_monthly_cap_blocks_before_anything_is_sent(run: RunResult, paths: Paths, settings) -> None:
    api = FakeAPI()
    tight = replace(settings, notes_monthly_cap_usd=0.01)
    with pytest.raises(NotesError, match="would pass the monthly cap"):
        mix_notes.generate(run, paths, tight, client=api.client())
    assert api.requests == [] and mix_notes.load(run) is None
    forced = mix_notes.generate(run, paths, tight, client=api.client(), force=True)
    assert len(api.requests) == 1 and mix_notes.spent_this_month(paths) == forced.cost_usd

    quote = mix_notes.estimate(mix_notes.build_payload(run), "claude-sonnet-5-5", paths, tight)
    assert quote.over_cap and 0.02 < quote.cost_usd < 0.08 and quote.spent_this_month_usd > 0
    with pytest.raises(NotesError, match="no price known for model 'gpt-9'"):
        mix_notes.generate(run, paths, settings, model="gpt-9", client=api.client())


@pytest.mark.parametrize(
    ("api", "message"),
    [
        (FakeAPI(stop_reason="refusal"), "declined"),
        (FakeAPI(stop_reason="max_tokens"), "cut off"),
        (FakeAPI(status=401), "rejected the key"),
    ],
)
def test_failed_requests_save_nothing(run: RunResult, paths: Paths, settings, api, message) -> None:
    with pytest.raises(NotesError, match=message):
        mix_notes.generate(run, paths, settings, client=api.client())
    assert mix_notes.load(run) is None and mix_notes.spent_this_month(paths) == 0.0


def test_invalid_json_reply_is_an_error(run: RunResult, paths: Paths, settings) -> None:
    api = FakeAPI()
    api.text = "Sure! Here are some notes."
    with pytest.raises(NotesError, match="not valid JSON"):
        mix_notes.generate(run, paths, settings, client=api.client())


def test_notes_are_indexed_and_survive_reindex(run: RunResult, paths: Paths, settings) -> None:
    notes = mix_notes.generate(run, paths, settings, client=FakeAPI().client())
    paths.db.unlink(missing_ok=True)
    with index.open_index(paths) as conn:
        assert index.reindex(conn, paths)["notes"] == 1
        row = conn.execute("SELECT run_id, model, cost_usd, payload_hash FROM notes").fetchone()
    assert tuple(row) == (run.manifest.id, notes.model, notes.cost_usd, notes.payload_hash)


def test_key_is_found_without_being_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")
    assert mix_notes.key_status() == "ANTHROPIC_API_KEY environment variable"
    assert mix_notes.make_client().api_key == "sk-secret-value"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert "none found" in mix_notes.key_status()


def test_prompt_is_versioned() -> None:
    prompt, version = mix_notes.system_prompt()
    assert version == 2 and "Logic Pro" in prompt
    assert mix_notes.payload_hash({}, "m", 1) != mix_notes.payload_hash({}, "m", 2)
