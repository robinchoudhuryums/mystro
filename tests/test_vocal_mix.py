from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pedalboard as pb
import pytest

from conftest import FakeAPI
from masterbench import audio_io, config, fixtures, index, summary
from masterbench.analysis import detectors, loudness
from masterbench.config import Paths, Settings, Thresholds
from masterbench.notes import mix_notes
from masterbench.references import library
from masterbench.stems import workflow as stem_workflow
from masterbench.ui import vocal_tab
from masterbench.vocal import dynamics, mix_assistant
from masterbench.vocal.mix_assistant import MixError, MixOptions, VocalChain

SR = fixtures.SR
TARGET = -2.0
REPLY = {
    "summary": "The vocal sits where it was asked to.",
    "issues": [
        {
            "title": "Hooks needed four dB of lift",
            "impact": "medium",
            "what": "The rider raised the vocal in both hooks.",
            "evidence": "vocal_mix.sections shows +4 dB of gain in the hooks.",
            "fix_in_mix": "Record or clip-gain the hooks hotter.",
            "fix_in_master": "Nothing to do in mastering.",
        }
    ],
    "already_good": ["Every section is within a dB of the target."],
    "caveats": [],
}


@pytest.fixture(scope="module")
def song():
    return fixtures.vocal_over_song()


@pytest.fixture(scope="module")
def sung(song):
    """The sections with singing in them, as (label, start, end)."""
    return [(s.label, s.start_s, s.end_s) for s in song[2] if s.label in ("verse", "hook")]


@pytest.fixture(scope="module")
def mixed(tmp_path_factory: pytest.TempPathFactory, song):
    """One full run: dry vocal and instrumental in, mastered track and AI notes out."""
    work = tmp_path_factory.mktemp("vocal-mix")
    paths = config.Paths(work / "MasterBench")
    paths.ensure()
    settings = Settings()
    vocal, beat, _ = song
    files = (
        fixtures.write_wav(work / "Night Drive vox.wav", vocal),
        fixtures.write_wav(work / "Night Drive.wav", beat),
    )
    library.add(
        paths,
        fixtures.write_wav(work / "ref.wav", fixtures.reference_master()),
        "hip-hop",
        settings,
    )
    api = FakeAPI(REPLY)
    stages: list[str] = []
    result = mix_assistant.mix(
        paths,
        settings,
        *files,
        MixOptions(genre="hip-hop", title="Night Drive"),
        progress=lambda stage, fraction: stages.append(stage),
        notes=True,
        intent="Vocal up front",
        client=api.client(),
    )
    return paths, result, api, stages


def section_ratio(vocal: np.ndarray, beat: np.ndarray, start_s: float, end_s: float) -> float:
    a, b = int(start_s * SR), int(end_s * SR)
    found = dynamics.ratio_db(vocal[a:b], beat[a:b], SR)
    assert found is not None
    return found


def band_db(audio: np.ndarray, band: tuple[float, float], start_s: float, end_s: float) -> float:
    part = dynamics.band_split(audio[int(start_s * SR) : int(end_s * SR)], SR, band)
    return float(10 * np.log10(np.mean(part**2) + 1e-20))


@pytest.mark.parametrize(("burst_db", "least"), [(0.0, 6.0), (-6.0, 3.5)])
def test_deesser_turns_down_planted_sibilance(burst_db: float, least: float) -> None:
    """Phase 9 criterion: the de-esser reduces planted sibilance by a measurable amount."""
    vocal, times = fixtures.sibilant_vocal(burst_db=burst_db)
    done = dynamics.deess(vocal, SR)
    assert done.events == 3 and done.max_reduction_db <= 8.0
    quiet = np.ones(len(vocal), dtype=bool)
    for at in times:
        before = band_db(vocal, dynamics.SIBILANCE_HZ, at, at + 0.12)
        after = band_db(done.audio, dynamics.SIBILANCE_HZ, at, at + 0.12)
        assert before - after > least, "each burst's 5-10 kHz comes down"
        assert before - after < 8.5, "and by no more than the limit"
        quiet[int((at - 0.02) * SR) : int((at + 0.16) * SR)] = False
    assert np.array_equal(done.audio[quiet], vocal[quiet]), "away from the bursts it is untouched"
    removed = vocal - done.audio
    low = dynamics.band_split(removed, SR, (100.0, 3000.0))
    assert np.sum(low**2) < 0.01 * np.sum(removed**2), "what it takes away is all top end"
    limits = Thresholds()
    assert len(detectors.sibilance_events(vocal, SR, limits)) == 3
    assert detectors.sibilance_events(done.audio, SR, limits) == []


def test_deesser_leaves_a_clean_vocal_alone() -> None:
    vocal, _ = fixtures.synthetic_vocal()
    done = dynamics.deess(vocal, SR)
    assert done.events == 0 and done.max_reduction_db == 0.0
    assert np.array_equal(done.audio, vocal)
    hissy, _ = fixtures.sibilant_vocal()
    assert np.array_equal(dynamics.deess(hissy, SR, max_reduction_db=0.0).audio, hissy)
    gentler = dynamics.deess(hissy, SR, max_reduction_db=3.0)
    assert 2.0 < gentler.max_reduction_db <= 3.0
    wide = dynamics.deess(np.stack([hissy, 0.5 * hissy], axis=1), SR)
    assert wide.audio.shape == (len(hissy), 2) and wide.events == 3
    silent = dynamics.deess(np.zeros(SR), SR)
    assert silent.events == 0 and not silent.audio.any()


def test_rider_holds_each_section_within_a_db_of_target(song, sung) -> None:
    """Phase 9 criterion: vocal-to-instrumental within ±1 dB of target in each section."""
    vocal, beat, _ = song
    before = [section_ratio(vocal, beat, a, b) for _, a, b in sung]
    assert max(before) - min(before) > 5, "as sung, verses sit 6 dB hotter than hooks"
    ridden = dynamics.ride(vocal, beat, SR, TARGET, sung)
    for (label, a, b), found in zip(sung, ridden.sections, strict=True):
        assert section_ratio(ridden.audio, beat, a, b) == pytest.approx(TARGET, abs=1.0), label
        assert found.after_db == pytest.approx(TARGET, abs=0.2) and not found.limited
        assert found.before_db is not None and found.gain_db is not None
    low, high = ridden.range_db
    assert -6.0 <= low < -2 and 2 < high <= 6.0, "it used the range it was given, and no more"
    assert np.abs(ridden.gain_db - ridden.fader_db).max() <= 6.01, "the fader is rounded"
    assert np.abs(np.diff(ridden.gain_db)).max() * SR < 10.0, "under 10 dB a second: no pumping"

    def spread(audio: np.ndarray) -> float:
        _, ratio, _ = dynamics._window_levels(audio, beat, SR)
        low, high = np.percentile(ratio[np.isfinite(ratio)], [5, 95])
        return float(high - low)

    assert spread(vocal) > 9 and spread(ridden.audio) < 3.5, "phrase to phrase it is evened out"


def test_rider_without_sections_and_at_its_limit(song, sung) -> None:
    vocal, beat, sections = song
    blind = dynamics.ride(vocal, beat, SR, TARGET)
    assert [s.label for s in blind.sections] == ["whole song"]
    for label, a, b in sung:
        assert section_ratio(blind.audio, beat, a, b) == pytest.approx(TARGET, abs=1.0), label

    tight = dynamics.ride(vocal, beat, SR, TARGET, sung, max_gain_db=2.0)
    assert all(s.limited for s in tight.sections) and tight.range_db == (-2.0, 2.0)
    assert max(abs((s.after_db or 0) - TARGET) for s in tight.sections) > 1.0
    assert tight.fader_db == blind.fader_db, "the fader does not depend on the rider's range"

    everything = [(s.label, s.start_s, s.end_s) for s in sections]
    full = dynamics.ride(vocal, beat, SR, TARGET, everything)
    assert [s.after_db is None for s in full.sections] == [True, False, False, False, False, True]
    assert full.sections[0].gain_db is None, "nothing is sung in the intro"

    silent = dynamics.ride(np.zeros(len(beat)), beat, SR, TARGET, sung)
    assert silent.fader_db == 0.0 and not silent.audio.any() and silent.sections == []
    louder = dynamics.ride(vocal, beat, SR, TARGET + 3.0, sung)
    assert louder.fader_db == pytest.approx(blind.fader_db + 3.0, abs=0.01)


def test_carve_dips_the_beat_only_while_the_vocal_is_there(song) -> None:
    vocal, beat, sections = song
    carved = dynamics.carve(beat, vocal, SR, depth_db=3.0)
    assert carved.max_dip_db == pytest.approx(3.0, abs=0.01) and carved.max_duck_db == 0.0
    assert carved.mean_dip_db == pytest.approx(3.0, abs=0.3)
    intro, hook = sections[0], sections[2]
    quiet_end = int((intro.end_s - 0.3) * SR)
    assert np.array_equal(carved.audio[:quiet_end], beat[:quiet_end]), "no vocal, no change"
    presence = dynamics.PRESENCE_HZ
    dip = band_db(beat, presence, hook.start_s + 1, hook.end_s - 1) - band_db(
        carved.audio, presence, hook.start_s + 1, hook.end_s - 1
    )
    assert dip == pytest.approx(2.8, abs=0.5), "about the asked 3 dB in 1-4 kHz"
    low = band_db(beat, (40.0, 400.0), hook.start_s, hook.end_s) - band_db(
        carved.audio, (40.0, 400.0), hook.start_s, hook.end_s
    )
    assert abs(low) < 0.1, "the low end is left alone"
    assert np.array_equal(dynamics.carve(beat, vocal, SR, depth_db=0.0).audio, beat)
    assert np.array_equal(dynamics.carve(beat, np.zeros(len(beat)), SR).audio, beat)

    hooks = [(s.start_s, s.end_s) for s in sections if s.label == "hook"]
    ducked = dynamics.carve(beat, vocal, SR, 0.0, duck_db=2.0, duck_spans=hooks)
    verse = sections[1]

    def level(audio: np.ndarray, start_s: float, end_s: float) -> float:
        part = audio[int(start_s * SR) : int(end_s * SR)]
        return float(10 * np.log10(np.mean(part**2)))

    assert level(beat, *hooks[0]) - level(ducked.audio, *hooks[0]) == pytest.approx(2.0, abs=0.4)
    inside = (verse.start_s + 0.5, verse.end_s - 0.5)
    assert level(beat, *inside) - level(ducked.audio, *inside) == pytest.approx(0.0, abs=0.01)
    assert ducked.max_duck_db == pytest.approx(2.0, abs=0.05)


def test_vocal_chain_presets(paths: Paths) -> None:
    assert mix_assistant.chain_names(paths) == ["gentle", "modern-rap", "aggressive"]
    chains = [mix_assistant.get_chain(paths, name) for name in mix_assistant.chain_names(paths)]
    ratios = [(found.compressor or {})["ratio"] for found in chains]
    assert ratios == sorted(ratios) and ratios[0] < ratios[-1], "gentle to aggressive"
    rap = mix_assistant.get_chain(paths, "modern-rap")
    assert rap.delay_note == "1/8" and rap.delay_db is not None and rap.saturation is not None
    assert [m.type for m in rap.before_deesser([])] == ["highpass"]
    assert [m.type for m in rap.after_deesser()] == [
        "compressor",
        "peak",
        "high_shelf",
        "saturation",
    ]
    assert mix_assistant.get_chain(paths, "gentle").delay_db is None

    paths.vocal_chains.mkdir(parents=True)
    (paths.vocal_chains / "mine.toml").write_text(
        'description = "Mine."\nhighpass_hz = 70\ninsert = "warm"\n'
        'delay_db = -20\ndelay_note = "1/4"\n'
    )
    mine = mix_assistant.get_chain(paths, "mine")
    assert mix_assistant.chain_names(paths)[-1] == "mine"
    assert (mine.insert, mine.highpass_hz, mine.compressor, mine.tone) == ("warm", 70, None, ())
    for text, message in (
        ("speed = 3\n", "unknown settings: speed"),
        ('delay_note = "1/3"\n', "delay_note must be one of"),
        ("reverb_db = 6\n", "between -60 and 0"),
        ('[[tone]]\ntype = "wobble"\n', "unknown type"),
        ("[compressor]\nratio = 99\n", "ratio"),
        ("highpass_hz = = 3\n", "not valid TOML"),
    ):
        with pytest.raises(MixError, match=message):
            mix_assistant.chain_from_toml(text, "x")
    with pytest.raises(MixError, match="no vocal chain named 'gone'"):
        mix_assistant.get_chain(paths, "gone")
    with pytest.raises(MixError, match="preset names use"):
        mix_assistant.get_chain(paths, "../x")


def test_vocal_is_gain_staged_and_its_own_chain_runs(paths: Paths, settings: Settings) -> None:
    vocal, _ = fixtures.sibilant_vocal()
    quiet = vocal * 0.1
    plain = VocalChain("plain", corrective_eq=False, deess_max_db=0.0)
    for audio in (vocal, quiet):
        done = mix_assistant.process_vocal(audio, SR, plain, paths, settings.thresholds)
        assert done.audio.shape == (len(vocal), 2), "a mono vocal comes out as centred stereo"
        assert loudness.integrated(done.audio, SR) == pytest.approx(-20.0, abs=0.3)
        assert done.cuts == [] and done.deess.events == 0
    hot = mix_assistant.process_vocal(vocal, SR, plain, paths, settings.thresholds)
    low = mix_assistant.process_vocal(quiet, SR, plain, paths, settings.thresholds)
    assert low.input_trim_db == pytest.approx(hot.input_trim_db + 20.0, abs=0.05)
    assert np.allclose(hot.audio, low.audio, atol=1e-4), "the file's own level does not matter"

    rap = mix_assistant.get_chain(paths, "modern-rap")
    full = mix_assistant.process_vocal(vocal, SR, rap, paths, settings.thresholds)
    assert full.deess.events == 3 and full.deess.max_reduction_db > 3
    warm = mix_assistant.process_vocal(
        vocal, SR, replace(plain, insert="warm"), paths, settings.thresholds
    )
    assert not np.allclose(warm.audio, hot.audio, atol=1e-3), "a chain of your own is applied"
    with pytest.raises(MixError, match="the vocal is silent"):
        mix_assistant.process_vocal(np.zeros(SR), SR, plain, paths, settings.thresholds)


def test_corrective_eq_cuts_at_a_planted_resonance(settings: Settings) -> None:
    rng = np.random.default_rng(3)
    breath = fixtures.pink_noise(6.0)[:, 0] + 0.02 * rng.standard_normal(6 * SR)
    ringing = pb.Pedalboard([pb.PeakFilter(3000.0, gain_db=14.0, q=12.0)])(  # pyright: ignore[reportPrivateImportUsage]
        breath.astype(np.float32)[None, :], SR
    )[0]
    assert mix_assistant.corrective_cuts(breath, SR, settings.thresholds) == []
    (cut,) = mix_assistant.corrective_cuts(ringing.astype(np.float64), SR, settings.thresholds)
    assert cut.type == "peak" and cut.params["freq_hz"] == pytest.approx(3000, rel=0.02)
    assert -6.0 <= cut.params["gain_db"] <= -3.0 and 2.0 <= cut.params["q"] <= 10.0


def test_sends_are_tempo_synced_and_sit_where_asked(paths: Paths) -> None:
    vocal, _ = fixtures.synthetic_vocal()
    dry = mix_assistant.stereo(vocal)
    echo = VocalChain("echo", delay_db=-12.0, delay_note="1/8", delay_feedback=0.0)
    wet, delay_ms = mix_assistant.sends(dry, SR, echo, 140.0)
    assert delay_ms is not None
    assert delay_ms == pytest.approx(60000 / 140 / 2, abs=0.1), "an eighth note at 140 BPM"
    assert wet.shape == dry.shape
    assert loudness.integrated(wet, SR) - loudness.integrated(dry, SR) == pytest.approx(
        -12.0, abs=0.3
    )
    first = int(0.25 * SR)
    assert np.abs(wet[: first + int(0.2 * SR)]).max() < 1e-3 * np.abs(wet).max(), (
        "nothing before the echo"
    )
    onset = int(np.argmax(np.abs(wet[:, 0]) > 0.05 * np.abs(wet).max()))
    assert onset / SR == pytest.approx(0.25 + delay_ms / 1000, abs=0.03)

    quarter, quarter_ms = mix_assistant.sends(dry, SR, replace(echo, delay_note="1/4"), 140.0)
    assert quarter_ms is not None and quarter_ms == pytest.approx(2 * delay_ms, abs=0.2)
    silent, none = mix_assistant.sends(dry, SR, echo, None)
    assert none is None and not silent.any(), "without a tempo there is no synced delay"
    room, _ = mix_assistant.sends(dry, SR, VocalChain("room", reverb_db=-18.0), 140.0)
    assert loudness.integrated(room, SR) - loudness.integrated(dry, SR) == pytest.approx(
        -18.0, abs=0.3
    )
    assert np.abs(room[:, 0] - room[:, 1]).max() > 0, "the reverb is stereo"
    nothing, _ = mix_assistant.sends(dry, SR, VocalChain("dry"), 140.0)
    assert not nothing.any()


def test_adlib_placement() -> None:
    vocal, _ = fixtures.synthetic_vocal()
    centre = mix_assistant.stereo(vocal)
    same = mix_assistant.place(centre, SR, 0.0, 0.0)
    assert np.allclose(same, centre), "centred and narrow is unchanged"
    left = mix_assistant.place(centre, SR, -0.6, 0.0)
    power = lambda x: float(np.mean(x**2))  # noqa: E731
    assert power(left[:, 0]) > 4 * power(left[:, 1])
    assert power(left) == pytest.approx(power(centre), rel=0.01), "panning keeps the power"
    wide = mix_assistant.place(centre, SR, 0.0, 1.0)
    shift = int(0.012 * SR)
    assert np.allclose(wide[shift:, 1], centre[:-shift, 1]) and np.allclose(
        wide[:, 0], centre[:, 0]
    )
    assert not wide[:shift, 1].any(), "the right side arrives 12 ms late"
    assert set(mix_assistant.ADLIB_PRESETS) == {"tucked", "wide", "left", "right"}


def test_one_click_gives_a_mastered_track_with_analysis_and_notes(mixed, song, sung) -> None:
    """Phase 9 criterion: dry vocal + instrumental to a mastered file, analysed, with AI notes."""
    paths, result, api, stages = mixed
    run = result.run
    assert (
        result.mastered
        and run.manifest.kind == "master"
        and run.manifest.track.slug == "night-drive"
    )
    master = run.reports["master"].loudness
    assert master.lufs_i == pytest.approx(-14.0, abs=0.3) and master.true_peak_dbtp <= -0.95
    assert run.output("master_24").is_file() and run.output("master_16").is_file()
    assert set(run.reports) == {"mix", "master", "reference"}
    mix = run.reports["mix"]
    assert mix.musical is not None and mix.musical.tempo_bpm == pytest.approx(140.0, abs=0.5)
    assert mix.sections is not None and len(mix.sections.items) == 6
    assert mix.lowend is not None and mix.band_deltas is not None
    assert (
        stages[0] == "Reading the vocal and the instrumental"
        and stages[-1] == "Asking for AI mix notes"
    )
    assert "Riding the vocal" in stages and "Matching to the reference" in stages

    assert result.notes is not None and result.notes_error is None
    assert result.notes.body == REPLY and result.notes.cost_usd > 0
    assert (run.run_dir / "notes.json").is_file(), "the notes are stored with the run"
    sent = json.loads(api.body()["messages"][0]["content"])
    assert sent["intent"] == "Vocal up front"
    told = sent["vocal_mix"]
    assert told["chain"] == "modern-rap" and told["target_db"] == TARGET
    assert len(told["sections"]) == 6 and told["deess"]["events"] == 0
    text = json.dumps(sent)
    assert ".wav" not in text and str(paths.root) not in text and "vox" not in text
    assert "`vocal_mix`, when present" in api.body()["system"]
    with index.open_index(paths) as conn:
        assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 1


def test_the_mix_holds_the_vocal_on_target_in_every_section(mixed, song, sung) -> None:
    """The rider's criterion again, measured on the files the assistant wrote."""
    _, result, _, _ = mixed
    lead = audio_io.load(result.path("lead_vocal"))[0]
    beat = audio_io.load(result.path("instrumental"))[0]
    for label, start_s, end_s in sung:
        assert section_ratio(lead, beat, start_s, end_s) == pytest.approx(TARGET, abs=1.0), label
    assert (result.target_db, result.target_source) == (TARGET, "the default for hip-hop")
    assert result.final_ratio_db == pytest.approx(TARGET, abs=0.5)
    assert [s.label for s in result.sections] == [
        "intro",
        "verse 1",
        "hook 1",
        "verse 2",
        "hook 2",
        "outro",
    ]
    for found in result.sections[1:5]:
        assert found.after_db == pytest.approx(TARGET, abs=0.5) and not found.limited
    assert result.reference_ratio_db is None, "the reference has no separated stems"
    assert result.tempo_bpm == pytest.approx(140.0, abs=0.5)
    assert result.delay_ms == pytest.approx(214.3, abs=1.0), "an eighth note at 140 BPM"
    assert result.carve["mean_dip_db"] == pytest.approx(3.0, abs=0.3) and result.trim_db <= 0
    assert result.flags == [], "nothing left to flag on the vocal"


def test_exports_add_up_and_every_setting_is_in_the_run(mixed) -> None:
    paths, result, _, _ = mixed
    assert result.folder == paths.projects / "night-drive" / "vocal_mix" / result.id
    assert set(result.files) == {"mix", "lead_vocal", "instrumental"}
    mix, info = audio_io.load(result.path("mix"))
    parts = (
        audio_io.load(result.path("lead_vocal"))[0] + audio_io.load(result.path("instrumental"))[0]
    )
    assert info.subtype == "PCM_24" and info.channels == 2
    error = 10 * np.log10(np.mean((mix - parts) ** 2) / np.mean(mix**2))
    assert error < -100, "vocal + instrumental is the mix, to 24-bit precision"
    assert np.abs(mix).max() <= 10 ** (-1 / 20) + 1e-6, "headroom for mastering"

    made = result.run.manifest.settings["made_from"]
    assert made["kind"] == "vocal_mix" and made["id"] == result.id
    assert (made["vocal"], made["instrumental"]) == ("Night Drive vox.wav", "Night Drive.wav")
    assert made["chain"]["name"] == "modern-rap" and made["chain"]["compressor"]["ratio"] == 4
    assert made["options"]["max_ride_db"] == 6.0 and made["options"]["carve_db"] == 3.0
    assert made["result"] == json.loads(json.dumps(result.summary()))
    stored = json.loads((result.run.run_dir / "run.json").read_text())
    assert stored["settings"]["made_from"] == made, "run.json holds every setting"
    beside = json.loads((result.folder / "vocal_mix.json").read_text())
    assert beside["run_id"] == result.run.manifest.id and beside["chain"] == made["chain"]
    assert result.run.mix_path.name == result.files["mix"], (
        "the run's input is the mix that was made"
    )

    text = summary.vocal_mix_text(result)
    assert "target -2.0 LU (the default for hip-hop)" in text
    assert "hook 1" in text and "De-esser: nothing to do." in text
    assert "Delay: 214 ms, from a tempo of 140.0 BPM." in text
    assert f"master: {result.run.output('master_24')}" in text


def test_without_a_reference_the_mix_is_analyzed_and_notes_can_fail(
    paths: Paths, settings: Settings, song, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocal, beat, _ = song
    hissy, _ = fixtures.sibilant_vocal()
    lead = vocal.copy()
    lead[: len(hissy)] = 0
    start = int(7.0 * SR)
    lead[start : start + len(hissy)] = hissy * (np.abs(vocal).max() / np.abs(hissy).max())
    files = (
        fixtures.write_wav(tmp_path / "vox.wav", lead),
        fixtures.write_wav(tmp_path / "Song.wav", beat),
    )
    extra = []
    for number, at in enumerate((30, 50)):
        adlib = np.zeros(len(beat))
        adlib[at * SR : at * SR + len(hissy)] = hissy
        extra.append(fixtures.write_wav(tmp_path / f"adlib {number}.wav", adlib))
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(mix_notes, "_keychain_key", lambda: None)

    options = MixOptions(chain="aggressive", target_db=-4.0, adlib_preset="left", duck_db=2.0)
    result = mix_assistant.mix(paths, settings, *files, options, extra, notes=True)
    assert not result.mastered and result.run.manifest.kind == "analysis_only"
    assert result.run.manifest.track.title == "Song" and result.run.manifest.track.genre is None
    assert [f.code for f in result.flags if f.code == "not_mastered"] == ["not_mastered"]
    assert result.notes is None and "No Anthropic API key was found" in (result.notes_error or "")
    assert (result.target_db, result.target_source) == (-4.0, "set by hand")
    assert result.deess["events"] >= 3 and result.deess["max_reduction_db"] > 3, "the planted esses"
    assert result.carve["max_duck_db"] == pytest.approx(2.0, abs=0.1)
    assert result.delay_ms == pytest.approx(428.6, abs=2.0), "a quarter note at 140 BPM"

    assert set(result.files) == {"mix", "lead_vocal", "instrumental", "adlibs"}
    backing = audio_io.load(result.path("adlibs"))[0]
    lead_out = audio_io.load(result.path("lead_vocal"))[0]
    first, second = (slice(at * SR, at * SR + len(hissy)) for at in (30, 50))
    lead_level = loudness.integrated(lead_out, SR)
    for window in (first, second):
        offset = loudness.integrated(backing[window], SR) - lead_level
        assert offset == pytest.approx(-6.0, abs=1.5), "each ad-lib sits about 6 LU under the lead"
    sides = [np.mean(backing[w, 0] ** 2) / np.mean(backing[w, 1] ** 2) for w in (first, second)]
    assert sides[0] > 3 and sides[1] < 1 / 3, "two files under 'left' take alternate sides"
    total = audio_io.load(result.path("mix"))[0]
    parts = lead_out + backing + audio_io.load(result.path("instrumental"))[0]
    assert 10 * np.log10(np.mean((total - parts) ** 2) / np.mean(total**2)) < -100
    assert "AI notes were not made" in summary.vocal_mix_text(result)
    assert "nothing to do" not in summary.vocal_mix_text(result)


def test_inputs_that_do_not_fit_are_refused_or_flagged(
    paths: Paths, settings: Settings, song, tmp_path: Path
) -> None:
    vocal, beat, _ = song
    beat_file = fixtures.write_wav(tmp_path / "Song.wav", beat)
    other_rate = fixtures.write_wav(tmp_path / "vox48.wav", vocal, 48000)
    with pytest.raises(MixError, match="bounce them at the same rate"):
        mix_assistant.mix(paths, settings, other_rate, beat_file)
    with pytest.raises(MixError, match="the vocal is silent"):
        mix_assistant.mix(
            paths, settings, fixtures.write_wav(tmp_path / "none.wav", vocal * 0), beat_file
        )
    refused: list[dict[str, Any]] = [
        {"max_ride_db": 20.0},
        {"carve_db": 9.0},
        {"adlib_preset": "everywhere"},
    ]
    for bad in refused:
        with pytest.raises(MixError):
            MixOptions(**bad)
    with pytest.raises(MixError, match="no vocal chain named 'gone'"):
        mix_assistant.mix(paths, settings, other_rate, beat_file, MixOptions(chain="gone"))
    fine = fixtures.write_wav(tmp_path / "vox.wav", vocal)
    with pytest.raises(library.ReferenceError):
        mix_assistant.mix(paths, settings, fine, beat_file, MixOptions(reference="hip-hop/nope"))

    short = fixtures.write_wav(tmp_path / "short.wav", vocal[: 40 * SR])
    result = mix_assistant.mix(
        paths, settings, short, beat_file, MixOptions(target_db=-3.0), separated=True
    )
    codes = [f.code for f in result.flags]
    assert codes[:2] == ["lengths_differ", "preview_quality"]
    assert result.preview_quality and audio_io.load(result.path("mix"))[1].frames == len(beat)
    late = [section for section in result.sections if section.start_s > 45]
    assert late and all(section.after_db is None for section in late), "nothing is sung after 40 s"


def test_target_comes_from_the_genre_a_number_or_the_reference(
    paths: Paths, settings: Settings, fx: dict[str, Path], tmp_path: Path
) -> None:
    def wanted(**options: Any) -> tuple[float, str, float | None]:
        chosen = MixOptions(**options)
        return mix_assistant.target_for(paths, settings, chosen, chosen.genre)

    assert wanted(genre="hip-hop") == (-2.0, "the default for hip-hop", None)
    assert wanted(genre="house") == (-6.0, "the default for house", None)
    assert wanted(genre="polka") == (-3.0, "the general default", None)
    assert wanted() == (-3.0, "the general default", None)
    assert wanted(genre="hip-hop", target_db=1.5) == (1.5, "set by hand", None)
    with pytest.raises(MixError, match="no reference with a vocal"):
        wanted(genre="hip-hop", target_from_reference=True)

    stems, mix = fixtures.song_stems(vocal_db=-5.0)
    reference = library.add(
        paths, fixtures.write_wav(tmp_path / "ref.wav", mix), "hip-hop", settings
    )

    def separator(staged: Path, model_file: str, out_dir: Path) -> list[Path]:
        audio, info = audio_io.load(staged)
        ours = stems["vocals"] * (np.abs(audio).max() / np.abs(mix).max())
        parts = (("Vocals", ours), ("Instrumental", audio - ours))
        return [
            audio_io.save_wav(out_dir / f"x_({n})_m.wav", p, info.sample_rate, 32) for n, p in parts
        ]

    stem_workflow.separate_reference(reference, runner=separator)
    target, source, theirs = wanted(genre="hip-hop", target_from_reference=True)
    assert source == "the reference's separated stems" and theirs is not None
    assert target == pytest.approx(theirs, abs=0.06) and -8 < theirs < 0
    assert wanted(genre="hip-hop") == (-2.0, "the default for hip-hop", theirs), "shown either way"


def test_mixing_from_a_tracks_stems(paths: Paths, settings: Settings, tmp_path: Path) -> None:
    stems, _ = fixtures.song_stems()
    names = {"drums": "Drums", "bass": "808 Bass", "other": "Pads", "vocals": "Lead Vox"}
    for role, audio in stems.items():
        fixtures.write_wav(tmp_path / "stems" / f"{names[role]}.wav", audio)
    stem_workflow.import_logic_stems(paths, tmp_path / "stems", "Night Drive", "hip-hop")
    result = mix_assistant.from_stems(paths, settings, "night-drive", MixOptions(chain="gentle"))
    assert not result.preview_quality, "stems from Logic are the real thing"
    assert "preview_quality" not in [f.code for f in result.flags]
    assert result.run.manifest.track.slug == "night-drive" and result.target_db == -2.0
    made = result.run.manifest.settings.get("made_from") or json.loads(
        (result.folder / "vocal_mix.json").read_text()
    )
    assert made["vocal"] == "lead-vox.wav" and made["instrumental"] == "instrumental-sum.wav"
    assert result.final_ratio_db == pytest.approx(-2.0, abs=1.0)
    with pytest.raises(stem_workflow.StemError):
        mix_assistant.from_stems(paths, settings, "nope")


def test_ui_helpers_for_the_vocal_mix_tab(mixed) -> None:
    paths, result, _, _ = mixed
    loaded = json.loads(vocal_tab.mix_player_config(result))
    labels = [source["label"] for source in loaded["sources"]]
    assert labels == ["Mix", "Master", "Reference", "Lead vocal, processed", "Instrumental"]
    assert loaded["track"] == "night-drive" and loaded["selected"] == "master"
    for source in loaded["sources"]:
        assert Path(source["url"].removeprefix("/gradio_api/file=")).is_file()
        assert -40 < source["lufs"] < 0
    assert [section["label"] for section in loaded["sections"]][1] == "verse 1"
    assert len(loaded["sections"]) == 6
    assert vocal_tab.tuned_choices(paths) == [], "nothing was tuned for this track"
