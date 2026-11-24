"""Plain-text and Markdown summaries of a run, shared by the CLI, UI and notifications."""

from __future__ import annotations

from pathlib import Path

from masterbench.analysis import sections as section_tools
from masterbench.analysis.detectors import clock
from masterbench.analysis.flags import Flag
from masterbench.analysis.lowend import LowEnd
from masterbench.analysis.report import AnalysisReport
from masterbench.batch import BatchResult
from masterbench.master.pipeline import RunResult
from masterbench.progress import Habit
from masterbench.release.versions import Versions
from masterbench.stems.workflow import Diagnosis

_ROLES = ("mix", "master", "reference")
_ICONS = {"problem": "✖", "warn": "▲", "info": "•"}
_COLUMNS = ("", "LUFS-I", "True peak", "LRA", "Crest", "Short-term max")


def _rows(result: RunResult) -> list[tuple[str, ...]]:
    rows = []
    for role in _ROLES:
        report = result.reports.get(role)
        if report is None:
            continue
        loud = report.loudness
        rows.append(
            (
                role.capitalize(),
                f"{loud.lufs_i:.1f}",
                f"{loud.true_peak_dbtp:+.1f} dBTP",
                f"{loud.lra:.1f} LU",
                f"{loud.crest_db:.1f} dB",
                f"{loud.lufs_s_max:.1f}",
            )
        )
    return rows


def flag_line(flag: Flag) -> str:
    return f"{_ICONS[flag.severity]} {flag.message}"


def headline(result: RunResult) -> str:
    """One line for notifications: the main numbers plus the worst flags."""
    role = "master" if "master" in result.reports else "mix"
    text = f"{role.capitalize()}: {result.reports[role].headline()}"
    worst = [f for f in result.flags if f.severity != "info"]
    if worst:
        text += f". {len(worst)} flag(s): " + ", ".join(f.code.replace("_", " ") for f in worst[:3])
    return text


def as_text(result: RunResult) -> str:
    rows = [_COLUMNS, *_rows(result)]
    widths = [max(len(row[i]) for row in rows) for i in range(len(_COLUMNS))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    ]
    lines.append("")
    lines += analysis_lines(result)
    lines += [flag_line(flag) for flag in result.flags] or ["No flags."]
    if result.manifest.outputs:
        lines.append("")
        lines += [str(result.output(name)) for name in result.manifest.outputs]
    lines.append(f"run: {result.run_dir}")
    return "\n".join(line.rstrip() for line in lines)


def as_markdown(result: RunResult) -> str:
    track = result.manifest.track
    kind = result.manifest.kind.replace("_", " ")
    lines = [
        f"### {track.title}",
        f"`{result.manifest.id}` · {kind} · {track.genre or 'no genre'}",
        "",
        "| " + " | ".join(_COLUMNS) + " |",
        "|" + "---|" * len(_COLUMNS),
    ]
    lines += ["| " + " | ".join(row) + " |" for row in _rows(result)]
    stats = result.manifest.stats
    if stats:
        lines += [
            "",
            f"Gain {stats['gain_db']:+.1f} dB, limiter reduction up to "
            f"{stats['max_reduction_db']:.1f} dB, {stats['seconds']:.1f} s.",
        ]
    return "\n".join(lines)


def flags_markdown(result: RunResult) -> str:
    if not result.flags:
        return "No flags."
    return "\n".join(f"- {flag_line(flag)}" for flag in result.flags)


_RATIO_NAMES = {
    "vocal_to_instrumental_db": "Vocal vs beat",
    "drums_to_total_db": "Drums vs mix",
    "bass_to_total_db": "Bass vs mix",
}


def stems_text(diagnosis: Diagnosis) -> str:
    """Stem levels, the headline ratios (with the reference's beside them) and flags."""
    balance, reference = diagnosis.balance, diagnosis.reference
    lines = [f"stem set: {diagnosis.stem_set}", "", "Level of each stem relative to the mix:"]
    lines += [f"  {name:<16} {level:+6.1f} dB" for name, level in balance.levels_db.items()]
    ratios = balance.ratios()
    if ratios:
        lines.append("")
    for key, value in ratios.items():
        theirs = reference.ratios().get(key) if reference else None
        versus = f"   (reference {theirs:+.1f} dB)" if theirs is not None else ""
        lines.append(f"  {_RATIO_NAMES[key]:<16} {value:+6.1f} dB{versus}")
    if balance.reconstruction_db is not None:
        lines += ["", f"Stems sum to within {balance.reconstruction_db:.0f} dB of the mix."]
    if diagnosis.flags:
        lines += ["", *[flag_line(flag) for flag in diagnosis.flags]]
    if diagnosis.warning:
        lines += ["", diagnosis.warning]
    return "\n".join(line.rstrip() for line in lines).rstrip()


def stems_markdown(diagnosis: Diagnosis) -> str:
    balance, reference = diagnosis.balance, diagnosis.reference
    lines = [f"**Stem set `{diagnosis.stem_set}`**", ""]
    ratios = balance.ratios()
    if ratios:
        lines += ["| | This track | Reference |", "|---|---|---|"]
        for key, value in ratios.items():
            theirs = reference.ratios().get(key) if reference else None
            shown = f"{theirs:+.1f} dB" if theirs is not None else ""
            lines.append(f"| {_RATIO_NAMES[key]} | {value:+.1f} dB | {shown} |")
        lines.append("")
    lines += [f"- {flag_line(flag)}" for flag in diagnosis.flags]
    if balance.reconstruction_db is not None:
        lines.append(f"- Stems sum to within {balance.reconstruction_db:.0f} dB of the mix.")
    if diagnosis.warning:
        lines += ["", f"_{diagnosis.warning}_"]
    return "\n".join(lines)


def batch_text(result: BatchResult) -> str:
    """Per-track table, the spreads, flags and failures of a batch."""
    lines = []
    report = result.report
    if report is not None:
        header = ("Track", "LUFS-I", "vs set", "True peak", "LRA", "Loudest part")
        rows = [header] + [
            (
                t.title,
                f"{t.lufs_i:.1f}",
                f"{t.loudness_offset_lu:+.1f} LU",
                f"{t.true_peak_dbtp:+.1f} dBTP",
                f"{t.lra:.1f} LU",
                f"{t.loudest_section_lufs:.1f}",
            )
            for t in report.tracks
        ]
        widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
        lines += [
            "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip()
            for row in rows
        ]
        lines += [
            "",
            f"Loudness spread {report.loudness_spread_lu:.1f} LU; loudest sections spread "
            f"{report.loudest_section_spread_lu:.1f} LU; highest true peak "
            f"{report.max_true_peak_dbtp:+.1f} dBTP.",
            "Played in album order, Spotify applies one gain to the whole set, so these "
            "differences between tracks are what listeners hear.",
            "",
            *([flag_line(flag) for flag in report.flags] or ["The set is consistent: no flags."]),
        ]
    elif result.runs:
        lines.append("One track mastered; a consistency report needs at least two.")
    for failure in result.failures:
        lines.append(f"✖ {Path(failure.source).name} failed: {failure.error}")
    lines.append(f"batch: {result.dir}")
    return "\n".join(lines)


def batch_markdown(result: BatchResult) -> str:
    report = result.report
    lines = [f"### Batch `{result.id}`" + (f": {result.name}" if result.name else ""), ""]
    if report is not None:
        lines += [
            "| Track | LUFS-I | vs set | True peak | LRA | Loudest part |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {t.title} | {t.lufs_i:.1f} | {t.loudness_offset_lu:+.1f} LU | "
            f"{t.true_peak_dbtp:+.1f} dBTP | {t.lra:.1f} LU | {t.loudest_section_lufs:.1f} |"
            for t in report.tracks
        ]
        lines += [
            "",
            f"Loudness spread **{report.loudness_spread_lu:.1f} LU**, loudest sections "
            f"**{report.loudest_section_spread_lu:.1f} LU**. In album order Spotify applies "
            "one gain to the whole set, so these differences are what listeners hear.",
            "",
            *([f"- {flag_line(f)}" for f in report.flags] or ["The set is consistent: no flags."]),
        ]
    lines += [f"- ✖ {Path(f.source).name} failed: {f.error}" for f in result.failures]
    return "\n".join(lines)


def musical_line(report: AnalysisReport) -> str:
    music = report.musical
    if music is None:
        return ""
    others = (
        f"; could also be {' or '.join(music.key_alternatives)}" if music.key_alternatives else ""
    )
    return (
        f"Key {music.key} (confidence {music.key_confidence:.0%}{others}), "
        f"{music.tempo_bpm:.1f} BPM"
    )


def analysis_lines(result: RunResult) -> list[str]:
    """Key, tempo and the shape of the song, for the run summary."""
    mix = result.reports.get("mix")
    if mix is None:
        return []
    lines = [musical_line(mix)] if mix.musical else []
    parts = mix.sections
    if parts is not None and len(parts.items) > 1:
        shape = ", ".join(f"{item.label} {clock(item.start_s)}" for item in parts.items)
        lines.append(f"Sections ({parts.source}): {shape}")
        if parts.contrast_lu is not None:
            genre = result.manifest.track.genre
            hook, verse = (section_tools.display_kind(k, genre) for k in ("hook", "verse"))
            lines.append(f"The {hook} is {parts.contrast_lu:.1f} LU louder than the {verse}.")
    return [*lines, ""] if lines else []


def sections_text(report: AnalysisReport) -> str:
    parts = report.sections
    if parts is None:
        return "No sections stored for this run; analyze it again."
    rows = [("Section", "From", "To", "LUFS", "Correlation")]
    rows += [
        (
            item.label,
            clock(item.start_s),
            clock(item.end_s),
            f"{item.lufs:.1f}",
            f"{item.correlation:+.2f}" if item.correlation is not None else "",
        )
        for item in parts.items
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    lines = [
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip() for row in rows
    ]
    lines.append("")
    lines.append(
        f"Loudest 30 s: {clock(parts.loudest_start_s)} to {clock(parts.loudest_end_s)} "
        f"at {parts.loudest_lufs:.1f} LUFS."
    )
    if parts.contrast_lu is not None:
        lines.append(f"Hook minus verse: {parts.contrast_lu:.1f} LU.")
    guess = "set by hand" if parts.source == "manual" else "detected; the kinds are guesses"
    lines.append(f"Sections were {guess}.")
    return "\n".join(lines)


def sections_markdown(report: AnalysisReport) -> str:
    parts = report.sections
    if parts is None:
        return ""
    lines = ["| Section | From | To | LUFS |", "|---|---|---|---|"]
    lines += [
        f"| {item.label} | {clock(item.start_s)} | {clock(item.end_s)} | {item.lufs:.1f} |"
        for item in parts.items
    ]
    extra = (
        f" Hook minus verse: **{parts.contrast_lu:.1f} LU**."
        if parts.contrast_lu is not None
        else ""
    )
    guess = "Set by hand." if parts.source == "manual" else "Detected; the kinds are guesses."
    lines += [
        "",
        f"Loudest 30 s: {clock(parts.loudest_start_s)} to {clock(parts.loudest_end_s)} "
        f"({parts.loudest_lufs:.1f} LUFS).{extra} {guess}",
    ]
    return "\n".join(lines)


def lowend_text(found: LowEnd) -> str:
    """Bass notes with their tuning, how much carries on small speakers, and the kick clash."""
    if not found.notes:
        return f"No sustained bass notes found in the {found.source}."
    rows = [("From", "To", "Note", "Hz", "Cents", "In key")]
    rows += [
        (
            clock(note.start_s),
            clock(note.end_s),
            note.note,
            f"{note.freq_hz:.1f}",
            f"{note.cents:+.0f}",
            "" if note.in_key is None else "yes" if note.in_key else "NO",
        )
        for note in found.notes[:40]
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(6)]
    lines = [
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip() for row in rows
    ]
    if len(found.notes) > 40:
        lines.append(f"... and {len(found.notes) - 40} more notes")
    lines.append("")
    key = f" and in {found.key}" if found.key else ""
    if found.in_tune_share is not None:
        lines.append(
            f"{found.in_tune_share:.0%} of the bass (by time) is in tune{key}. "
            f"Measured from the {found.source}."
        )
    if found.harmonics_db is not None:
        lines.append(
            f"Strongest harmonic above the small-speaker range: {found.harmonic_hz:.0f} Hz, "
            f"{-found.harmonics_db:.0f} dB under the fundamental."
        )
    if found.clash is not None:
        clash = found.clash
        lines.append(
            f"Kick and bass: on {clash.clashing} of {clash.hits} kick hits the bass is close to "
            f"the kick's level in 40 to 120 Hz (median {clash.overlap_db:+.1f} dB, strongest "
            f"overlap {clash.low_hz:.0f} to {clash.high_hz:.0f} Hz)."
        )
    elif found.source == "mix":
        lines.append("Kick and bass clash needs drum and bass stems (Logic stems or demucs-4).")
    return "\n".join(lines)


def lowend_markdown(found: LowEnd | None) -> str:
    if found is None:
        return ""
    return "**Low end**\n\n```\n" + lowend_text(found) + "\n```"


def versions_text(found: Versions, folder: Path) -> str:
    """The alternate versions with their levels, and how well they add up to the master."""
    how = {
        "exact": "the master's stored EQ and gain, applied exactly",
        "fitted": "one fixed filter fitted to what the master did to the mix",
    }[found.method]
    lines = [f"Processing: {how}."]
    for version in found.versions:
        lines.append(
            f"{version.name:<13} {version.lufs_i:6.1f} LUFS  {version.true_peak_dbtp:+6.1f} dBTP  "
            f"limiter {version.limiter_reduction_db:.1f} dB  {folder / version.file}"
        )
    lines.append(
        f"Instrumental + a cappella is within {found.sum_error_db:.0f} dB of the master "
        f"(the more negative, the closer)."
    )
    lines += [flag_line(flag) for flag in found.flags]
    return "\n".join(lines)


def versions_markdown(found: Versions) -> str:
    lines = ["| Version | LUFS-I | True peak | Limiter |", "|---|---|---|---|"]
    lines += [
        f"| {v.name} | {v.lufs_i:.1f} | {v.true_peak_dbtp:+.1f} dBTP "
        f"| {v.limiter_reduction_db:.1f} dB |"
        for v in found.versions
    ]
    lines += [
        "",
        f"Instrumental + a cappella is within **{found.sum_error_db:.0f} dB** of the master "
        f"(the more negative, the closer). Stems: `{found.stem_set}`.",
        "",
        *[f"- {flag_line(flag)}" for flag in found.flags],
    ]
    return "\n".join(lines)


def habits_text(habits: list[Habit], considered: int) -> str:
    if considered < 3:
        verb = "is" if considered == 1 else "are"
        return f"Habits need at least 3 analyzed tracks; there {verb} {considered}."
    if not habits:
        return f"No consistent habits across the last {considered} tracks."
    lines = [f"Across the last {considered} tracks:"]
    for habit in habits:
        lines.append(f"- {habit.statement} (confidence {habit.confidence:.0%})")
        lines.append(f"    e.g. {', '.join(habit.examples)}")
    lines.append("")
    lines.append(
        "Confidence is the lower bound of a 95% interval on the share of tracks, so it "
        "grows as more tracks show the same thing."
    )
    return "\n".join(lines)


def release_checklist(
    result: RunResult, target_lufs: float, ceiling_dbtp: float
) -> list[tuple[bool | None, str]]:
    """Final checks before release. Each item is (passed, text); None means not checked yet."""
    from masterbench.preview import speakers, streaming

    master = result.reports.get("master")
    if master is None:
        return [(False, "This run has no master yet.")]
    loud = master.loudness
    wanted = result.manifest.settings.get("target_lufs")
    target = target_lufs if wanted is None else wanted
    ceiling = result.manifest.settings.get("ceiling_dbtp", ceiling_dbtp)
    items: list[tuple[bool | None, str]] = [
        (
            abs(loud.lufs_i - target) <= 0.5,
            f"Loudness {loud.lufs_i:.1f} LUFS (target {target:.1f}).",
        ),
        (
            loud.true_peak_dbtp <= ceiling + 0.05,
            f"True peak {loud.true_peak_dbtp:+.2f} dBTP (ceiling {ceiling:+.1f}).",
        ),
    ]
    problems = [f for f in master.flags if f.severity == "problem"]
    warnings = [f for f in master.flags if f.severity == "warn"]
    items.append(
        (not problems, f"{len(problems)} problem(s) and {len(warnings)} warning(s) on the master.")
    )
    preview = streaming.load(result)
    if preview is None:
        items.append((None, "Streaming previews not rendered: encoded peaks are unchecked."))
    else:
        worst = max((c.true_peak_dbtp for c in preview.codecs), default=loud.true_peak_dbtp)
        items.append((worst < 0, f"Highest true peak after encoding: {worst:+.2f} dBTP."))
        normal = next(p for p in preview.platforms if p.platform == "Spotify Normal")
        items.append((abs(normal.gain_db) <= 1.0, normal.message))
    stereo = master.stereo
    if stereo is not None:
        items.append(
            (stereo.correlation >= 0, f"Mono compatibility: correlation {stereo.correlation:+.2f}.")
        )
    sims = speakers.load(result)
    items.append(
        (
            None if sims is None else True,
            "Speaker sims rendered: listen to phone and mono."
            if sims
            else "Speaker sims not rendered yet.",
        )
    )
    return items


def checklist_markdown(items: list[tuple[bool | None, str]]) -> str:
    marks = {True: "✔", False: "✖", None: "•"}
    return "\n".join(f"- {marks[passed]} {text}" for passed, text in items)
