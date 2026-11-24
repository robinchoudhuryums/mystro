"""Plain-text and Markdown summaries of a run, shared by the CLI, UI and notifications."""

from __future__ import annotations

from masterbench.analysis.flags import Flag
from masterbench.master.pipeline import RunResult
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
