"""Plotly figures for the Analyze and Progress tabs."""

from __future__ import annotations

import plotly.graph_objects as go

from masterbench.analysis.compare import BAND_LABELS
from masterbench.analysis.report import AnalysisReport
from masterbench.config import BANDS
from masterbench.progress import FlagCount, TrackPoint

ROLE_COLORS = {"mix": "#6b7280", "master": "#ea580c", "reference": "#2563eb"}
_LAYOUT = {
    "margin": {"l": 50, "r": 20, "t": 40, "b": 45},
    "height": 340,
    "template": "plotly_white",
}


def _empty(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, font={"size": 14})
    figure.update_layout(**_LAYOUT, xaxis={"visible": False}, yaxis={"visible": False})
    return figure


def spectrum(reports: dict[str, AnalysisReport]) -> go.Figure:
    """Smoothed spectra overlaid at equal loudness."""
    shown = {role: r for role, r in reports.items() if r.spectrum is not None}
    if not shown:
        return _empty("This run was analyzed before spectra were stored. Analyze it again.")
    figure = go.Figure()
    for role, report in shown.items():
        assert report.spectrum is not None
        figure.add_scatter(
            x=report.spectrum.freqs_hz,
            y=report.spectrum.level_db,
            name=role.capitalize(),
            line={"color": ROLE_COLORS.get(role), "width": 2},
        )
    for low, _ in list(BANDS.values())[1:]:
        figure.add_vline(x=low, line={"color": "#d1d5db", "width": 1, "dash": "dot"})
    figure.update_layout(
        **_LAYOUT,
        title="Frequency balance at equal loudness (1/6-octave smoothing)",
        xaxis={"type": "log", "title": "Hz", "range": [1.3, 4.3]},
        yaxis={"title": "dB relative to integrated loudness"},
    )
    return figure


def band_deltas(reports: dict[str, AnalysisReport], limit_db: float) -> go.Figure:
    shown = {role: r for role, r in reports.items() if r.band_deltas}
    if not shown:
        return _empty("No reference to compare against.")
    figure = go.Figure()
    for role, report in shown.items():
        assert report.band_deltas is not None
        figure.add_bar(
            x=[BAND_LABELS[name] for name in report.band_deltas],
            y=list(report.band_deltas.values()),
            name=role.capitalize(),
            marker={"color": ROLE_COLORS.get(role)},
        )
    for edge in (limit_db, -limit_db):
        figure.add_hline(y=edge, line={"color": "#9ca3af", "width": 1, "dash": "dash"})
    figure.update_layout(
        **_LAYOUT,
        title="Band level versus the reference (dB, equal loudness)",
        yaxis={"title": "dB", "zeroline": True},
        barmode="group",
    )
    return figure


def loudness_over_time(reports: dict[str, AnalysisReport]) -> go.Figure:
    shown = {role: r for role, r in reports.items() if r.short_term_lufs and role != "reference"}
    if not shown:
        return _empty("No loudness curve stored for this run.")
    figure = go.Figure()
    for role, report in shown.items():
        assert report.short_term_lufs is not None
        seconds = [1.5 + i for i in range(len(report.short_term_lufs))]
        figure.add_scatter(
            x=seconds,
            y=report.short_term_lufs,
            name=role.capitalize(),
            line={"color": ROLE_COLORS.get(role), "width": 2},
        )
    figure.update_layout(
        **_LAYOUT,
        title="Short-term loudness over time",
        xaxis={"title": "seconds"},
        yaxis={"title": "LUFS"},
    )
    return figure


def band_trends(points: list[TrackPoint]) -> go.Figure:
    """Each band's distance from the reference, track by track."""
    compared = [p for p in points if p.band_deltas]
    if not compared:
        return _empty("No tracks have been compared with a reference yet.")
    figure = go.Figure()
    labels = [f"{p.title}<br>{p.created_at[:10]}" for p in compared]
    for name, label in BAND_LABELS.items():
        figure.add_scatter(
            x=labels,
            y=[p.band_deltas.get(name) for p in compared],
            name=label,
            mode="lines+markers",
        )
    figure.add_hline(y=0, line={"color": "#9ca3af", "width": 1})
    figure.update_layout(
        **_LAYOUT,
        title="Mix balance versus reference, by track (dB at equal loudness)",
        yaxis={"title": "dB"},
    )
    return figure


def loudness_trends(points: list[TrackPoint]) -> go.Figure:
    if not points:
        return _empty("No tracks analyzed yet.")
    labels = [f"{p.title}<br>{p.created_at[:10]}" for p in points]
    figure = go.Figure()
    figure.add_scatter(
        x=labels, y=[p.lufs_i for p in points], name="Mix LUFS-I", mode="lines+markers"
    )
    figure.add_scatter(
        x=labels, y=[p.crest_db for p in points], name="Crest (dB)", mode="lines+markers"
    )
    figure.add_scatter(x=labels, y=[p.lra for p in points], name="LRA (LU)", mode="lines+markers")
    figure.update_layout(**_LAYOUT, title="Mix loudness and dynamics, by track")
    return figure


def flag_frequency(counts: list[FlagCount], total: int) -> go.Figure:
    if not counts:
        return _empty("No flags raised so far.")
    figure = go.Figure()
    figure.add_bar(
        x=[c.tracks for c in reversed(counts)],
        y=[c.code.replace("_", " ") for c in reversed(counts)],
        orientation="h",
        marker={"color": "#ea580c"},
    )
    figure.update_layout(
        **{**_LAYOUT, "margin": {"l": 170, "r": 20, "t": 40, "b": 45}},
        title=f"Recurring flags (tracks out of {total})",
        xaxis={"dtick": 1, "range": [0, max(total, 1)]},
    )
    return figure
