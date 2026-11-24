"""Plotly figures for the Analyze and Progress tabs."""

from __future__ import annotations

import plotly.graph_objects as go

from masterbench.analysis.compare import BAND_LABELS
from masterbench.analysis.report import AnalysisReport
from masterbench.config import BANDS
from masterbench.progress import FlagCount, TrackPoint
from masterbench.references.target_curve import TargetCurve

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


def spectrum(reports: dict[str, AnalysisReport], curve: TargetCurve | None = None) -> go.Figure:
    """Smoothed spectra overlaid at equal loudness, over the genre curve's range if there is one."""
    shown = {role: r for role, r in reports.items() if r.spectrum is not None}
    if not shown:
        return _empty("This run was analyzed before spectra were stored. Analyze it again.")
    figure = go.Figure()
    if curve is not None:
        name = f"{curve.genre} references (middle half)"
        figure.add_scatter(x=curve.freqs_hz, y=curve.p75_db, line={"width": 0}, showlegend=False)
        figure.add_scatter(
            x=curve.freqs_hz,
            y=curve.p25_db,
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(22, 163, 74, 0.18)",
            name=name,
        )
        figure.add_scatter(
            x=curve.freqs_hz,
            y=curve.median_db,
            name=f"{curve.genre} curve (median)",
            line={"color": "#16a34a", "width": 1.5, "dash": "dash"},
        )
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
    """Band levels against the genre curve's median when there is one, else the reference."""
    by_curve = any(r.curve_deltas for r in reports.values())
    pick = (lambda r: r.curve_deltas) if by_curve else (lambda r: r.band_deltas)
    shown = {role: pick(r) for role, r in reports.items() if pick(r)}
    if not shown:
        return _empty("No reference to compare against.")
    against = "genre curve" if by_curve else "reference"
    figure = go.Figure()
    for role, deltas in shown.items():
        assert deltas is not None
        figure.add_bar(
            x=[BAND_LABELS[name] for name in deltas],
            y=list(deltas.values()),
            name=role.capitalize(),
            marker={"color": ROLE_COLORS.get(role)},
        )
    for edge in (limit_db, -limit_db):
        figure.add_hline(y=edge, line={"color": "#9ca3af", "width": 1, "dash": "dash"})
    figure.update_layout(
        **_LAYOUT,
        title=f"Band level versus the {against} (dB, equal loudness)",
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


def shape_trends(points: list[TrackPoint]) -> go.Figure:
    """How much the hook lifts, and how much of the bass is in tune, track by track."""
    shown = [
        p for p in points if p.hook_contrast_lu is not None or p.bass_in_tune_share is not None
    ]
    if not shown:
        return _empty("No tracks with sections or bass notes measured yet.")
    labels = [f"{p.title}<br>{p.created_at[:10]}" for p in shown]
    figure = go.Figure()
    figure.add_scatter(
        x=labels,
        y=[p.hook_contrast_lu for p in shown],
        name="Hook minus verse (LU)",
        mode="lines+markers",
    )
    figure.add_scatter(
        x=labels,
        y=[None if p.bass_in_tune_share is None else 100 * p.bass_in_tune_share for p in shown],
        name="Bass in tune (%)",
        mode="lines+markers",
        yaxis="y2",
    )
    figure.update_layout(
        **_LAYOUT,
        title="Hook lift and bass tuning, by track",
        yaxis={"title": "LU"},
        yaxis2={"title": "%", "overlaying": "y", "side": "right", "range": [0, 105]},
    )
    return figure
