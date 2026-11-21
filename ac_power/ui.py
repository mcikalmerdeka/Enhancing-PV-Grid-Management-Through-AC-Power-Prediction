"""Altair chart builders for AC Power Studio.

No Streamlit import here on purpose: the charts are plain Altair objects, so
they can be built and inspected from a script or a test without a server.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

ACCENT = "#D97706"
POSITIVE = "#16A34A"
NEGATIVE = "#DC2626"
INK = "#1C1917"
TRACK = "#E7E5E4"
MUTED = "#57534E"

PLANT_COLORS = ["#D97706", "#0F766E"]


def output_by_plant_chart(frame: pd.DataFrame) -> alt.Chart:
    """Mean output per plant, highest first."""
    order = frame.sort_values("AC_POWER", ascending=False)["PLANT_NAME"].tolist()
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadius=3, color=ACCENT)
        .encode(
            x=alt.X("AC_POWER:Q", title="Mean AC power (kW)"),
            y=alt.Y("PLANT_NAME:N", title=None, sort=order),
            tooltip=[
                alt.Tooltip("PLANT_NAME:N", title="Plant"),
                alt.Tooltip("AC_POWER:Q", title="Mean kW", format=",.1f"),
            ],
        )
        .properties(height=alt.Step(34))
    )


def output_by_hour_chart(frame: pd.DataFrame) -> alt.Chart:
    """Mean output per hour of day, in clock order."""
    return (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=55), strokeWidth=2.4, color=ACCENT)
        .encode(
            x=alt.X("HOUR:O", title="Hour of day"),
            y=alt.Y("AC_POWER:Q", title="Mean AC power (kW)"),
            tooltip=[
                alt.Tooltip("HOUR:O", title="Hour"),
                alt.Tooltip("AC_POWER:Q", title="Mean kW", format=",.1f"),
            ],
        )
        .properties(height=260)
    )


def daily_output_chart(frame: pd.DataFrame) -> alt.Chart:
    """Mean daylight output per day across the observed window."""
    return (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2.4, color=ACCENT)
        .encode(
            x=alt.X("DATE:T", title=None),
            y=alt.Y("AC_POWER:Q", title="Mean AC power (kW)"),
            tooltip=[
                alt.Tooltip("DATE:T", title="Day", format="%d %b"),
                alt.Tooltip("AC_POWER:Q", title="Mean kW", format=",.1f"),
            ],
        )
        .properties(height=240)
    )


def irradiation_scatter_chart(frame: pd.DataFrame) -> alt.Chart:
    """Irradiation against output, coloured by plant."""
    return (
        alt.Chart(frame)
        .mark_circle(size=26, opacity=0.45)
        .encode(
            x=alt.X("IRRADIATION:Q", title="Irradiation (kW/m²)"),
            y=alt.Y("AC_POWER:Q", title="AC power (kW)"),
            color=alt.Color(
                "PLANT_NAME:N",
                title=None,
                scale=alt.Scale(range=PLANT_COLORS),
            ),
            tooltip=[
                alt.Tooltip("PLANT_NAME:N", title="Plant"),
                alt.Tooltip("IRRADIATION:Q", title="Irradiation", format=".3f"),
                alt.Tooltip("AC_POWER:Q", title="AC kW", format=",.1f"),
            ],
        )
        .properties(height=300)
    )


def importance_chart(frame: pd.DataFrame) -> alt.Chart:
    """Feature importance, largest first."""
    order = frame["feature"].tolist()
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadius=3, color=ACCENT)
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", title=None, sort=order),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".3f"),
            ],
        )
        .properties(height=alt.Step(32))
    )


def contribution_chart(frame: pd.DataFrame) -> alt.Chart:
    """Per-prediction deltas: what these conditions add or remove versus typical.

    A dot is drawn on every row as well as the bar, so a feature that changes
    nothing still reads as "0.0" instead of vanishing into a zero-width bar.
    """
    frame = frame.assign(
        direction=frame["delta"].map(
            lambda delta: "Adds output" if delta >= 0 else "Reduces output"
        )
    )
    order = frame["display"].tolist()
    span = max(float(frame["delta"].abs().max()), 1.0) * 1.15
    scale = alt.Scale(domain=[-span, span])

    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("delta:Q", title="Change vs typical conditions (kW)", scale=scale),
            y=alt.Y("display:N", title=None, sort=order),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Adds output", "Reduces output"],
                    range=[POSITIVE, NEGATIVE],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("display:N", title="Feature"),
                alt.Tooltip("input:Q", title="This value", format=",.2f"),
                alt.Tooltip("baseline:Q", title="Typical value", format=",.2f"),
                alt.Tooltip("delta:Q", title="Change (kW)", format="+,.1f"),
            ],
        )
    )
    dots = (
        alt.Chart(frame)
        .mark_point(size=55, filled=True, color=MUTED)
        .encode(
            x=alt.X("delta:Q", scale=scale),
            y=alt.Y("display:N", sort=order),
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"zero": [0.0]}))
        .mark_rule(color="#A8A29E", strokeWidth=1)
        .encode(x="zero:Q")
    )
    return (bars + rule + dots).properties(height=alt.Step(34))


def gauge_chart(value: float, low: float = 0.0, high: float = 30_000.0) -> alt.Chart:
    """Bullet gauge: a full-range track with the prediction marked on it.

    Deliberately a horizontal bullet rather than an arc: a bar encodes length,
    which is read accurately, and it carries a text label in the same view.
    The label uses the clamped value so it can never claim a number outside
    the track.
    """
    clamped = min(max(float(value), low), high)
    track = (
        alt.Chart(pd.DataFrame({"low": [low], "high": [high]}))
        .mark_bar(color=TRACK, cornerRadius=4, height=34)
        .encode(x=alt.X("low:Q", title=None), x2="high:Q")
    )
    fill = (
        alt.Chart(pd.DataFrame({"low": [low], "value": [clamped]}))
        .mark_bar(color=ACCENT, cornerRadius=4, height=34)
        .encode(x=alt.X("low:Q", title=None), x2="value:Q")
    )
    label = (
        alt.Chart(
            pd.DataFrame({"x": [(low + high) / 2], "label": [f"{clamped:,.1f} kW"]})
        )
        .mark_text(fontSize=18, fontWeight="bold", color=INK)
        .encode(x=alt.X("x:Q", title=None), text="label:N")
    )
    return (
        (track + fill + label)
        .encode(x=alt.X(scale=alt.Scale(domain=[low, high], nice=False)))
        .properties(height=60)
    )
