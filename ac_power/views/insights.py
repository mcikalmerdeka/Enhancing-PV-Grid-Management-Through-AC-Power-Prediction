"""Model and dataset insights view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ac_power.analytics import feature_importance, model_comparison, per_plant_metrics
from ac_power.schema import TARGET_COLUMN
from ac_power.ui import (
    daily_output_chart,
    importance_chart,
    irradiation_scatter_chart,
    output_by_hour_chart,
    output_by_plant_chart,
)


def _takeaway(text: str) -> None:
    st.caption(f"**Takeaway** · {text}")


def render_insights_tab(
    manifest: dict,
    artifacts: dict,
    dataset: pd.DataFrame,
    presets: pd.DataFrame,
) -> None:
    """Render the model and dataset insights tab."""
    st.markdown(
        "How the models score, where output comes from, and what they rely on."
    )

    st.subheader("Model comparison")
    comparison = model_comparison(artifacts)
    st.dataframe(
        comparison,
        hide_index=True,
        column_config={
            "CV R2": st.column_config.NumberColumn("CV R²", format="%.4f"),
            "CV MAE": st.column_config.NumberColumn("CV MAE (kW)", format="%,.0f"),
            "CV RMSE": st.column_config.NumberColumn("CV RMSE (kW)", format="%,.0f"),
            "Test R2": st.column_config.NumberColumn("Holdout R²", format="%.4f"),
            "Test MAE": st.column_config.NumberColumn("Holdout MAE (kW)", format="%,.0f"),
            "Test RMSE": st.column_config.NumberColumn("Holdout RMSE (kW)", format="%,.0f"),
        },
    )

    ranked = comparison.sort_values("CV R2", ascending=False).reset_index(drop=True)
    best, runner_up = ranked.iloc[0], ranked.iloc[1]
    _takeaway(
        f"{best['Model']} leads on cross-validated R² ({best['CV R2']:.4f}) by "
        f"{abs(best['CV R2'] - runner_up['CV R2']):.4f} over {runner_up['Model']}. "
        "CV ranks the models; the holdout columns are the chronological report."
    )

    st.subheader("Held-out error by plant")
    test_start = pd.Timestamp(manifest["split"]["test_start"])
    test_rows = dataset[dataset["DATE_TIME"] >= test_start]
    per_plant = per_plant_metrics(artifacts, test_rows)
    st.dataframe(
        per_plant,
        hide_index=True,
        column_config={
            "MAE": st.column_config.NumberColumn("MAE (kW)", format="%,.0f"),
            "RMSE": st.column_config.NumberColumn("RMSE (kW)", format="%,.0f"),
            "R2": st.column_config.NumberColumn("R²", format="%.4f"),
        },
    )

    worst = per_plant.sort_values("RMSE", ascending=False).iloc[0]
    _takeaway(
        f"The largest held-out miss is {worst['Model']} on {worst['Plant']} "
        f"(RMSE {worst['RMSE']:,.0f} kW over {len(test_rows):,} rows). "
        "Pooling did not hide the difference between the plants."
    )

    st.subheader("Where output comes from")
    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Mean daylight output by plant**")
            plant_means = (
                dataset.groupby("PLANT_NAME")[TARGET_COLUMN].mean().reset_index()
            )
            st.altair_chart(output_by_plant_chart(plant_means), width="stretch")

    with right:
        with st.container(border=True):
            st.markdown("**Mean daylight output by hour**")
            hour_means = dataset.groupby("HOUR")[TARGET_COLUMN].mean().reset_index()
            st.altair_chart(output_by_hour_chart(hour_means), width="stretch")

    ranked_plants = plant_means.sort_values(TARGET_COLUMN, ascending=False)
    top, bottom = ranked_plants.iloc[0], ranked_plants.iloc[-1]
    _takeaway(
        f"{top['PLANT_NAME']} averages {top[TARGET_COLUMN] / bottom[TARGET_COLUMN]:.2f}× "
        f"{bottom['PLANT_NAME']}'s daylight output, and the hourly curve peaks around "
        "midday — the two facts the model must reproduce before any weather detail matters."
    )

    with st.container(border=True):
        st.markdown("**Daily mean daylight output**")
        daily = dataset.groupby("DATE")[TARGET_COLUMN].mean().reset_index()
        daily["DATE"] = pd.to_datetime(daily["DATE"])
        st.altair_chart(daily_output_chart(daily), width="stretch")

    _takeaway(
        f"Daily means span {daily[TARGET_COLUMN].min():,.0f}–"
        f"{daily[TARGET_COLUMN].max():,.0f} kW across {len(daily)} days. That day-to-day "
        "spread is what the chronological holdout has to absorb."
    )

    st.subheader("What the models rely on")
    importance_tables = {
        label: feature_importance(artifact) for label, artifact in artifacts.items()
    }
    columns = st.columns(len(artifacts), gap="large")
    for column, (label, table) in zip(columns, importance_tables.items()):
        with column:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                if table.empty:
                    st.caption("This model does not expose feature importance.")
                else:
                    st.altair_chart(importance_chart(table), width="stretch")

    leaders = {
        label: table.iloc[0]["feature"]
        for label, table in importance_tables.items()
        if not table.empty
    }
    if leaders:
        unique = set(leaders.values())
        if len(unique) == 1:
            _takeaway(
                f"Both models rank `{unique.pop()}` first. Importance ranks influence, "
                "not causation; correlated inputs can share the credit by order."
            )
        else:
            detail = ", ".join(f"{label}: `{feature}`" for label, feature in leaders.items())
            _takeaway(
                f"The models disagree on the top feature ({detail}). Where that happens, "
                "the cause is usually correlated inputs satisfying the same split."
            )

    st.subheader("Irradiation and output")
    st.altair_chart(
        irradiation_scatter_chart(
            dataset[["IRRADIATION", TARGET_COLUMN, "PLANT_NAME"]]
        ),
        width="stretch",
    )

    feature_columns = [f for f in manifest["feature_columns"] if f in dataset.columns]
    correlation = dataset[feature_columns + [TARGET_COLUMN]].corr(method="spearman")
    st.dataframe(
        correlation.round(3),
        column_config={
            column: st.column_config.NumberColumn(column, format="%.3f")
            for column in correlation.columns
        },
    )

    if "HOUR" in correlation.index:
        hour_correlation = correlation.loc["HOUR", TARGET_COLUMN]
        _takeaway(
            f"`HOUR`'s rank correlation with output is {hour_correlation:+.3f}, yet the "
            "model keeps it: within a plant the irradiation–output relationship bends "
            "over the day, and trees represent that where a monotone coefficient cannot."
        )
    else:
        _takeaway(
            "Spearman correlation captures monotone relationships only; a feature can "
            "matter to a tree split without ranking high here."
        )
