"""Batch scoring view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ac_power.analytics import out_of_range_counts, output_band
from ac_power.config import BATCH_EXAMPLE_PATH
from ac_power.data import prepare_batch
from ac_power.schema import INPUT_BOUNDS, feature_label

OUTPUT_COLUMN = "Predicted AC power (kW)"
BAND_COLUMN = "Band"


def _render_results(display: pd.DataFrame, predictions: list[float]) -> None:
    """Render summary metrics, the scored table, and the download button."""
    results = display.copy()
    results[OUTPUT_COLUMN] = predictions
    results[BAND_COLUMN] = [output_band(value) for value in predictions]

    with st.container(horizontal=True):
        st.metric("Rows scored", f"{len(results):,}", border=True)
        st.metric(
            "Mean output",
            f"{sum(predictions) / len(predictions):,.1f} kW",
            border=True,
        )
        st.metric("Peak output", f"{max(predictions):,.1f} kW", border=True)

    st.dataframe(
        results,
        hide_index=True,
        column_config={
            OUTPUT_COLUMN: st.column_config.ProgressColumn(
                "Predicted AC power",
                format="%.1f kW",
                min_value=0,
                max_value=30_000,
            )
        },
    )

    st.download_button(
        "Download predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="ac_power_predictions.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def render_batch_tab(
    manifest: dict,
    artifacts: dict,
    dataset: pd.DataFrame,
    presets: pd.DataFrame,
) -> None:
    """Render the batch scoring tab."""
    features = list(manifest["feature_columns"])

    st.markdown(
        "Upload weather observations and score every row in one pass. The file needs "
        "the model's feature columns; any other columns are ignored."
    )

    source = (
        st.segmented_control(
            "Data source",
            ["Upload CSV", "Example data"],
            default="Upload CSV",
            key="batch_source",
        )
        or "Upload CSV"
    )

    raw: pd.DataFrame | None = None
    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "Weather CSV",
            type="csv",
            help="Same columns as the model features. The target and timestamp are optional.",
        )
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read the CSV: {exc}")
                return
    else:
        raw = pd.read_csv(BATCH_EXAMPLE_PATH)
        st.caption("Using the 12-row example from data/batch_example_ac_power.csv.")

    if raw is None:
        with st.expander("Expected columns"):
            st.markdown("\n".join(f"- `{feature}`" for feature in features))
            st.caption(
                "`PLANT_ID` must be 1 or 2. The source ids 4135001 and 4135002 are "
                "not accepted."
            )
        return

    if raw.empty:
        st.error("The uploaded file has no rows.")
        return

    try:
        prepared = prepare_batch(raw, features)
    except ValueError as exc:
        st.error(f"Could not score this file: {exc}")
        return
    except Exception as exc:
        st.error(f"Could not prepare the input: {exc}")
        return

    outside = out_of_range_counts(prepared, INPUT_BOUNDS)
    if outside:
        details = ", ".join(
            f"{feature_label(name)} ({count} row(s))" for name, count in outside.items()
        )
        st.warning(
            "Values outside the training range were found: "
            + details
            + ". Those predictions are extrapolations and should be treated as "
            "indicative only."
        )

    model_name = st.selectbox("Model", list(artifacts), key="batch_model")
    artifact = artifacts[model_name]

    try:
        predictions = [float(value) for value in artifact["model"].predict(prepared)]
    except Exception as exc:
        st.error(f"Could not score the batch: {exc}")
        return

    display_columns = [
        column for column in raw.columns if column in features or column == "DATE_TIME"
    ]
    display = raw.loc[prepared.index, display_columns]
    _render_results(display, predictions)
