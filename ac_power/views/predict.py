"""Individual prediction view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ac_power.analytics import contributions, output_band, predict_power
from ac_power.data import preset_for
from ac_power.schema import (
    CATEGORICAL_FEATURES,
    DISPLAY_NAMES,
    INPUT_BOUNDS,
    PLANT_LABELS,
)
from ac_power.ui import contribution_chart, gauge_chart

PLANT_KEY = "predict_plant"
HOUR_KEY = "predict_hour"
MODEL_KEY = "predict_model"
RESULT_KEY = "predict_result"
SIGNATURE_KEY = "predict_signature"

HOURS = list(range(6, 19))


def _input_key(feature: str) -> str:
    """Session-state key for one numeric form input."""
    return f"predict_input_{feature}"


def _numeric_features(feature_columns: list[str]) -> list[str]:
    """Features that get a number input rather than a selector."""
    return [
        feature
        for feature in feature_columns
        if feature not in CATEGORICAL_FEATURES and feature != "HOUR"
    ]


def _sync_preset(
    plant: int,
    hour: int,
    preset: dict,
    numeric_features: list[str],
) -> None:
    """Refill the weather inputs when the plant/hour selection changes."""
    signature = (plant, hour)
    if st.session_state.get(SIGNATURE_KEY) == signature:
        return

    for feature in numeric_features:
        st.session_state[_input_key(feature)] = float(preset[feature])
    st.session_state[SIGNATURE_KEY] = signature
    st.session_state.pop(RESULT_KEY, None)


def _render_result(result: dict, artifacts: dict) -> None:
    predictions = result["predictions"]
    selected = result["model"]

    st.markdown("### Prediction")
    with st.container(horizontal=True):
        st.metric(
            f"Predicted AC power · {selected}",
            f"{predictions[selected]:,.1f} kW",
            border=True,
        )
        st.metric("Band", output_band(predictions[selected]), border=True)
        st.metric("vs typical", f"{result['total']:+,.1f} kW", border=True)
        st.metric("Plant", result["plant"], border=True)
        st.metric("Hour", f"{result['hour']:02d}:00", border=True)

    st.altair_chart(gauge_chart(predictions[selected]), width="stretch")
    st.caption(
        "Track spans 0–30,000 kW. Night output is 0 kW by definition and is outside "
        "this model's scope."
    )

    with st.container(border=True):
        st.markdown("**What moved this prediction**")
        st.altair_chart(contribution_chart(result["breakdown"]), width="stretch")

        table = result["breakdown"].copy()
        table["Change (kW)"] = table["delta"].round(1)
        st.dataframe(
            table[["display", "input", "baseline", "Change (kW)"]].rename(
                columns={
                    "display": "Feature",
                    "input": "This value",
                    "baseline": "Typical value",
                }
            ),
            hide_index=True,
            column_config={
                "This value": st.column_config.NumberColumn(format="%.2f"),
                "Typical value": st.column_config.NumberColumn(format="%.2f"),
                "Change (kW)": st.column_config.NumberColumn(format="%+,.1f"),
            },
        )
        st.caption(
            f"Reference: median conditions across {result['n_rows']:,} daylight rows, "
            f"with the plant fixed to {result['plant']}. Features are applied one at a "
            f"time in model order, so the changes sum to {result['total']:+,.1f} kW."
        )

    if len(predictions) > 1:
        comparison = pd.DataFrame(
            {
                "Model": list(predictions),
                "Predicted AC power (kW)": [
                    round(value, 1) for value in predictions.values()
                ],
            }
        )
        st.dataframe(comparison, hide_index=True)

    with st.expander(f"Model input ({result['model_input'].shape[1]} features)"):
        st.dataframe(result["model_input"], hide_index=True)

    st.caption(
        "Plant identity is a model input: the two plants differ in capacity, so the "
        "same weather does not imply the same output."
    )


def render_predict_tab(
    manifest: dict,
    artifacts: dict,
    dataset: pd.DataFrame,
    presets: pd.DataFrame,
) -> None:
    """Render the single-prediction tab."""
    features = list(manifest["feature_columns"])
    numeric = _numeric_features(features)

    st.markdown(
        "Score the expected AC output for a plant and hour. Pick a plant and hour to "
        "prefill that hour's median conditions, then adjust the weather."
    )

    left, right = st.columns(2, gap="large")
    with left:
        plant = st.selectbox(
            "Plant",
            options=[1, 2],
            format_func=lambda value: PLANT_LABELS[value],
            key=PLANT_KEY,
        )
    with right:
        hour = st.selectbox("Hour of day", HOURS, index=6, key=HOUR_KEY)

    preset = preset_for(presets, dataset, plant, hour, features)
    _sync_preset(plant, hour, preset, numeric)

    st.caption(
        f"{PLANT_LABELS[plant]} at {hour:02d}:00 — inputs prefilled with this plant's "
        "median conditions for this hour."
    )

    with st.form("predict_form", border=True):
        st.markdown("**Weather conditions**")
        st.caption(
            "Values are bounded to the range observed in the daylight training data, "
            "so the model never extrapolates from this form."
        )

        columns = st.columns(2, gap="large")
        for index, feature in enumerate(numeric):
            low, high, step = INPUT_BOUNDS[feature]
            with columns[index % 2]:
                st.number_input(
                    DISPLAY_NAMES[feature],
                    min_value=float(low),
                    max_value=float(high),
                    step=float(step),
                    format="%.2f",
                    key=_input_key(feature),
                )

        model = st.segmented_control(
            "Model",
            list(artifacts),
            default=list(artifacts)[0],
            key=MODEL_KEY,
        )
        submitted = st.form_submit_button(
            "Predict AC power", type="primary", icon=":material/bolt:"
        )

    if submitted:
        model = model or list(artifacts)[0]
        inputs = {
            feature: float(st.session_state[_input_key(feature)])
            for feature in numeric
        }
        if "HOUR" in features:
            inputs["HOUR"] = float(hour)
        if "PLANT_ID" in features:
            inputs["PLANT_ID"] = float(plant)

        row = pd.Series(inputs)[features]

        try:
            model_input = row.to_frame().T
            predictions = {
                name: predict_power(artifact, model_input)
                for name, artifact in artifacts.items()
            }
            baseline = dataset[features].median()
            if "PLANT_ID" in features:
                baseline["PLANT_ID"] = float(plant)
            breakdown = contributions(artifacts[model], row, baseline)
        except Exception as exc:  # surface the reason instead of blanking the page
            st.error(f"Could not score these conditions: {exc}")
        else:
            st.session_state[RESULT_KEY] = {
                "predictions": predictions,
                "model": model,
                "breakdown": breakdown,
                "total": float(breakdown["delta"].sum()),
                "model_input": model_input,
                "plant": PLANT_LABELS[plant],
                "hour": hour,
                "n_rows": len(dataset),
            }

    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.caption("Submit the form to score these conditions.")
        return

    _render_result(result, artifacts)
