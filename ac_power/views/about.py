"""About and model card view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ac_power.analytics import model_comparison
from ac_power.schema import TARGET_COLUMN, get_feature_definitions


def _feature_table(manifest: dict) -> pd.DataFrame:
    """Data dictionary with a column saying whether the model consumes each feature."""
    used = set(manifest["feature_columns"])
    rows = []
    for name, meta in get_feature_definitions().items():
        if name == TARGET_COLUMN:
            status = "Target"
        elif name in used:
            status = "Yes"
        else:
            status = "No"

        rows.append(
            {
                "Feature": name,
                "Description": meta["description"],
                "Type": f"{meta['data_type']} · {meta['specific_type']}",
                "Used by model": status,
            }
        )
    return pd.DataFrame(rows)


def render_about_tab(
    manifest: dict,
    artifacts: dict,
    dataset: pd.DataFrame,
    presets: pd.DataFrame,
) -> None:
    """Render the project, model card, and limitations."""
    features = list(manifest["feature_columns"])

    st.markdown(
        "AC output is intermittent: it depends on irradiance, panel temperature, air "
        "temperature, the time of day, and the plant itself. AC Power Studio models that "
        "relationship directly, so a dispatch team can estimate output from conditions "
        "it can know before the interval starts."
    )

    st.markdown("**Why this matters**")
    st.markdown(
        "Grid operators schedule generation in blocks. A forecast that is accurate at the "
        "plant level lets them reserve the right amount of backup and decide when "
        "curtailment is worth considering. The hard part is not panel technology; it is "
        "knowing how much power a specific plant will deliver on a specific afternoon."
    )

    comparison = model_comparison(artifacts)
    best = comparison.sort_values("CV R2", ascending=False).iloc[0]

    with st.container(horizontal=True):
        st.metric("Best CV R²", f"{best['CV R2']:.4f}", border=True)
        st.metric("Best CV MAE", f"{best['CV MAE']:,.0f} kW", border=True)
        st.metric("Daylight observations", f"{len(dataset):,}", border=True)
        st.metric("Plants", f"{dataset['PLANT_ID'].nunique()}", border=True)

    with st.container(border=True):
        st.markdown("**Model card**")
        st.markdown(
            f"- **Algorithms** · {' and '.join(artifact['label'] for artifact in artifacts.values())}\n"
            f"- **Target** · `{TARGET_COLUMN}` in kW, summed across each plant's 22 inverters\n"
            f"- **Features** · {len(features)} numeric: "
            + ", ".join(f"`{feature}`" for feature in features)
            + "\n"
            f"- **Training data** · {manifest['rows']['train']:,} daylight rows from two "
            "Indian plants, 15-minute intervals, 2020-05-15 to 2020-06-17\n"
            f"- **Evaluation** · 5-fold, 3-repeat cross-validation inside the training "
            f"window, plus a chronological holdout of {manifest['rows']['test']:,} rows "
            f"starting {manifest['split']['test_start']}\n"
            "- **Artifacts** · "
            + ", ".join(f"`models/{entry['file']}`" for entry in manifest["models"])
        )
        st.caption(
            "Each artifact is a bundle: model, feature order, stored metrics, label, and "
            "target. Re-running the notebook regenerates both bundles and the manifest."
        )

    st.subheader("Data dictionary")
    st.caption(
        "The feature list is whatever the notebook's feature-selection stage chose. "
        "`MODULE_TEMPERATURE` exists in the source but is excluded because it is a "
        "panel-mounted sensor reading, not something available before dispatch."
    )
    st.dataframe(
        _feature_table(manifest),
        hide_index=True,
        column_config={
            "Description": st.column_config.TextColumn(width="large"),
            "Used by model": st.column_config.TextColumn(width="small"),
        },
    )

    with st.expander("Source data preview"):
        st.dataframe(dataset.head(10), hide_index=True)

    st.subheader("Limitations")
    st.markdown(
        "- **Thirty-four days, one season.** The holdout is a single weather week; a "
        "different week would move the number.\n"
        "- **Irradiation dominance.** Within daylight, irradiation correlates 0.99 with "
        "Plant 1's output and 0.75 with Plant 2's. The model's contribution is the shape "
        "around that relationship, not the order of magnitude.\n"
        "- **Nine Plant 2 outages** report zero output under strong irradiation. They are "
        "unpredictable from weather and sit in the residuals by construction.\n"
        "- **Plant 1's DC channel is unusable**, so no DC-side or efficiency analysis is "
        "attempted.\n"
        "- **`PLANT_ID` is two-level.** The model interpolates between these two plants; "
        "it says nothing about a third.\n"
        "- **Daylight only.** Night output is defined as zero, not predicted.\n"
        "- **One chronological window.** A rolling-origin evaluation would give a "
        "distribution instead of a single number."
    )

    st.subheader("Source")
    st.markdown(
        "- Kaggle: [Solar Power Generation Data]"
        "(https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) — two "
        "Indian plants, 15-minute generation and sensor readings.\n"
        "- Notebook: `Prediction of AC-Power.ipynb` — the six-stage analysis that "
        "produces these artifacts."
    )

    st.subheader("Author")
    st.markdown(
        "Developed by **Muhammad Cikal Merdeka** · Data Analyst / Data Scientist\n\n"
        "[GitHub](https://github.com/mcikalmerdeka) · "
        "[LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)"
    )
    st.caption(
        "Light and dark themes are available from the app menu (⋮ → Settings → Theme)."
    )
