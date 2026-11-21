"""AC Power Studio — Streamlit entry point."""

from __future__ import annotations

import streamlit as st

from ac_power.analytics import load_artifacts, load_manifest
from ac_power.config import MANIFEST_PATH
from ac_power.data import load_dataset, plant_presets
from ac_power.views import (
    render_about_tab,
    render_batch_tab,
    render_insights_tab,
    render_predict_tab,
)

st.set_page_config(
    page_title="AC Power Studio",
    page_icon=":material/solar_power:",
    layout="wide",
)


@st.cache_resource
def load_bundles() -> tuple[dict, dict]:
    """Load the manifest and every bundle once per session."""
    manifest = load_manifest(MANIFEST_PATH)
    return manifest, load_artifacts(manifest)


@st.cache_data
def load_reference_data() -> tuple:
    """Load the analysis frame and its plant/hour presets once per session."""
    dataset = load_dataset()
    return dataset, plant_presets(dataset)


if not MANIFEST_PATH.exists():
    st.error(
        f"Model manifest not found: {MANIFEST_PATH}. Run "
        "`Prediction of AC-Power.ipynb` end to end to regenerate `models/`, then reload."
    )
    st.stop()

try:
    manifest, artifacts = load_bundles()
except (ValueError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()

dataset, presets = load_reference_data()

st.title("AC Power Studio")
st.caption(
    "Predict plant-level AC output from irradiation, air temperature, hour, and plant "
    "identity — the inputs a dispatch team can know before the interval starts."
)

with st.container(horizontal=True):
    for label in artifacts:
        st.badge(label, icon=":material/model_training:", color="orange")
    st.badge(
        f"{len(manifest['feature_columns'])} features",
        icon=":material/tune:",
        color="gray",
    )
    st.badge(
        f"{len(dataset):,} daylight observations",
        icon=":material/database:",
        color="gray",
    )

tab_predict, tab_batch, tab_insights, tab_about = st.tabs(
    ["Predict", "Batch", "Insights", "About"]
)

with tab_predict:
    render_predict_tab(manifest, artifacts, dataset, presets)

with tab_batch:
    render_batch_tab(manifest, artifacts, dataset, presets)

with tab_insights:
    render_insights_tab(manifest, artifacts, dataset, presets)

with tab_about:
    render_about_tab(manifest, artifacts, dataset, presets)
