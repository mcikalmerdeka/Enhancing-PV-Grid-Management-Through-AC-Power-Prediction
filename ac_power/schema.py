"""Dataset schema and feature metadata for AC Power Studio.

This module is the single source of truth for column names, display labels,
input bounds, and the data dictionary. It has no Streamlit or modelling
import, so data, analytics, and views can all import it without a cycle.

The deployed feature list is not defined here: it comes from
``models/manifest.json``, because the notebook decides which inputs survive
feature selection. This module only knows how to describe and bound the
features that can appear in that list.
"""

from __future__ import annotations

from typing import Dict, Tuple

TARGET_COLUMN = "AC_POWER"

# Daylight rows only; night output is defined as zero, not predicted.
DAYLIGHT_THRESHOLD = 0.05

PLANT_LABELS: Dict[int, str] = {1: "Plant 1", 2: "Plant 2"}
PLANT_SOURCE_IDS: Dict[int, int] = {4135001: 1, 4135002: 2}

# Features that can appear in a manifest produced by the notebook.
KNOWN_FEATURES: Tuple[str, ...] = (
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "HOUR",
    "PLANT_ID",
)

CATEGORICAL_FEATURES: Dict[str, set] = {"PLANT_ID": {1, 2}}

DISPLAY_NAMES: Dict[str, str] = {
    "IRRADIATION": "Irradiation (kW/m²)",
    "AMBIENT_TEMPERATURE": "Ambient temperature (°C)",
    "HOUR": "Hour of day",
    "PLANT_ID": "Plant",
    "MODULE_TEMPERATURE": "Module temperature (°C)",
    "AC_POWER": "AC power (kW)",
    "DATE_TIME": "Timestamp",
    "DC_POWER": "DC power (kW)",
}

# (minimum, maximum, step) for the numeric form inputs, from the training range.
INPUT_BOUNDS: Dict[str, Tuple[float, float, float]] = {
    "IRRADIATION": (0.05, 1.25, 0.01),
    "AMBIENT_TEMPERATURE": (20.0, 40.0, 0.5),
    "HOUR": (6.0, 18.0, 1.0),
}

# Coarse labels used in result summaries (kW).
OUTPUT_BAND_LOW = 5_000.0
OUTPUT_BAND_HIGH = 15_000.0


def feature_label(name: str) -> str:
    """Return the human-readable label for a feature, falling back to its name."""
    return DISPLAY_NAMES.get(name, name)


def get_feature_definitions() -> Dict[str, Dict[str, str]]:
    """Data dictionary rendered by the About tab."""
    return {
        "DATE_TIME": {
            "description": "15-minute interval timestamp",
            "data_type": "Temporal",
            "specific_type": "Interval",
        },
        "PLANT_ID": {
            "description": "Plant identifier (1 or 2); the pooled model's only plant signal",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "IRRADIATION": {
            "description": "Plane-of-array irradiance; the dominant physical driver",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "AMBIENT_TEMPERATURE": {
            "description": "Air temperature at the plant; also a proxy for panel heating",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "MODULE_TEMPERATURE": {
            "description": "Panel-mounted sensor reading; excluded because it is not available before dispatch",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "HOUR": {
            "description": "Hour of day; captures sun angle and thermal lag",
            "data_type": "Numerical",
            "specific_type": "Cyclical",
        },
        "DC_POWER": {
            "description": "Inverter DC input; unusable at Plant 1 and excluded from every model",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        TARGET_COLUMN: {
            "description": "Plant-level AC output, summed across 22 inverters (kW) — the target",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
    }
