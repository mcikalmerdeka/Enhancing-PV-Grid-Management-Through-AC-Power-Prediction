"""Dataset loading and feature assembly for AC Power Studio.

Pure pandas functions with no Streamlit dependency. Caching belongs to the
entry point, so these stay cheap to call from tests and scripts.

The aggregation here mirrors the notebook exactly: AC power is summed across
inverters, weather is merged one-to-one on timestamp, and only daylight rows
survive. A different row count means the two sides have diverged.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import config, schema

DATASET_COLUMNS = [
    "DATE_TIME",
    "AC_POWER",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "PLANT_ID",
    "PLANT_NAME",
    "DATE",
    "HOUR",
]

PRESET_FEATURES = [
    feature
    for feature in schema.KNOWN_FEATURES
    if feature not in schema.CATEGORICAL_FEATURES and feature != "HOUR"
]


def _parse_generation_dates(values: pd.Series) -> pd.Series:
    """Parse a generation timestamp column, whose format differs by plant.

    Plant 1 stores ``15-05-2020 00:00`` (day first); Plant 2 stores
    ``2020-05-15 00:00:00`` (ISO). Inferring with ``dayfirst=True`` emits a
    warning for the ISO file and ``format="mixed", dayfirst=True`` silently
    swaps month and day, so the format is chosen from the first value instead.
    """
    sample = str(values.iloc[0])
    if sample[:4].isdigit() and sample[4:5] == "-":
        return pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S")
    return pd.to_datetime(values, format="%d-%m-%Y %H:%M")


def load_generation(plant_id: int) -> pd.DataFrame:
    """Load one plant's per-inverter generation file."""
    frame = pd.read_csv(config.GENERATION_PATHS[plant_id])
    frame["DATE_TIME"] = _parse_generation_dates(frame["DATE_TIME"])
    return frame


def load_sensor(plant_id: int) -> pd.DataFrame:
    """Load one plant's per-plant weather file."""
    return pd.read_csv(config.SENSOR_PATHS[plant_id], parse_dates=["DATE_TIME"])


def load_dataset() -> pd.DataFrame:
    """Load both plants as daylight, plant-level rows."""
    frames = []
    for plant_id in config.PLANT_IDS:
        generation = load_generation(plant_id)
        sensor = load_sensor(plant_id)

        ac_only = generation.groupby("DATE_TIME", as_index=False)[["AC_POWER"]].sum()
        weather = sensor.drop(columns=["PLANT_ID", "SOURCE_KEY"], errors="ignore")
        merged = ac_only.merge(weather, on="DATE_TIME", how="inner")
        merged["PLANT_ID"] = plant_id
        frames.append(merged)

    dataset = pd.concat(frames, ignore_index=True)
    dataset["PLANT_NAME"] = dataset["PLANT_ID"].map(schema.PLANT_LABELS)
    dataset["DATE"] = dataset["DATE_TIME"].dt.date
    dataset["HOUR"] = dataset["DATE_TIME"].dt.hour

    dataset = dataset[dataset["IRRADIATION"] > schema.DAYLIGHT_THRESHOLD]
    dataset = dataset.sort_values(["DATE_TIME", "PLANT_ID"]).reset_index(drop=True)
    return dataset[DATASET_COLUMNS]


def plant_presets(dataset: pd.DataFrame) -> pd.DataFrame:
    """Median weather per plant and hour of day.

    Used to prefill the prediction form with conditions a plant actually saw,
    rather than with a neutral default.
    """
    return dataset.groupby(["PLANT_ID", "HOUR"], as_index=False)[
        PRESET_FEATURES
    ].median()


def preset_for(
    presets: pd.DataFrame,
    dataset: pd.DataFrame,
    plant_id: int,
    hour: int,
    feature_columns: List[str],
) -> Dict[str, float]:
    """Resolve conditions for a plant/hour, degrading gracefully.

    Order: the exact plant-hour cell, then the plant's overall median, then the
    dataset median. The result always carries every requested feature in the
    manifest's order, so it can be handed straight to a model.
    """
    match = presets[
        (presets["PLANT_ID"] == plant_id) & (presets["HOUR"] == hour)
    ]

    if not match.empty:
        source = match.iloc[0]
    else:
        plant_rows = dataset[dataset["PLANT_ID"] == plant_id]
        reference = plant_rows if not plant_rows.empty else dataset
        source = reference[PRESET_FEATURES].median()

    resolved: Dict[str, float] = {}
    for feature in feature_columns:
        if feature == "HOUR":
            resolved[feature] = float(hour)
        elif feature in schema.CATEGORICAL_FEATURES:
            resolved[feature] = float(plant_id)
        else:
            resolved[feature] = float(source[feature])

    return resolved


def to_model_matrix(frame: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Select and coerce the columns a persisted model expects."""
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError("missing required feature columns: " + ", ".join(missing))

    return frame[feature_columns].astype(float)


def prepare_batch(raw: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Turn an uploaded CSV into a scoring matrix.

    Exports often carry an index column and may include the target or the
    timestamp; extras are tolerated and ignored. Missing columns, non-numeric
    values, invalid plant ids, and NaN/inf are rejected with the column name so
    the caller can fix the file instead of guessing.
    """
    if raw is None or raw.empty:
        raise ValueError("the uploaded file has no rows")

    frame = raw.loc[
        :, [column for column in raw.columns if not str(column).startswith("Unnamed:")]
    ].copy()

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError("missing required feature columns: " + ", ".join(missing))

    for feature, allowed in schema.CATEGORICAL_FEATURES.items():
        if feature not in feature_columns:
            continue
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        invalid = numeric.isna() | ~numeric.isin(allowed)
        if invalid.any():
            sample = frame.loc[invalid, feature].iloc[0]
            raise ValueError(
                f"{feature} must be one of {sorted(allowed)}; got {sample!r}"
            )
        frame[feature] = numeric.astype(float)

    matrix = pd.DataFrame(index=frame.index)
    for feature in feature_columns:
        try:
            matrix[feature] = pd.to_numeric(frame[feature], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be numeric") from exc

    values = matrix.to_numpy()
    if not np.isfinite(values).all():
        bad = matrix.columns[~np.isfinite(values).all(axis=0)].tolist()
        raise ValueError(
            "feature columns contain NaN or infinite values: " + ", ".join(bad)
        )

    return matrix
