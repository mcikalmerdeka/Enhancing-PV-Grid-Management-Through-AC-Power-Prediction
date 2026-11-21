"""Model and dataset analytics for AC Power Studio.

Pure functions with no Streamlit dependency, so the views and the tests use
the same code. Metrics are read with a missing-key fallback: a bundle produced
by an older notebook should render as "not available", not crash the page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from . import config, schema

ARTIFACT_KEYS = ("model", "feature_columns", "metrics", "label", "target")


def load_manifest(path: Path = config.MANIFEST_PATH) -> Dict[str, Any]:
    """Load and validate the notebook's manifest."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model manifest not found: {path}. Run the notebook end to end to "
            "regenerate models/ and manifest.json."
        )

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"{path.name} could not be read ({type(exc).__name__}). "
            "Re-run the notebook to regenerate it."
        ) from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"{path.name} is not a manifest object.")

    missing = [key for key in ("target", "feature_columns", "split", "models") if key not in manifest]
    if missing:
        raise ValueError(f"{path.name} is missing: {', '.join(missing)}.")

    if manifest["target"] != schema.TARGET_COLUMN:
        raise ValueError(
            f"{path.name} declares target {manifest['target']!r}, but this app "
            f"scores {schema.TARGET_COLUMN!r}."
        )

    if not manifest["feature_columns"]:
        raise ValueError(f"{path.name} declares an empty feature list.")

    return manifest


def load_artifact(
    path: Path,
    feature_columns: List[str],
    target: str = schema.TARGET_COLUMN,
) -> Dict[str, Any]:
    """Load a notebook bundle and refuse anything that is not one."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run the notebook end to end to "
            "regenerate models/."
        )

    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ValueError(
            f"{path.name} could not be read ({type(exc).__name__}). "
            "Re-run the notebook to regenerate it."
        ) from exc

    if not isinstance(bundle, dict):
        raise ValueError(
            f"{path.name} is not a notebook bundle (loaded object is not a dictionary). "
            "Re-run the notebook to regenerate it."
        )

    missing = [key for key in ARTIFACT_KEYS if key not in bundle]
    if missing:
        raise ValueError(
            f"{path.name} is not a notebook bundle (missing: {', '.join(missing)}). "
            "Re-run the notebook to regenerate it."
        )

    if list(bundle["feature_columns"]) != list(feature_columns):
        raise ValueError(
            f"{path.name} was trained on features {list(bundle['feature_columns'])}, "
            f"but the manifest declares {list(feature_columns)}. "
            "Re-run the notebook to regenerate both."
        )

    if bundle["target"] != target:
        raise ValueError(
            f"{path.name} scores {bundle['target']!r}, but this app scores {target!r}."
        )

    return bundle


def load_artifacts(manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Load every bundle the manifest lists, keyed by label."""
    feature_columns = list(manifest["feature_columns"])
    artifacts: Dict[str, Dict[str, Any]] = {}
    for entry in manifest["models"]:
        artifacts[entry["label"]] = load_artifact(
            config.MODELS_DIR / entry["file"], feature_columns
        )
    return artifacts


def predict_power(artifact: Mapping[str, Any], frame: pd.DataFrame) -> float:
    """Predict AC power in kW for one or more rows, returning the first value."""
    model_input = frame[list(artifact["feature_columns"])]
    return float(artifact["model"].predict(model_input)[0])


def final_estimator(model: Any) -> Any:
    """Return the last step of a pipeline, or the model itself."""
    steps = getattr(model, "named_steps", None)
    if steps:
        return list(steps.values())[-1]
    return model


def contributions(
    artifact: Mapping[str, Any],
    row: pd.Series,
    baseline: pd.Series,
) -> pd.DataFrame:
    """Attribute the gap between this prediction and a reference prediction.

    Walks the features in the bundle's order: start from the baseline row, set
    one feature to this row's value, and record how the prediction moved. The
    deltas sum exactly to ``predict(row) - predict(baseline)``. The
    decomposition is order-dependent; correlated features share credit by
    order, which is inherent to additive attribution of a non-additive model.
    """
    features = list(artifact["feature_columns"])
    current = baseline.copy()
    previous = predict_power(artifact, current.to_frame().T)

    records = []
    for feature in features:
        current[feature] = row[feature]
        updated = predict_power(artifact, current.to_frame().T)
        records.append(
            {
                "feature": feature,
                "display": schema.feature_label(feature),
                "input": float(row[feature]),
                "baseline": float(baseline[feature]),
                "delta": updated - previous,
            }
        )
        previous = updated

    table = pd.DataFrame(records)
    order = table["delta"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def _metric(metrics: Mapping[str, Any], group: str, name: str) -> float:
    """Read one stored metric, tolerating bundles that predate it."""
    try:
        if group == "cv":
            return float(metrics["cv"][name]["test_mean"])
        return float(metrics["test"][name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def model_comparison(artifacts: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Side-by-side CV and chronological holdout metrics for every bundle."""
    rows = []
    for label, artifact in artifacts.items():
        metrics = artifact.get("metrics", {})
        rows.append(
            {
                "Model": label,
                "CV R2": _metric(metrics, "cv", "r2"),
                "CV MAE": _metric(metrics, "cv", "mae"),
                "CV RMSE": _metric(metrics, "cv", "rmse"),
                "Test R2": _metric(metrics, "test", "r2"),
                "Test MAE": _metric(metrics, "test", "mae"),
                "Test RMSE": _metric(metrics, "test", "rmse"),
            }
        )
    return pd.DataFrame(rows)


def per_plant_metrics(
    artifacts: Mapping[str, Mapping[str, Any]],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Held-out error per model and plant, so pooling cannot hide one plant."""
    rows = []
    for label, artifact in artifacts.items():
        features = list(artifact["feature_columns"])
        predictions = artifact["model"].predict(frame[features])
        scored = frame.assign(_prediction=predictions)
        for plant_id, group in scored.groupby("PLANT_ID"):
            rows.append(
                {
                    "Model": label,
                    "Plant": schema.PLANT_LABELS.get(int(plant_id), str(plant_id)),
                    "MAE": mean_absolute_error(group[schema.TARGET_COLUMN], group._prediction),
                    "RMSE": root_mean_squared_error(
                        group[schema.TARGET_COLUMN], group._prediction
                    ),
                    "R2": r2_score(group[schema.TARGET_COLUMN], group._prediction),
                }
            )
    return pd.DataFrame(rows)


def feature_importance(artifact: Mapping[str, Any]) -> pd.DataFrame:
    """Importance for tree bundles, normalised coefficient magnitude for linear."""
    model = final_estimator(artifact["model"])
    features = list(artifact["feature_columns"])

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
        total = float(values.sum())
        values = values / total if total else values
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    if len(values) != len(features):
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": features, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def out_of_range_counts(
    frame: pd.DataFrame,
    bounds: Mapping[str, tuple],
) -> Dict[str, int]:
    """Count rows outside the training range, treating NaN as outside."""
    counts: Dict[str, int] = {}
    for feature, (low, high, _step) in bounds.items():
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        outside = values.isna() | (values < low) | (values > high)
        if int(outside.sum()):
            counts[feature] = int(outside.sum())
    return counts


def output_band(value: float) -> str:
    """Coarse label for a predicted output, used in result summaries."""
    if value >= schema.OUTPUT_BAND_HIGH:
        return "High"
    if value >= schema.OUTPUT_BAND_LOW:
        return "Moderate"
    return "Low"


def extrapolation_flags(row: pd.Series, bounds: Mapping[str, tuple]) -> List[str]:
    """Name the features whose value sits outside the training range."""
    flagged = []
    for feature, (low, high, _step) in bounds.items():
        if feature in row and not (low <= float(row[feature]) <= high):
            flagged.append(feature)
    return flagged


__all__ = [
    "ARTIFACT_KEYS",
    "contributions",
    "extrapolation_flags",
    "feature_importance",
    "final_estimator",
    "load_artifact",
    "load_artifacts",
    "load_manifest",
    "model_comparison",
    "out_of_range_counts",
    "output_band",
    "per_plant_metrics",
    "predict_power",
]
