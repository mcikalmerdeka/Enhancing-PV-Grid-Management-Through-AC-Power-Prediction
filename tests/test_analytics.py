"""Tests for artifact loading, prediction, attribution, and metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from ac_power import analytics, config, data, schema


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = analytics.load_manifest()
        cls.artifacts = analytics.load_artifacts(cls.manifest)
        cls.features = list(cls.manifest["feature_columns"])
        cls.dataset = data.load_dataset()
        cls.peak = cls.dataset.iloc[[int(cls.dataset[schema.TARGET_COLUMN].argmax())]]
        cls.baseline = cls.dataset[cls.features].median()
        cls.baseline["PLANT_ID"] = float(cls.peak["PLANT_ID"].iloc[0])

    def test_manifest_contract(self) -> None:
        self.assertEqual(self.manifest["target"], schema.TARGET_COLUMN)
        self.assertEqual(len(self.manifest["models"]), 2)
        self.assertTrue(self.manifest["feature_columns"])
        pd.Timestamp(self.manifest["split"]["test_start"])

    def test_loads_both_bundles_by_label(self) -> None:
        self.assertEqual(len(self.artifacts), 2)
        self.assertEqual(
            set(self.artifacts),
            {entry["label"] for entry in self.manifest["models"]},
        )

    def test_predictions_are_finite_and_plausible(self) -> None:
        frame = self.peak[self.features]
        for label, artifact in self.artifacts.items():
            value = analytics.predict_power(artifact, frame)
            self.assertTrue(np.isfinite(value), label)
            self.assertGreaterEqual(value, 0.0, label)
            self.assertLessEqual(value, 30_000.0, label)

    def test_contributions_sum_to_the_prediction_gap(self) -> None:
        row = self.peak[self.features].iloc[0]
        for label, artifact in self.artifacts.items():
            breakdown = analytics.contributions(artifact, row, self.baseline)
            expected = analytics.predict_power(
                artifact, row.to_frame().T
            ) - analytics.predict_power(artifact, self.baseline.to_frame().T)
            self.assertAlmostEqual(
                float(breakdown["delta"].sum()),
                float(expected),
                places=6,
                msg=label,
            )

    def test_model_comparison_has_finite_metrics(self) -> None:
        table = analytics.model_comparison(self.artifacts)
        self.assertEqual(len(table), 2)
        for column in ("CV R2", "CV MAE", "CV RMSE", "Test R2", "Test MAE", "Test RMSE"):
            self.assertIn(column, table.columns)
            self.assertTrue(np.isfinite(table[column]).all(), column)

    def test_per_plant_metrics_covers_both_models_and_plants(self) -> None:
        test_rows = self.dataset[
            self.dataset["DATE_TIME"] >= pd.Timestamp(self.manifest["split"]["test_start"])
        ]
        table = analytics.per_plant_metrics(self.artifacts, test_rows)
        self.assertEqual(len(table), 4)
        self.assertEqual(set(table["Plant"]), {"Plant 1", "Plant 2"})
        self.assertTrue(np.isfinite(table["MAE"]).all())
        self.assertTrue(np.isfinite(table["RMSE"]).all())

    def test_feature_importance_is_available_for_real_bundles(self) -> None:
        for label, artifact in self.artifacts.items():
            table = analytics.feature_importance(artifact)
            self.assertEqual(len(table), len(self.features), label)
            self.assertAlmostEqual(float(table["importance"].sum()), 1.0, places=6)

    def test_feature_importance_is_empty_for_knn(self) -> None:
        fake = {
            "model": KNeighborsRegressor(n_neighbors=2),
            "feature_columns": self.features,
        }
        self.assertTrue(analytics.feature_importance(fake).empty)

    def test_missing_artifact_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            analytics.load_artifact(config.MODELS_DIR / "nope.joblib", self.features)

    def test_feature_mismatch_is_refused(self) -> None:
        path = config.MODELS_DIR / self.manifest["models"][0]["file"]
        with self.assertRaises(ValueError) as ctx:
            analytics.load_artifact(path, list(reversed(self.features)))
        self.assertIn("feature", str(ctx.exception).lower())

    def test_missing_bundle_key_is_named(self) -> None:
        bundle = joblib.load(config.MODELS_DIR / self.manifest["models"][0]["file"])
        broken = {key: value for key, value in bundle.items() if key != "metrics"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.joblib"
            joblib.dump(broken, path)
            with self.assertRaises(ValueError) as ctx:
                analytics.load_artifact(path, self.features)
        self.assertIn("metrics", str(ctx.exception))


class SmallHelperTests(unittest.TestCase):
    def test_output_band_boundaries(self) -> None:
        self.assertEqual(analytics.output_band(4_999.9), "Low")
        self.assertEqual(analytics.output_band(5_000.0), "Moderate")
        self.assertEqual(analytics.output_band(14_999.9), "Moderate")
        self.assertEqual(analytics.output_band(15_000.0), "High")

    def test_out_of_range_counts_names_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "IRRADIATION": [0.05, 2.0, np.nan],
                "AMBIENT_TEMPERATURE": [30.0, 30.0, 30.0],
            }
        )
        counts = analytics.out_of_range_counts(frame, schema.INPUT_BOUNDS)
        self.assertEqual(counts, {"IRRADIATION": 2})

    def test_extrapolation_flags(self) -> None:
        row = pd.Series({"IRRADIATION": 0.5, "AMBIENT_TEMPERATURE": 99.0})
        self.assertEqual(
            analytics.extrapolation_flags(row, schema.INPUT_BOUNDS),
            ["AMBIENT_TEMPERATURE"],
        )


if __name__ == "__main__":
    unittest.main()
