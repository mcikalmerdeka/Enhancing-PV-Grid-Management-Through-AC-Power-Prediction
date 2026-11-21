"""Tests for dataset loading, presets, and batch preparation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ac_power import data, schema


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = data.load_dataset()
        cls.features = ["IRRADIATION", "AMBIENT_TEMPERATURE", "HOUR", "PLANT_ID"]

    def test_shape_and_columns(self) -> None:
        expected = [
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
        self.assertEqual(list(self.df.columns), expected)
        self.assertEqual(self.df.shape, (3039, 9))

    def test_plant_counts_and_labels(self) -> None:
        self.assertEqual(
            self.df.PLANT_ID.value_counts().sort_index().to_dict(),
            {1: 1513, 2: 1526},
        )
        self.assertEqual(set(self.df.PLANT_NAME), {"Plant 1", "Plant 2"})

    def test_no_nulls(self) -> None:
        self.assertEqual(int(self.df.isna().sum().sum()), 0)

    def test_timestamp_range(self) -> None:
        self.assertEqual(self.df.DATE_TIME.min(), pd.Timestamp("2020-05-15 06:45"))
        self.assertEqual(self.df.DATE_TIME.max(), pd.Timestamp("2020-06-17 17:45"))

    def test_daylight_filter(self) -> None:
        self.assertTrue((self.df.IRRADIATION > schema.DAYLIGHT_THRESHOLD).all())
        self.assertEqual(self.df.DATE_TIME.dt.hour.nunique(), 13)

    def test_input_bounds_cover_the_dataset(self) -> None:
        for feature, (low, high, _step) in schema.INPUT_BOUNDS.items():
            self.assertGreaterEqual(float(self.df[feature].min()), low, feature)
            self.assertLessEqual(float(self.df[feature].max()), high, feature)


class PresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = data.load_dataset()
        cls.presets = data.plant_presets(cls.df)
        cls.features = ["IRRADIATION", "AMBIENT_TEMPERATURE", "HOUR", "PLANT_ID"]

    def test_presets_cover_every_plant_and_hour(self) -> None:
        self.assertEqual(len(self.presets), 26)
        self.assertEqual(
            set(map(tuple, self.presets[["PLANT_ID", "HOUR"]].to_numpy())),
            {(plant, hour) for plant in (1, 2) for hour in range(6, 19)},
        )

    def test_preset_for_returns_the_manifest_features_in_order(self) -> None:
        resolved = data.preset_for(self.presets, self.df, 2, 12, self.features)
        self.assertEqual(list(resolved), self.features)
        self.assertEqual(resolved["PLANT_ID"], 2.0)
        self.assertEqual(resolved["HOUR"], 12.0)
        self.assertTrue(np.isfinite(list(resolved.values())).all())

    def test_preset_for_falls_back_for_an_unobserved_hour(self) -> None:
        resolved = data.preset_for(self.presets, self.df, 1, 3, self.features)
        self.assertEqual(resolved["HOUR"], 3.0)
        self.assertTrue(np.isfinite(list(resolved.values())).all())

    def test_to_model_matrix_selects_and_coerces(self) -> None:
        matrix = data.to_model_matrix(self.df.head(3), self.features)
        self.assertEqual(list(matrix.columns), self.features)
        self.assertTrue((matrix.dtypes == "float64").all())
        self.assertEqual(matrix.shape, (3, 4))


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = data.load_dataset()
        cls.features = ["IRRADIATION", "AMBIENT_TEMPERATURE", "HOUR", "PLANT_ID"]

    def _raw(self) -> pd.DataFrame:
        raw = self.df.head(3)[
            ["DATE_TIME", "PLANT_ID", "IRRADIATION", "AMBIENT_TEMPERATURE", "HOUR"]
        ].copy()
        raw.insert(0, "Unnamed: 0", range(3))
        raw["AC_POWER"] = self.df.head(3)["AC_POWER"].to_numpy()
        return raw

    def test_drops_index_and_target_and_keeps_feature_order(self) -> None:
        matrix = data.prepare_batch(self._raw(), self.features)
        self.assertEqual(list(matrix.columns), self.features)
        self.assertEqual(matrix.shape, (3, 4))

    def test_missing_column_is_named(self) -> None:
        raw = self._raw().drop(columns=["HOUR"])
        with self.assertRaises(ValueError) as ctx:
            data.prepare_batch(raw, self.features)
        self.assertIn("HOUR", str(ctx.exception))

    def test_text_plant_id_is_rejected(self) -> None:
        raw = self._raw()
        raw["PLANT_ID"] = "Plant 1"
        with self.assertRaises(ValueError) as ctx:
            data.prepare_batch(raw, self.features)
        self.assertIn("PLANT_ID", str(ctx.exception))

    def test_source_plant_id_is_rejected(self) -> None:
        raw = self._raw()
        raw["PLANT_ID"] = 4135001
        with self.assertRaises(ValueError) as ctx:
            data.prepare_batch(raw, self.features)
        self.assertIn("PLANT_ID", str(ctx.exception))

    def test_nan_feature_is_rejected(self) -> None:
        raw = self._raw()
        raw.loc[1, "IRRADIATION"] = np.nan
        with self.assertRaises(ValueError) as ctx:
            data.prepare_batch(raw, self.features)
        self.assertIn("IRRADIATION", str(ctx.exception))

    def test_infinite_feature_is_rejected(self) -> None:
        raw = self._raw()
        raw.loc[1, "IRRADIATION"] = np.inf
        with self.assertRaises(ValueError) as ctx:
            data.prepare_batch(raw, self.features)
        self.assertIn("IRRADIATION", str(ctx.exception))

    def test_empty_frame_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            data.prepare_batch(pd.DataFrame(columns=self.features), self.features)
        self.assertIn("no rows", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
