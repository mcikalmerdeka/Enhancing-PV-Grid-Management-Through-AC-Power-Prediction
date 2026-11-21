"""Tests for paths and the single source of truth for feature metadata."""

from __future__ import annotations

import unittest

from ac_power import config, schema


class ConfigTests(unittest.TestCase):
    def test_generation_paths_exist(self) -> None:
        self.assertEqual(set(config.GENERATION_PATHS), {1, 2})
        for plant_id, path in config.GENERATION_PATHS.items():
            self.assertTrue(path.exists(), f"missing generation file for plant {plant_id}: {path}")
            self.assertIn(f"Plant_{plant_id}", path.name)

    def test_sensor_paths_exist(self) -> None:
        self.assertEqual(set(config.SENSOR_PATHS), {1, 2})
        for plant_id, path in config.SENSOR_PATHS.items():
            self.assertTrue(path.exists(), f"missing sensor file for plant {plant_id}: {path}")
            self.assertIn(f"Plant_{plant_id}", path.name)

    def test_manifest_exists(self) -> None:
        self.assertTrue(config.MANIFEST_PATH.exists(), config.MANIFEST_PATH)
        self.assertEqual(config.MANIFEST_PATH.name, "manifest.json")

    def test_project_root_is_the_repository(self) -> None:
        self.assertTrue((config.PROJECT_ROOT / "README.md").exists())
        self.assertTrue((config.PROJECT_ROOT / "data").is_dir())
        self.assertTrue((config.PROJECT_ROOT / "models").is_dir())


class SchemaTests(unittest.TestCase):
    def test_target_and_threshold(self) -> None:
        self.assertEqual(schema.TARGET_COLUMN, "AC_POWER")
        self.assertEqual(schema.DAYLIGHT_THRESHOLD, 0.05)

    def test_plant_labels_and_source_ids(self) -> None:
        self.assertEqual(set(schema.PLANT_LABELS), {1, 2})
        self.assertEqual(set(schema.PLANT_SOURCE_IDS), {4135001, 4135002})
        self.assertEqual(set(schema.PLANT_SOURCE_IDS.values()), {1, 2})

    def test_every_known_feature_has_a_display_name(self) -> None:
        for feature in schema.KNOWN_FEATURES:
            self.assertIn(feature, schema.DISPLAY_NAMES, feature)

    def test_categorical_features_are_plant_only(self) -> None:
        self.assertEqual(set(schema.CATEGORICAL_FEATURES), {"PLANT_ID"})
        self.assertFalse(set(schema.CATEGORICAL_FEATURES) & set(schema.INPUT_BOUNDS))

    def test_numeric_features_have_sane_bounds(self) -> None:
        numeric = [f for f in schema.KNOWN_FEATURES if f not in schema.CATEGORICAL_FEATURES]
        for feature in numeric:
            low, high, step = schema.INPUT_BOUNDS[feature]
            self.assertLess(low, high, feature)
            self.assertGreater(step, 0, feature)

    def test_feature_label_falls_back_to_the_name(self) -> None:
        self.assertEqual(schema.feature_label("HOUR"), "Hour of day")
        self.assertEqual(schema.feature_label("NOT_A_FEATURE"), "NOT_A_FEATURE")

    def test_data_dictionary_covers_target_and_excluded_feature(self) -> None:
        definitions = schema.get_feature_definitions()
        for name in (schema.TARGET_COLUMN, "MODULE_TEMPERATURE", "IRRADIATION", "PLANT_ID"):
            self.assertIn(name, definitions, name)
            for key in ("description", "data_type", "specific_type"):
                self.assertTrue(definitions[name][key], f"{name}.{key}")


if __name__ == "__main__":
    unittest.main()
