"""Headless boot test for the Streamlit app.

AppTest renders the real script and every tab; the owner does the interactive
pass in a browser. This catches import errors, broken artifact contracts, and
tab wiring regressions before that.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from ac_power import analytics, config, data

APP = Path(__file__).resolve().parents[1] / "main.py"


def _widget_by_key(widgets, key: str):
    """Find a widget by its session key, which is stable across tabs."""
    for widget in widgets:
        if getattr(widget, "key", None) == key:
            return widget
    raise AssertionError(f"widget with key {key!r} not found")


class AppSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.at = AppTest.from_file(str(APP), default_timeout=120).run()

    def test_app_runs_without_exception(self) -> None:
        self.assertEqual(list(self.at.exception), [])

    def test_four_tabs(self) -> None:
        self.assertEqual(len(self.at.tabs), 4)

    def test_title(self) -> None:
        self.assertTrue(any("AC Power Studio" in title.value for title in self.at.title))

    def test_tab_labels(self) -> None:
        self.assertEqual(
            [tab.label for tab in self.at.tabs],
            ["Predict", "Batch", "Insights", "About"],
        )

    def test_predict_form_scores_without_exception(self) -> None:
        at = AppTest.from_file(str(APP), default_timeout=120).run()
        _widget_by_key(at.selectbox, "predict_plant").select(2).run()
        _widget_by_key(at.selectbox, "predict_hour").select(12).run()

        submit = next(button for button in at.button if button.label == "Predict AC power")
        submit.click().run()

        self.assertEqual(list(at.exception), [])
        labels = [metric.label for metric in at.metric]
        self.assertTrue(any("Predicted AC power" in label for label in labels), labels)

    def test_batch_example_is_scoreable(self) -> None:
        manifest = analytics.load_manifest()
        frame = pd.read_csv(config.BATCH_EXAMPLE_PATH)
        prepared = data.prepare_batch(frame, list(manifest["feature_columns"]))
        self.assertEqual(len(prepared), 12)
        self.assertEqual(
            list(prepared.columns), list(manifest["feature_columns"])
        )

    def test_batch_example_path_renders_results(self) -> None:
        at = AppTest.from_file(str(APP), default_timeout=120).run()
        source = _widget_by_key(at.segmented_control, "batch_source")
        source.select("Example data").run()

        self.assertEqual(list(at.exception), [])
        labels = [metric.label for metric in at.metric]
        self.assertIn("Rows scored", labels)
        self.assertTrue(
            any(len(frame.value) == 12 for frame in at.dataframe), "no 12-row result table"
        )

    def test_insights_render_takeaways(self) -> None:
        captions = " ".join(caption.value for caption in self.at.caption)
        self.assertIn("Takeaway", captions)
        self.assertIn("cross-validated R²", captions)

    def test_about_mentions_target_and_units(self) -> None:
        text = " ".join(markdown.value for markdown in self.at.markdown)
        self.assertIn("AC_POWER", text)
        self.assertIn("kW", text)


if __name__ == "__main__":
    unittest.main()
