"""Tests for the Altair builders."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from ac_power import ui


class ChartTests(unittest.TestCase):
    def test_output_by_plant(self) -> None:
        frame = pd.DataFrame(
            {"PLANT_NAME": ["Plant 1", "Plant 2"], "AC_POWER": [14_000.0, 10_500.0]}
        )
        chart = ui.output_by_plant_chart(frame)
        self.assertIn("mark", chart.to_dict())
        self.assertEqual(len(chart.data), 2)

    def test_output_by_hour(self) -> None:
        frame = pd.DataFrame({"HOUR": [6, 9, 12], "AC_POWER": [1_000.0, 8_000.0, 20_000.0]})
        chart = ui.output_by_hour_chart(frame)
        self.assertIn("mark", chart.to_dict())
        self.assertEqual(len(chart.data), 3)

    def test_daily_output(self) -> None:
        frame = pd.DataFrame(
            {
                "DATE": pd.to_datetime(["2020-05-15", "2020-05-16"]),
                "AC_POWER": [12_000.0, 13_000.0],
            }
        )
        chart = ui.daily_output_chart(frame)
        self.assertIn("mark", chart.to_dict())
        self.assertEqual(len(chart.data), 2)

    def test_irradiation_scatter(self) -> None:
        frame = pd.DataFrame(
            {
                "IRRADIATION": [0.1, 0.5, 0.9],
                "AC_POWER": [1_000.0, 10_000.0, 20_000.0],
                "PLANT_NAME": ["Plant 1", "Plant 1", "Plant 2"],
            }
        )
        chart = ui.irradiation_scatter_chart(frame)
        self.assertIn("mark", chart.to_dict())
        self.assertEqual(len(chart.data), 3)

    def test_importance(self) -> None:
        frame = pd.DataFrame(
            {"feature": ["IRRADIATION", "HOUR"], "importance": [0.7, 0.3]}
        )
        chart = ui.importance_chart(frame)
        self.assertIn("mark", chart.to_dict())
        self.assertEqual(len(chart.data), 2)

    def test_contribution_is_layered(self) -> None:
        frame = pd.DataFrame(
            {
                "display": ["Irradiation", "Hour of day"],
                "input": [0.8, 12.0],
                "baseline": [0.4, 11.0],
                "delta": [5_000.0, -200.0],
            }
        )
        spec = ui.contribution_chart(frame).to_dict()
        self.assertIn("layer", spec)
        self.assertGreaterEqual(len(spec["layer"]), 3)

    def test_gauge_clamps_value_and_label(self) -> None:
        low = json.dumps(ui.gauge_chart(-100).to_dict())
        high = json.dumps(ui.gauge_chart(99_999).to_dict())
        self.assertIn("0.0 kW", low)
        self.assertIn("30,000.0 kW", high)

    def test_ui_has_no_streamlit_import(self) -> None:
        source = Path(ui.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import streamlit", source)
        self.assertNotIn("from streamlit", source)


if __name__ == "__main__":
    unittest.main()
