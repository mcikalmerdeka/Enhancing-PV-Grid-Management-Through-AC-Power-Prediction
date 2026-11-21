"""Filesystem paths used by AC Power Studio.

Keeping every path here avoids path arithmetic in the views and keeps the app
working regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MANIFEST_PATH = MODELS_DIR / "manifest.json"
BATCH_EXAMPLE_PATH = DATA_DIR / "batch_example_ac_power.csv"

PLANT_IDS = (1, 2)
GENERATION_PATHS = {
    plant_id: DATA_DIR / f"Plant_{plant_id}_Generation_Data.csv"
    for plant_id in PLANT_IDS
}
SENSOR_PATHS = {
    plant_id: DATA_DIR / f"Plant_{plant_id}_Weather_Sensor_Data.csv"
    for plant_id in PLANT_IDS
}
