# Enhancing PV Grid Management Through AC-Power Prediction

![Project header](assets/Project%20Header.jpg)

**Plant-level AC power prediction for two Indian solar farms — a six-stage notebook, a reproducible model contract, and a Streamlit decision-support studio.**

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/dependencies-uv-DE5FE9)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![Tests](https://img.shields.io/badge/tests-59-2ea44f)

## Overview

Solar output is intermittent, and grid operators schedule generation in blocks. Knowing how much a plant will
deliver in the next interval is the difference between reserving the right amount of backup and guessing.

This project estimates **plant-level AC power** from conditions available before dispatch — irradiation, ambient
temperature, hour of day, and plant identity — for two adjacent PV plants in India. Each plant has 22 inverters
reporting every 15 minutes for 34 days. The analysis is deliberately conservative: a chronological holdout instead
of a random split, model selection on cross-validation only, and a persisted artifact contract that the app consumes
rather than re-implements.

<p align="center">
  <img src="assets/Standalone-Solar-Electric-Systems.jpg" alt="Solar Energy Generating System" width="620">
</p>

## Headline Results

Nine regressors were screened on a 7-day chronological holdout. Two were tuned and shipped:

| Shipped model | CV R² | Holdout R² | Holdout RMSE |
| --- | ---: | ---: | ---: |
| **GradientBoosting (tuned)** | 0.901 | 0.896 | 1,948 kW |
| **RandomForest (tuned)** | 0.902 | 0.896 | 1,952 kW |

For context, the plants peak at 29.2 MW and 26.0 MW. XGBoost ranked fourth on CV RMSE (R² 0.876) and was **not**
shipped; the linear family finished around R² 0.77, which is the clearest evidence that the irradiation-to-power
relationship is not linear once plant capacity and time of day are in play.

`models/manifest.json` is authoritative: the app reads the bundle list, feature order, and holdout start from it, so
re-running the notebook can change the winning models without touching application code.

## Engineering Highlights

- **Chronological evaluation.** The final seven days are held out (2,322 train / 717 test daylight rows). A random
  row split would place the same day and near-duplicate 15-minute readings on both sides and inflate the score.
- **Selection without leakage.** Models are ranked by cross-validated RMSE inside the training window; the holdout
  is reported once and never used to choose. Tuning must beat the untuned baseline under the same 5-fold protocol,
  otherwise the untuned pipeline ships.
- **Forecastable inputs only.** `MODULE_TEMPERATURE` is a panel-mounted sensor reading, unavailable before dispatch.
  Excluding it costs 0.0065 CV R² — measured and reported instead of hidden.
- **Plant-aware pooling.** The two plants differ in capacity by about 12%, so the pooled model carries `PLANT_ID`
  as a feature rather than averaging two different plants into one curve.
- **Self-describing artifacts.** Each joblib bundle carries the model, exact feature order, stored metrics, label,
  and target. The app refuses a bundle whose feature list does not match the manifest.
- **Tested beyond the happy path.** 59 automated checks cover data contracts, batch validation, artifact refusal,
  chart builders, the Streamlit boot path, the prediction form, and the batch example.

## Dataset

Source: [Kaggle — Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
(two Indian plants, 15-minute generation and weather readings, 2020-05-15 to 2020-06-17).

| | Plant 1 | Plant 2 |
| --- | ---: | ---: |
| Inverters | 22 | 22 |
| 15-minute timestamps | 3,157 | 3,259 |
| Peak AC power | 29.2 MW | 26.0 MW |
| Daylight rows (`IRRADIATION > 0.05`) | 1,513 | 1,526 |

Generation files are per-inverter; weather files are per-plant. The notebook sums AC power across inverters and
merges weather one-to-one on timestamp. After the daylight filter, **3,039 rows** remain.

Data-quality findings that shaped the analysis:

- **Plant 1's `DC_POWER` is unusable** — it reports roughly ten times its own AC power on every inverter row
  (median DC/AC 10.2, versus 1.02 at Plant 2). It is excluded from every model, and no DC-side claim is made.
- **Nine Plant 2 daylight rows report zero output under strong irradiation** (0.38–1.08 kW/m²). These are outages,
  not weather effects; they are kept, counted, and treated as irreducible error.
- **The two plants do not share a timestamp format** — Plant 1 is day-first, Plant 2 is ISO. Both are parsed with
  an explicit format chosen from the first value, so month and day can never be swapped.

## Approach

The notebook is organised into six stages, each with its own decisions recorded in markdown:

1. **Exploratory analysis** — load and merge, aggregate inverter data, inspect distributions and correlations, and
   document the data-quality findings above.
2. **Preprocessing** — restrict to daylight, split chronologically at 2020-06-10, inspect IQR flags without clipping
   physical peaks, and demonstrate min-max scaling.
3. **Feature selection** — rank correlation, VIF, and a measured comparison of the forecastable set with and without
   module temperature. Final inputs: `IRRADIATION`, `AMBIENT_TEMPERATURE`, `HOUR`, `PLANT_ID`.
4. **Screening** — nine regressors through `utils.eval_regression`, ranked by 3-fold CV RMSE, with the holdout
   reported for every candidate.
5. **Tuning** — randomized search over the top two only, 30 candidates each, 5-fold × 3-repeat, with a guard that
   refuses a tuned model worse than its baseline.
6. **Persistence** — two bundles plus a manifest, both created under `models/`.

## The Studio

`AC Power Studio` is a Streamlit app over the shipped artifacts:

- **Predict** — choose a plant and hour, get that hour's median conditions prefilled, adjust the weather, and read
  the prediction with a bullet gauge and an additive "what moved this prediction" breakdown.
- **Batch** — score an uploaded CSV (or the committed 12-row example), with validation for missing columns, invalid
  plant ids, NaN/inf values, and out-of-range extrapolation warnings; results download as CSV.
- **Insights** — CV versus holdout metrics, per-plant held-out error, daily and hourly output profiles, feature
  importance, irradiation scatter, and rank correlation.
- **About** — model card, data dictionary, limitations, and source links.

## Tech Stack

| Layer | Tools |
| --- | --- |
| Language | Python 3.14 |
| Packaging | **uv**, `pyproject.toml`, `uv.lock`, `.python-version` |
| Analysis | pandas, scikit-learn, XGBoost, `utils/` (vendored ML toolkit) |
| App | Streamlit, Altair |
| Persistence | joblib bundles, JSON manifest |
| Testing | `unittest`, Streamlit `AppTest` |

## Quickstart

The project is managed with [uv](https://docs.astral.sh/uv/). `uv sync` creates `.venv` from the lockfile, and
`uv run` executes inside it — no manual activation.

```powershell
# 1. Install uv (Windows)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the locked environment
uv sync

# 3. Regenerate the model artifacts
uv run python tests/verify_notebook.py
#    or open Prediction of AC-Power.ipynb and run all cells

# 4. Launch the studio
uv run streamlit run main.py

# 5. Run the test suite
uv run python -m unittest discover -s tests -v
```

The notebook writes `models/gradientboosting_tuned.joblib`, `models/randomforest_tuned.joblib`, and
`models/manifest.json`. The app will not start until those exist; it also refuses a bundle whose feature order does
not match the manifest.

If you prefer a Jupyter kernel, point your editor at the `.venv` interpreter created by `uv sync`.

## Project Structure

```
.
├── Prediction of AC-Power.ipynb   six-stage analysis; writes models/
├── main.py                        Streamlit entry point
├── ac_power/                      app package
│   ├── config.py                  paths only
│   ├── schema.py                  feature metadata, bounds, data dictionary
│   ├── data.py                    loading, presets, batch validation
│   ├── analytics.py               artifact loading, prediction, attribution, metrics
│   ├── ui.py                      Altair builders (no Streamlit import)
│   └── views/                     predict, batch, insights, about
├── utils/                         vendored ML toolkit
├── data/                          generation, weather, and the batch example
├── models/                        generated bundles and manifest.json
├── tests/                         unittest suite and notebook verifier
├── docs/utils-bug-report.md       defects found in the vendored toolkit
├── pyproject.toml
├── uv.lock
└── .python-version
```

## Testing

```powershell
uv run python -m unittest discover -s tests -v
```

Covers dataset contracts (3,039 daylight rows, per-plant counts), preset resolution, batch validation against
missing columns and invalid plant ids, artifact and feature-order refusal, chart construction, app boot with four
tabs, the prediction form, and the batch example path.

`tests/verify_notebook.py` executes all 34 notebook code cells in one process, so the notebook can be verified
without `nbformat`/`nbclient`. A full run takes about 7–8 minutes; tuning dominates.

## Limitations

- **Thirty-four days, one season.** The holdout is a single weather week; a different week would move the number.
- **Irradiation dominance.** Within daylight, irradiation correlates 0.99 with Plant 1's output and 0.75 with
  Plant 2's. The model's contribution is the shape around that relationship, not the order of magnitude.
- **Nine unpredictable outages** sit in the residuals by construction.
- **Plant 1's DC channel is unusable**, so no efficiency or DC-side analysis is attempted.
- **`PLANT_ID` is two-level.** The model interpolates between these two plants; it says nothing about a third.
- **Daylight only.** Night output is defined as zero, not predicted.
- **One chronological window.** A rolling-origin evaluation would give a distribution instead of one number.

## Future Work

- Integrate a real irradiance forecast and propagate its error, rather than treating irradiation as known.
- Evaluate with rolling-origin splits across several weeks.
- Add quantile predictions so dispatch gets a range for reserve scheduling.
- Extend the analysis to soiling and maintenance detection — the second half of the original brief, not addressed
  in this version.
- Validate transfer to a third plant instead of interpolating a two-level identifier.

## References

- Kaggle dataset: [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
- Toolkit audit: [`docs/utils-bug-report.md`](docs/utils-bug-report.md)

## Author

**Muhammad Cikal Merdeka** — Data Analyst / Data Scientist

[GitHub](https://github.com/mcikalmerdeka) · [LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)
