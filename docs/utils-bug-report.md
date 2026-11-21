# `utils/` Toolkit — Defect Report

**Reported:** 2026-11-21
**Source:** vendored copy in this repository (added in commit `2e2b824`), maintained in a separate repository
**Verified against:** Python 3.14.3, scikit-learn 1.9.1, pandas 3.1.1, xgboost 3.4.1
**How these were found:** rewriting `Prediction of AC-Power.ipynb` to use `utils` end to end, including screening
nine regressors and tuning the top two.

## Summary

| ID | Severity | Location | Symptom | Status in this repo |
| --- | --- | --- | --- | --- |
| U-01 | **High** | `regression_evals_and_tuning.py` lines 648, 656, 706, 714 | Tuning `decisiontree` or `randomforest` crashes on `criterion='friedman_mse'`, removed in scikit-learn 1.9 | Filtered RF grid in the notebook; DT path untested |
| U-02 | **High** | `regression_evals_and_tuning.py` line 153 | `eval_regression` requires a pre-fitted estimator but neither says so nor fits it; raises `NotFittedError` | Notebook fits every candidate first |
| U-03 | Medium | `regression_evals_and_tuning.py` lines 590, 737 | `get_model_pipeline` / `get_hyperparameters` reject natural names such as `GradientBoosting`; only `gb` works | Explicit `MODEL_REGISTRY_KEYS` map in the notebook |
| U-04 | Medium | `feature_selection.py` line 289 | `calculate_vif(df, target_col=...)` raises `KeyError` when the target is not already a column, although it only drops it | Pass the target in the frame |
| U-05 | Medium | `regression_evals_and_tuning.py` lines 205-236 | Raw metric shape is `metrics[metric]["test"]`, while CV is `metrics["cv"][metric]`; consumers must hand-normalise | `build_bundle` normalises to `{"cv", "test"}` |
| U-06 | Low | `regression_evals_and_tuning.py` lines 125-147 | All frames are converted to numpy, losing feature names and emitting sklearn warnings on every `predict` | Notebook suppresses warnings |
| U-07 | Low | `regression_evals_and_tuning.py` line 188 | `cross_validate(..., n_jobs=-1)` is hardcoded; no way to disable process parallelism on small data | Accepted; 3-fold linear fits cost ~5 s each |

Design hazards that are not defects are listed separately at the end.

---

## U-01 — Invalid `friedman_mse` in tree tuning grids

**Severity:** High — the documented tuning path fails for two of the nine supported models.

`get_hyperparameters` offers `friedman_mse` for both `decisiontree` and `randomforest`, in the exhaustive grid and
in the random distributions:

```python
# utils/regression_evals_and_tuning.py:648 and :706
'dt__criterion': ['squared_error', 'friedman_mse', 'absolute_error'],

# utils/regression_evals_and_tuning.py:656
'rf__criterion': ['squared_error', 'friedman_mse'],

# utils/regression_evals_and_tuning.py:714
'rf__criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
```

In scikit-learn 1.9.1 the accepted regression criteria are:

| Estimator | Accepted `criterion` |
| --- | --- |
| `DecisionTreeRegressor` | `squared_error`, `absolute_error`, `poisson` |
| `RandomForestRegressor` | `squared_error`, `absolute_error`, `poisson` |
| `GradientBoostingRegressor` | `squared_error` only |

**Reproduction**

```python
from utils import get_hyperparameters, get_model_pipeline, tune_pipelines

tune_pipelines(
    {"RandomForest": get_model_pipeline("randomforest")},
    {"RandomForest": get_hyperparameters("randomforest", "random")},
    X_train, y_train, X_test, y_test,
    search_method="random", n_iter=30, n_jobs=-1,
)
```

Observed failure:

```
sklearn.utils._param_validation.InvalidParameterError:
The 'criterion' parameter of RandomForestRegressor must be a str among
{'squared_error', 'poisson', 'absolute_error'}. Got 'friedman_mse' instead.
```

A random search fails **intermittently**, only when the sampler draws `friedman_mse`; a grid search fails always.
The same defect exists for `DecisionTreeRegressor`, which was not tuned in this project but is confirmed against
the same parameter constraints.

**Suggested fix**

```python
'dt__criterion': ['squared_error', 'absolute_error'],   # drop friedman_mse
'rf__criterion': ['squared_error', 'absolute_error'],   # drop friedman_mse
```

`poisson` is also valid but assumes a strictly positive target; it should not be offered by a generic regressor
grid without that caveat.

**Workaround in this repo:** the notebook replaces the RandomForest criterion list with
`["squared_error", "absolute_error"]` and documents why. `utils` is untouched.

---

## U-02 — `eval_regression` does not fit the model

**Severity:** High — the most natural reading of the function name and docstring is wrong, and the failure is a
bare `NotFittedError` from deep inside `predict`.

`eval_regression` predicts on the test set before it does anything else:

```python
# utils/regression_evals_and_tuning.py:153
y_pred_test = model.predict(X_test)
```

Nothing in the signature or docstring says the estimator must already be fitted. Passing a fresh pipeline is the
obvious thing to do:

```python
from utils import eval_regression, get_model_pipeline

eval_regression(
    model=get_model_pipeline("randomforest"),
    X_test=X_test, y_test=y_test, X_train=X_train, y_train=y_train,
)
```

Observed:

```
sklearn.exceptions.NotFittedError: This Pipeline instance is not fitted yet.
Call 'fit' with appropriate arguments before using this estimator.
```

The reference notebook works because it fits every candidate on the line before calling `eval_regression`.

**Suggested fix (choose one):**

1. **Document and guard** — add “`model` must already be fitted” to the docstring and raise a clear error:
   ```python
   from sklearn.utils.validation import check_is_fitted
   try:
       check_is_fitted(model)
   except NotFittedError as exc:
       raise ValueError(
           "eval_regression expects an already-fitted model; call model.fit(X_train, y_train) first."
       ) from exc
   ```
2. **Fit inside** — call `model.fit(X_train, y_train)` before predicting. Convenient, but silently overwrites a
   fitted estimator and doubles the work when the caller already fitted it.

Option 1 preserves the current contract and removes the confusing failure.

**Workaround in this repo:** every candidate is explicitly fitted before evaluation; the requirement is stated in
the notebook.

---

## U-03 — Registry keys are not normalised to display names

**Severity:** Medium — ergonomics; the error message is clear enough, but natural names fail.

`get_model_pipeline` and `get_hyperparameters` lower-case the input and look it up directly:

```python
# utils/regression_evals_and_tuning.py:590 and :737
model_key = model_name.lower()
```

The registry uses abbreviations that do not follow from the class names:

```python
SUPPORTED_MODELS = [
    'linearregression', 'ridge', 'lasso', 'elasticnet', 'knn',
    'decisiontree', 'randomforest', 'gb', 'xgboost',
]
```

So `"GradientBoosting"` → `"gradientboosting"` fails, while `"RandomForest"` → `"randomforest"` succeeds. Mixed
naming is a trap for callers iterating over display labels.

**Suggested fix:** add an alias normaliser.

```python
_ALIASES = {
    "gradientboosting": "gb",
    "gradient_boosting": "gb",
    "gradient boosting": "gb",
    "linearregression": "linearregression",
    ...
}

def _normalise(name: str) -> str:
    key = re.sub(r"[\s_-]+", "", name.strip().lower())
    return _ALIASES.get(key, key)
```

Use it in both `get_model_pipeline` and `get_hyperparameters` so they can never disagree.

**Workaround in this repo:** an explicit `MODEL_REGISTRY_KEYS` dict in the notebook maps display names to keys.

---

## U-04 — `calculate_vif` requires the target column it is about to drop

**Severity:** Medium — an avoidable `KeyError` in the obvious usage.

```python
# utils/feature_selection.py:289
features_df = df.drop(columns=[target_col])
```

The target is only needed so it can be removed, yet passing a feature-only frame fails:

```python
from utils import calculate_vif

calculate_vif(X_train, target_col="AC_POWER")
```

```
KeyError: "['AC_POWER'] not found in axis"
```

**Suggested fix**

```python
features_df = df.drop(columns=[target_col], errors="ignore")
```

This keeps the current behaviour when the target is present and makes the common feature-only call work.

**Workaround in this repo:** the notebook passes `train_df[FEATURE_CANDIDATES + [TARGET]]`.

---

## U-05 — Raw metric shape is inconsistent between CV and single-run results

**Severity:** Medium — the shape is easy to misread, and getting it wrong fails late (or produces `NaN` in an app).

`eval_regression` returns:

```python
metrics["cv"]["r2"]["test_mean"]   # CV: metric -> stat
metrics["r2"]["test"]              # single run: metric -> {"train", "test"}
```

There is no `metrics["test"]["r2"]`, even though a consumer naturally reaches for it. The reference notebook hides
this by normalising inside `build_bundle`:

```python
"metrics": {
    "cv": metrics["cv"],
    "test": {key: value["test"] for key, value in metrics.items() if key != "cv"},
}
```

A consumer that skips the normalisation silently reads `NaN`; the reference app's `_metric` helper falls back to
`NaN` on `KeyError`, which means a missing test metric looks like “not available” rather than a bug.

**Suggested fix (choose one):**

1. Add a public helper:
   ```python
   def bundle_metrics(result: dict) -> dict:
       return {
           "cv": result["cv"],
           "test": {k: v["test"] for k, v in result.items() if k != "cv"},
       }
   ```
2. Document the two shapes prominently in the `eval_regression` docstring and return type.

**Workaround in this repo:** the notebook's `build_bundle` performs the same normalisation as the reference.

---

## U-06 — Frames are converted to numpy, losing feature names

**Severity:** Low — correctness is preserved when column order is right, but every prediction emits a warning and
the column-order guard disappears.

```python
# utils/regression_evals_and_tuning.py:125-147
def to_numpy(arr):
    if isinstance(arr, (pd.DataFrame, pd.Series)):
        return arr.to_numpy()
    return arr
```

A model fitted on a DataFrame and predicted on a numpy array raises:

```
UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names
```

**Suggested fix:** keep pandas objects as-is. Every supported estimator accepts DataFrames, and
`train_test_split`/`cross_validate` already preserve them. If numpy is required for a specific metric, convert only
at the metric boundary.

**Workaround in this repo:** warnings are suppressed in the notebook, and every candidate is fitted and scored on
the same column order.

---

## U-07 — `n_jobs=-1` is hardcoded in `eval_regression`

**Severity:** Low — performance control, not correctness.

```python
# utils/regression_evals_and_tuning.py:188
cv = cross_validate(..., n_jobs=-1)
```

On 2,322 training rows, process startup dominates: a 3-fold `LinearRegression` evaluation took about 5 seconds,
while `XGBoost` took under 1 second. There is no way to request serial execution for small data or constrained
environments.

**Suggested fix:** add `n_jobs: int = -1` to the signature and pass it through.

**Workaround in this repo:** none; the cost is accepted.

---

## Design hazards (not defects)

These are behaviours worth documenting or redesigning, but they did not fail here.

### H-01 — Pipelines are unscaled; persistence is a silent trap

`get_model_pipeline` returns bare estimators (`Pipeline([('rf', ...)])`), while scaling is a separate
`feature_scaling` step. Trees are invariant to scaling, so the reference ships trees and the issue never surfaces.
A scale-sensitive model fitted on scaled data and persisted **without its scaler** would accept raw values at
inference and produce wrong predictions without raising.

**Recommendation:** either document the requirement in `get_model_pipeline`, or accept a `scaler` argument that
returns a single self-contained pipeline. This project embeds a `MinMaxScaler` step in every scale-sensitive
candidate so the bundle can always accept raw values.

### H-02 — `SUPPORTED_MODELS` does not match the models a normal screening would include

`AdaBoost`, `MLP`, and `SVR` are common regression baselines but have no registry entry. The reference notebook
benchmarks eleven models by importing estimators from scikit-learn directly, so the toolkit and the documented
workflow disagree about what “supported” means.

**Recommendation:** add them or state explicitly that the registry is a curated subset.

### H-03 — `tune_pipelines` returns fit times as a positional list

`tune_pipelines` returns `(fitted_models, fit_times)` where `fit_times` is a list. `tune_single_model` assumes a
single entry and returns `fit_times[0]`. Any filtering or reordering between the two call sites silently misaligns
model and timing.

**Recommendation:** return `dict[str, float]` keyed by model name.

### H-04 — Cross-validation is always random `RepeatedKFold`

`eval_regression` exposes `n_splits` and `n_repeats` but no time-aware CV. On timestamped data, random folds place
the same day on both sides of a split and inflate the score. This project therefore uses a chronological holdout
outside the helper and treats the CV columns as a ranking device only.

**Recommendation:** document the assumption, or accept a `cv` splitter argument.

---

## Reproduction notes

All findings were reproduced against the vendored copy in this repository:

```
utils/regression_evals_and_tuning.py
utils/feature_selection.py
```

Nothing in `utils/` was modified by this project. The notebook-side workarounds are documented in
`Prediction of AC-Power.ipynb` (Stages 3-5) and in
`docs/superpowers/plans/2026-11-21-ac-power-notebook-and-studio.md`.
