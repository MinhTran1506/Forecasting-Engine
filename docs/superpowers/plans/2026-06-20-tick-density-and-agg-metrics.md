# Smart Tick Density + Always-On Aggregate Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thin the forecast chart's x-axis ticks so long series stay readable, and show headline MAPE/MAE/RMSE for the summed forecast in the no-filter / partial-filter view.

**Architecture:** Add a pure `Visualizer._thin_ticks` helper that picks a stride by period count and pins the last label; `plot_forecast` uses it instead of emitting every tick. In the aggregated branch of the Interactive Forecast Explorer, recompute MAPE/MAE/RMSE on the already-available summed test arrays (`test_data_agg` vs `test_predictions`) via the existing `ForecastMetrics` and render them as metric cards.

**Tech Stack:** Python 3.13, plotly, pandas, numpy, Streamlit; pytest for tests.

## Global Constraints

- Tick thinning is by stride, tiered on `n = len(x_labels)`: `n <= 24` → stride 1; `25 <= n <= 60` → stride 3; `n > 60` → stride 6.
- The last period (index `n-1`) is ALWAYS in the ticks, even when it does not fall on the stride; never duplicate it if the stride already lands on it.
- All data points are still plotted; only axis ticks are thinned. The `x_labels=None` path (integer "Period" axis) and `plot_model_comparison` / `plot_residuals` are unchanged.
- Aggregate headline metrics are recomputed on the summed test series (`test_data_agg`, `test_predictions`) — NOT a volume-weighted average of per-group metrics.
- If the aggregate retrain guard (`if test_result and len(aggregated_future_forecast) > 0:`) is false, the headline metrics are simply not shown; nothing else changes.
- The `all_filters_selected` branch (per-group metrics) is NOT modified.

---

### Task 1: `Visualizer._thin_ticks` helper

**Files:**
- Modify: `utils/visualizer.py` (add a static method to the `Visualizer` class, e.g. directly above `plot_forecast` which starts at line 11)
- Test: `tests/test_thin_ticks.py` (create)

**Interfaces:**
- Produces: `Visualizer._thin_ticks(x_labels) -> tuple[list[int], list[str]]`. Given a list of label strings, returns `(tickvals, ticktext)` where `tickvals` is a sorted list of integer indices into `x_labels` (stride-selected, with `len(x_labels)-1` always included and no duplicate), and `ticktext = [x_labels[i] for i in tickvals]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_thin_ticks.py`:

```python
from utils.visualizer import Visualizer


def _labels(n):
    return [f"L{i}" for i in range(n)]


def test_small_series_shows_all_ticks():
    # n <= 24 -> stride 1, every index present.
    vals, text = Visualizer._thin_ticks(_labels(12))
    assert vals == list(range(12))
    assert text == _labels(12)


def test_boundary_24_stride_1():
    vals, _ = Visualizer._thin_ticks(_labels(24))
    assert vals == list(range(24))


def test_25_uses_stride_3_no_duplicate_last():
    # 25 items, stride 3 -> 0,3,...,24; 24 is last, must not duplicate.
    vals, _ = Visualizer._thin_ticks(_labels(25))
    assert vals == [0, 3, 6, 9, 12, 15, 18, 21, 24]
    assert vals.count(24) == 1


def test_40_stride_3_pins_last():
    # 40 items, stride 3 -> 0,3,...,39? 39 % 3 == 0 so last lands on stride.
    vals, _ = Visualizer._thin_ticks(_labels(40))
    assert vals[0] == 0
    assert vals[-1] == 39          # last always present
    assert all((b - a) == 3 for a, b in zip(vals, vals[1:]))


def test_boundary_60_stride_3():
    vals, _ = Visualizer._thin_ticks(_labels(60))
    # stride 3 across 60 -> 0..57 by 3, last index 59 pinned (59 % 3 != 0).
    assert vals[0] == 0
    assert vals[-1] == 59
    assert 59 not in range(0, 60, 3)  # confirms it was pinned, not on stride


def test_61_uses_stride_6():
    vals, _ = Visualizer._thin_ticks(_labels(61))
    assert vals[1] == 6            # stride 6
    assert vals[-1] == 60          # last pinned


def test_72_stride_6_pins_last():
    vals, text = Visualizer._thin_ticks(_labels(72))
    assert vals[0] == 0
    assert vals[1] == 6
    assert vals[-1] == 71          # 71 % 6 != 0 -> pinned
    assert vals.count(71) == 1
    assert text == [f"L{i}" for i in vals]   # ticktext aligns with tickvals
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_thin_ticks.py -v`
Expected: FAIL with `AttributeError: type object 'Visualizer' has no attribute '_thin_ticks'`

- [ ] **Step 3: Implement the helper**

In `utils/visualizer.py`, add this static method inside the `Visualizer` class, immediately above `plot_forecast` (line 11). Match the existing 4-space indentation:

```python
    @staticmethod
    def _thin_ticks(x_labels):
        """
        Select a readable subset of x-axis tick positions for the given labels.

        Stride by total count n = len(x_labels):
          n <= 24 -> stride 1 (all), 25..60 -> stride 3, n > 60 -> stride 6.
        The last index (n-1) is always included; it is never duplicated if the
        stride already lands on it.

        Returns (tickvals, ticktext): integer indices into x_labels and the
        matching label strings.
        """
        n = len(x_labels)
        if n == 0:
            return [], []

        if n <= 24:
            stride = 1
        elif n <= 60:
            stride = 3
        else:
            stride = 6

        tickvals = list(range(0, n, stride))
        if tickvals[-1] != n - 1:
            tickvals.append(n - 1)

        ticktext = [x_labels[i] for i in tickvals]
        return tickvals, ticktext
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_thin_ticks.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add utils/visualizer.py tests/test_thin_ticks.py
git commit -m "feat: add _thin_ticks helper for readable x-axis tick density"
```

---

### Task 2: Use `_thin_ticks` in `plot_forecast`

**Files:**
- Modify: `utils/visualizer.py:110-116` (the `if x_labels:` tick block inside `plot_forecast`)
- Test: `tests/test_plot_forecast_labels.py` (existing — add one case)

**Interfaces:**
- Consumes: `Visualizer._thin_ticks(x_labels)` from Task 1.

- [ ] **Step 1: Add a failing test for thinned ticks on a long series**

Append to the existing `tests/test_plot_forecast_labels.py`:

```python
def test_long_series_thins_axis_ticks():
    # 50 actual + 10 forecast = 60 labels -> stride 3, last pinned.
    labels = [f"2020/{i}" for i in range(60)]
    actual = list(range(50))
    fitted = list(range(50))
    forecast = list(range(10))
    fig = Visualizer.plot_forecast(actual, fitted, forecast, x_labels=labels)
    tickvals = list(fig.layout.xaxis.tickvals)
    # Far fewer ticks than 60, and the last label is present.
    assert len(tickvals) < 60
    assert tickvals[0] == 0
    assert tickvals[-1] == 59
    assert list(fig.layout.xaxis.ticktext)[-1] == "2020/59"
```

Note: the existing `test_labels_set_time_axis_and_ticktext` uses 5 labels (n <= 24 → stride 1), so it still expects `tickvals == [0,1,2,3,4]` and continues to pass unchanged.

- [ ] **Step 2: Run the new test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_plot_forecast_labels.py::test_long_series_thins_axis_ticks -v`
Expected: FAIL — current code emits all 60 tickvals, so `len(tickvals) < 60` is false.

- [ ] **Step 3: Wire the helper into `plot_forecast`**

In `utils/visualizer.py`, replace the current block (lines ~110-116):

```python
        if x_labels:
            tickvals = list(range(len(x_labels)))
            fig.update_xaxes(
                tickmode='array',
                tickvals=tickvals,
                ticktext=list(x_labels)
            )
```

with:

```python
        if x_labels:
            tickvals, ticktext = Visualizer._thin_ticks(x_labels)
            fig.update_xaxes(
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext
            )
```

- [ ] **Step 4: Run the full visualizer test file**

Run: `.venv/Scripts/python.exe -m pytest tests/test_plot_forecast_labels.py -v`
Expected: PASS (3 passed — the 2 existing + the new one)

- [ ] **Step 5: Commit**

```bash
git add utils/visualizer.py tests/test_plot_forecast_labels.py
git commit -m "feat: thin forecast chart x-axis ticks via _thin_ticks"
```

---

### Task 3: Headline aggregate metrics in the summed view

**Files:**
- Modify: `pages/multi_model.py` — two spots in the aggregated/partial-filter branch: the metric-cards row (lines ~888-890) and inside the `if test_result and len(aggregated_future_forecast) > 0:` block (around lines ~950-972).

**Interfaces:**
- Consumes: `ForecastMetrics.calculate_all(actual, predicted) -> dict` (already imported at `pages/multi_model.py:15`; returns a dict with keys including `'MAPE'`, `'MAE'`, `'RMSE'`, each already rounded). Local arrays `test_data_agg` (summed test actuals) and `test_predictions` (`np.maximum(test_result['predictions'], 0)`) already exist in that block.

There is no unit test for this Streamlit UI wiring; verification is a syntax check plus the existing suite plus a manual check. This is one cohesive task: the metrics must be computed before they can be displayed.

- [ ] **Step 1: Compute aggregate metrics and widen the metric row**

In `pages/multi_model.py`, the metric-cards row currently reads (lines ~888-890):

```python
                    col1, col2 = st.columns(2)
                    col1.metric("Groups Summed", len(matching_groups))
                    col2.metric("Models Aggregated", len(model_info['models_used']))
```

Leave this row as-is. The new metric cards are added LATER, inside the retrain block where `test_data_agg` and `test_predictions` exist, because the metrics depend on those arrays. Locate the block (around lines ~950-972):

```python
                        if test_result and len(aggregated_future_forecast) > 0:
                            # Create fitted values: train portion = actual, test portion = predictions
                            test_predictions = np.maximum(test_result['predictions'], 0)
                            full_fitted = np.concatenate([
                                train_data_agg,      # Training portion = actual
                                test_predictions      # Test portion = model predictions
                            ])
```

Immediately AFTER that `full_fitted = np.concatenate([...])` statement (still inside the `if test_result ...` block, before the `x_labels = build_period_labels(...)` call), insert:

```python
                            # Headline accuracy for the summed view, recomputed on
                            # the aggregated test series (captures cross-group
                            # error cancellation; not a weighted average).
                            agg_metrics = ForecastMetrics.calculate_all(
                                test_data_agg, test_predictions
                            )
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Agg MAPE", f"{agg_metrics['MAPE']:.2f}%")
                            m2.metric("Agg MAE", f"{agg_metrics['MAE']:.2f}")
                            m3.metric("Agg RMSE", f"{agg_metrics['RMSE']:.2f}")
```

This places three accuracy cards just above the aggregated chart, shown only when the retrain succeeded (the enclosing guard). The existing "Groups Summed" / "Models Aggregated" row stays where it is.

- [ ] **Step 2: Syntax check**

Run: `.venv/Scripts/python.exe -c "import ast; ast.parse(open('pages/multi_model.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Confirm no regression in the test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: PASS (all existing tests + Tasks 1-2 tests; count = prior 13 + 7 thin-ticks + 1 plot = 21 passed)

- [ ] **Step 4: Manual check (if app launchable)**

Launch the app (requires `FUTURE1_USERS` env var; see CLAUDE.md), run a grouped forecast, then in the Interactive Forecast Explorer:
- Leave all filters on "All" → confirm "Agg MAPE / Agg MAE / Agg RMSE" cards appear above the summed chart with plausible values.
- Apply a partial filter (narrow one column, leave another "All") → confirm the cards update and the summed chart still renders.
- Narrow every column to a specific value → confirm the all-filters-selected per-group metrics still show as before (unchanged path).

If the app cannot be launched here, rely on Step 2 + Step 3.

- [ ] **Step 5: Commit**

```bash
git add pages/multi_model.py
git commit -m "feat: show aggregate MAPE/MAE/RMSE in no/partial-filter forecast view"
```

---

## Self-Review Notes

- **Spec coverage:** Feature 1 thresholds + last-pin (Task 1) ✓; wired into chart, points still plotted, None path untouched (Task 2) ✓; Feature 2 recompute-on-summed via ForecastMetrics, shown only when retrain guard passes, all-filters branch untouched (Task 3) ✓.
- **Placeholder scan:** none — every code step shows full code.
- **Type consistency:** `_thin_ticks(x_labels) -> (tickvals, ticktext)` defined in Task 1, consumed identically in Task 2. `ForecastMetrics.calculate_all` returns a dict keyed `'MAPE'/'MAE'/'RMSE'` (verified in `utils/metrics.py`), accessed with those keys in Task 3.

## Notes / Risks

- Stride thresholds are fixed constants per the spec (not configurable). If they ever need tuning, it is a one-line change in `_thin_ticks`.
- Task 3 line numbers are approximate (the file is ~1300 lines and shifts as edited); locate the insertion point by the `if test_result and len(aggregated_future_forecast) > 0:` guard and the `full_fitted = np.concatenate([...])` statement, not by line number.
- `calculate_all` also computes MSE/R²/SMAPE; we only display MAPE/MAE/RMSE. That is intended (matches the per-group cards in the all-filters branch); the extra keys are ignored.
