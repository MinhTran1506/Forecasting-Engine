# Smart Tick Density + Always-On Aggregate Metrics

**Date:** 2026-06-20
**Status:** Approved design, pending implementation plan

## Problem

Two usability issues in the Best-Fit Forecast page:

1. **Tick overcrowding.** `Visualizer.plot_forecast` forces every period onto the
   x-axis (`tickmode='array'` with a tickval per period). For series spanning
   many periods the `Year/Time` labels overlap and become unreadable.

2. **Metrics only on full filter.** In the Interactive Forecast Explorer, the
   per-group accuracy metrics (Best Model, MAPE, MAE, RMSE) appear only when
   every group column is narrowed to a specific value (`all_filters_selected`).
   With no filter or a partial filter, the aggregated/summed view shows only
   "Groups Summed" and "Models Aggregated" — no accuracy figure.

## Requirements

### Feature 1 — Smart tick density

- The forecast chart shows at most a readable number of x-axis labels, thinning
  them as the series grows, while still plotting every data point and keeping
  every point's exact label available on hover.
- Thinning is by stride, tiered on total period count `n` (`n = len(x_labels)`):
  - `n <= 24` → stride 1 (show all)
  - `25 <= n <= 60` → stride 3 (every 3rd)
  - `n > 60` → stride 6 (every 6th)
- The **last** period is always labelled (pin it even if it does not fall on the
  stride), so the forecast end is identified.
- The `x_labels=None` path (integer "Period" axis) is unchanged.
- `plot_model_comparison` and `plot_residuals` are unaffected.

### Feature 2 — Headline aggregate metrics in the summed view

- In the aggregated/partial-filter branch of the Interactive Forecast Explorer,
  display headline MAPE/MAE/RMSE for the summed forecast as metric cards next to
  the existing "Groups Summed" / "Models Aggregated" cards.
- The headline metrics are **recomputed on the summed test series** — using the
  `test_data_agg` (summed test actuals) and `test_predictions` (summed model
  predictions) the branch already computes to draw the fitted line — via the
  existing `ForecastMetrics` static methods. This is computed on the exact
  aggregated curve being plotted, so it correctly reflects cross-group error
  cancellation (a volume-weighted average of per-group MAPEs would overstate
  error and is explicitly NOT used).
- If the aggregate retrain fails (the existing `if test_result` guard is false),
  the headline metrics are omitted; the rest of the view still renders.
- The `all_filters_selected` branch already shows per-group metrics and is
  unchanged.

## Architecture

### Feature 1 (`utils/visualizer.py`)

New pure static method:

```
Visualizer._thin_ticks(x_labels) -> (tickvals, ticktext)
```

- `n = len(x_labels)`. Choose stride: `1` if `n <= 24`, `3` if `n <= 60`, else
  `6`.
- `tickvals = list(range(0, n, stride))`; if `n - 1` is not already the last
  entry, append `n - 1` (pin the last label). `ticktext = [x_labels[i] for i in
  tickvals]`.
- Pure (no plotting, no Streamlit) → unit-testable.

`plot_forecast` change: the block currently at lines ~110-116 (which builds
`tickvals = list(range(len(x_labels)))`) instead calls
`tickvals, ticktext = Visualizer._thin_ticks(x_labels)` and passes both to
`fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext)`. No
other change to the function; the `if x_labels:` guard and the `None` path stay.

### Feature 2 (`pages/multi_model.py`)

In the aggregated/partial-filter branch, inside the existing
`if test_result and len(aggregated_future_forecast) > 0:` block (where
`test_data_agg` and `test_predictions` already exist), compute:

```
from utils.metrics import ForecastMetrics  # already imported at module top
agg_mape = ForecastMetrics.mape(test_data_agg, test_predictions)
agg_mae  = ForecastMetrics.mae(test_data_agg, test_predictions)
agg_rmse = ForecastMetrics.rmse(test_data_agg, test_predictions)
```

Render these as three additional `st.metric` cards. The existing two cards
("Groups Summed", "Models Aggregated") are created earlier at lines ~888-890 with
`st.columns(2)`; widen that to `st.columns(5)` (or a second `st.columns(3)` row
beneath) so the three new cards sit alongside. Exact layout decided in the plan;
the requirement is that all five values are visible together.

`test_predictions` in this branch is already `np.maximum(test_result['predictions'], 0)`
(non-negative), matching what is plotted.

## Testing

### Feature 1 — `Visualizer._thin_ticks` (pure)

- `n = 12` (<=24) → stride 1, all 12 ticks, tickvals `0..11`.
- `n = 24` (boundary) → stride 1, all ticks.
- `n = 25` → stride 3, tickvals `[0,3,6,...,24]` (24 already last; no dup).
- `n = 40` → stride 3, last value `39` appended (39 % 3 != 0).
- `n = 60` (boundary) → stride 3.
- `n = 61` → stride 6.
- `n = 72` → stride 6, tickvals `[0,6,...,66]` + `71` pinned.
- Each case: `ticktext` matches `x_labels` at the chosen `tickvals`; last label
  always present; no duplicate final index.

### Feature 2

- The aggregate metric computation is a thin reuse of `ForecastMetrics` on arrays
  that already exist; covered by `ForecastMetrics`' own behavior. Manual check:
  run the app, leave filters at "All", confirm Agg MAPE/MAE/RMSE cards appear and
  are plausible; apply a partial filter, confirm they update; the
  all-filters-selected path still shows per-group metrics unchanged.

## Out of scope

- Per-group breakdown table in the aggregated view (considered, then dropped).
- Volume-weighted aggregate metrics (rejected in favor of recompute-on-summed).
- Changing the tick thresholds at runtime / making them configurable.
- Any change to `plot_model_comparison`, `plot_residuals`, or the
  `all_filters_selected` branch.
