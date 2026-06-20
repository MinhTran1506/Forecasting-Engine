# Time-Based X-Axis Labels for Forecast Charts

**Date:** 2026-06-20
**Status:** Approved design, pending implementation plan

## Problem

The forecast chart (`Visualizer.plot_forecast`) currently labels its x-axis with
sequential integer periods (0..n) under the title "Period". Users forecasting by
month or week cannot tell which calendar period each point corresponds to. They
want real time labels instead — e.g. `2024/6` for a numeric June, `2024/Jun` for
a text-based June, and the analogous `Year/Week` form for weekly data.

The Excel download already solves the underlying period→date conversion (via an
inline `period_to_date` mapping plus `calculate_date_from_reference`). This
feature reuses that logic so the chart and the export stay consistent.

## Requirements

- The forecast chart x-axis shows `Year/Time` labels instead of integer periods.
- Label format **matches the existing download format** exactly:
  - Numeric time: `2024/6`, `2024/12`, `2025/1`
  - Text months: `2024/Jun`, `2024/Dec`
  - Weekly numeric: `2024/5`, etc.
- Labels cover both historical (actual) periods and forecast periods, with
  forecast labels extrapolated forward from the last known period (month/week
  rollover with year increment).
- Missing leading history periods are extrapolated backward from the first known
  period (preserving today's download behavior).
- Long series do not overcrowd the axis: all points are still plotted, but
  Plotly auto-thins the visible ticks (no forced rotation). Hover shows the exact
  label for every point.
- Backward compatible: if the time mapping cannot be built (missing
  `df_processed`, `year_col`, or `time_col`), the chart falls back to today's
  integer-period behavior with the "Period" axis title.
- `plot_model_comparison` and any other existing caller are unaffected.

## Architecture

### New shared helper (`pages/multi_model.py`)

```
build_period_labels(df_processed, year_col, time_col,
                    num_actual, num_forecast, group_dict=None) -> list[str] | None
```

- Filters `df_processed` by `group_dict` when provided (per-group chart);
  otherwise uses all rows (aggregated and overall charts).
- Builds the `period_to_date` mapping the same way the current download code does
  (unique Period → Year, Time; numeric Time coerced to int, text Time kept as
  string).
- Returns a list of `num_actual + num_forecast` label strings, indexed
  positionally (index 0 = period 1). Periods present in the mapping use
  `f"{year}/{time}"`; periods outside it use `calculate_date_from_reference`
  (backward extrapolation for leading history gaps, forward extrapolation for
  forecast periods).
- Returns `None` when `df_processed`, `year_col`, or `time_col` is missing/empty,
  signalling the chart to fall back to integer periods.
- Pure function (no Streamlit calls) → independently unit-testable.

### Chart change (`utils/visualizer.py`)

`plot_forecast` gains one optional parameter:

```
plot_forecast(actual, fitted, forecast, title="Forecast", x_labels=None)
```

- The numeric x-position geometry is unchanged. The fitted→forecast connection
  still relies on the shared numeric position
  (`forecast_start_x = len(actual) - 1`), so the dotted fitted line continues to
  visually join the dashed forecast line.
- When `x_labels` is provided, map integer positions to labels with
  `fig.update_xaxes(tickmode='array', tickvals=..., ticktext=...)`, let Plotly
  auto-thin ticks, and set the axis title to "Time".
- When `x_labels is None`, behavior is identical to today (integer positions,
  "Period" title).
- `hovermode='x unified'` is already set, so tick labels carry into the hover
  readout automatically.

## Call sites (all in `pages/multi_model.py`)

1. **Per-group chart** (~line 766): labels built with `group_dict` parsed from
   `group_name` (`split(', ')` → `col=val`), length
   `len(result['full_data']) + settings['forecast_periods']`.
2. **Aggregated/overall-sum chart** (~line 865): labels built with no
   `group_dict` (full `df_processed` range), length
   `len(aggregated_y) + len(aggregated_future_forecast)`. All groups share the
   same Year/Time periods, so the unfiltered mapping is the correct timeline.
3. **Overall-no-groups chart** (~line 1063): labels built with no `group_dict`,
   length `num_actual + num_forecast`.

Each call becomes `viz.plot_forecast(..., x_labels=labels)`, with `labels` being
`None` on the fallback path.

### Refactor cleanup (in scope)

The inline `period_to_date` + extrapolation blocks in the download sections are
replaced by `build_period_labels`, removing duplication. The download `Date`
column is derived from the same label list, keeping chart and export identical.

## Testing

Unit tests for `build_period_labels` (pure function):

- Numeric months → `2024/1`; text months → `2024/Jun`; weekly → `2024/5`.
- Forward extrapolation into forecast periods, including month rollover
  (12 → 1 with year increment) and week rollover (52 → 1).
- Backward extrapolation for missing leading history periods.
- Group filtering returns only that group's timeline.
- Missing `df_processed`/columns → returns `None`.

Manual check: `streamlit run app.py`, upload sample data, confirm all three chart
variants show `Year/Time` labels and the Excel `Date` column still matches.

## Out of scope

- Changing the label format itself (zero-padding, ISO week notation).
- Any change to `plot_model_comparison` or `plot_residuals`.
- Date handling for the hidden tabs (`auto_future`, `holt_winters`, `home`).
