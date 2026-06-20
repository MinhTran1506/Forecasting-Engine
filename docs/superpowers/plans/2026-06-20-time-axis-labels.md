# Time-Based X-Axis Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace integer "Period" x-axis labels on the forecast charts with real `Year/Time` time labels (e.g. `2024/6`, `2024/Jun`), reusing the existing download date logic.

**Architecture:** Add a pure helper `build_period_labels` in `pages/multi_model.py` that converts the period range into label strings (reusing `calculate_date_from_reference`). Add an optional `x_labels` parameter to `Visualizer.plot_forecast` that relabels the existing numeric x-positions via Plotly tick arrays. Wire the helper into the three `plot_forecast` call sites and replace the duplicated inline date-mapping blocks in the download sections.

**Tech Stack:** Python 3.13, pandas, numpy, plotly, Streamlit; pytest for tests.

## Global Constraints

- Label format matches the existing download format exactly: `f"{year}/{time}"` — numeric time `2024/6`, text months `2024/Jun`. No zero-padding, no ISO week notation.
- Backward compatible: `plot_forecast(x_labels=None)` must behave identically to today (integer positions, "Period" axis title). `plot_model_comparison` and `plot_residuals` are untouched.
- `build_period_labels` is a pure function — no Streamlit (`st.*`) calls inside it.
- Chart geometry (numeric x-positions, fitted→forecast connection) is unchanged; only tick labelling and axis title change.
- Reuse the existing `calculate_date_from_reference` helper — do not write new date math.

---

### Task 0: Install pytest (one-time dev setup)

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add pytest to requirements**

Append to `requirements.txt`:

```
pytest>=8.0.0
```

- [ ] **Step 2: Install pytest into the project venv**

Run: `.venv/Scripts/pip.exe install "pytest>=8.0.0"`
Expected: ends with `Successfully installed pytest-...`

- [ ] **Step 3: Verify pytest runs**

Run: `.venv/Scripts/python.exe -m pytest --version`
Expected: prints `pytest 8.x.x`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add pytest for unit tests"
```

---

### Task 1: `build_period_labels` — actual periods present in mapping (no extrapolation)

**Files:**
- Modify: `pages/multi_model.py` (add function after `calculate_date_from_reference`, which ends at line 141)
- Test: `tests/test_build_period_labels.py` (create)

**Interfaces:**
- Consumes: `calculate_date_from_reference(ref_year, ref_time, periods_offset, time_col, is_text_based=False)` — already defined at `pages/multi_model.py:97`.
- Produces: `build_period_labels(df_processed, year_col, time_col, num_actual, num_forecast, group_dict=None) -> list[str] | None`. Returns a list of length `num_actual + num_forecast`, indexed positionally (index 0 = period 1). Returns `None` when the mapping cannot be built.

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_period_labels.py`:

```python
import pandas as pd
from pages.multi_model import build_period_labels


def _df(periods, years, times):
    return pd.DataFrame({
        "Period": periods,
        "Year": years,
        "Month": times,
    })


def test_numeric_months_present_in_mapping():
    # Periods 1..3 all present; no forecast periods.
    df = _df([1, 2, 3], [2024, 2024, 2024], [1, 2, 3])
    labels = build_period_labels(df, "Year", "Month", num_actual=3, num_forecast=0)
    assert labels == ["2024/1", "2024/2", "2024/3"]


def test_text_months_present_in_mapping():
    df = _df([1, 2, 3], [2024, 2024, 2024], ["Jan", "Feb", "Mar"])
    labels = build_period_labels(df, "Year", "Month", num_actual=3, num_forecast=0)
    assert labels == ["2024/Jan", "2024/Feb", "2024/Mar"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_build_period_labels.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_period_labels'`

- [ ] **Step 3: Write minimal implementation**

Add to `pages/multi_model.py` immediately after `calculate_date_from_reference` (after line 141):

```python
def build_period_labels(df_processed, year_col, time_col,
                        num_actual, num_forecast, group_dict=None):
    """
    Build a positional list of Year/Time labels for the chart x-axis and the
    download Date column.

    Returns a list of length (num_actual + num_forecast) where index i
    corresponds to period (i + 1). Periods found in df_processed use the exact
    "Year/Time" string; periods outside the known range are extrapolated with
    calculate_date_from_reference (backward for leading-history gaps, forward
    for forecast periods).

    Returns None when df_processed / year_col / time_col is missing or the
    filtered frame yields no period mapping, signalling the caller to fall back
    to integer-period labels.
    """
    if df_processed is None or not year_col or not time_col:
        return None
    if year_col not in df_processed.columns or time_col not in df_processed.columns:
        return None

    work = df_processed
    if group_dict:
        mask = pd.Series([True] * len(work), index=work.index)
        for col, val in group_dict.items():
            if col in work.columns:
                mask = mask & (work[col] == val)
        work = work[mask]

    if work.empty:
        return None

    # Build Period -> {Year, Time} mapping (numeric Time coerced to int,
    # text Time kept as-is), matching the existing download logic.
    period_to_date = {}
    unique_periods = work[["Period", year_col, time_col]].drop_duplicates().sort_values("Period")
    for _, row in unique_periods.iterrows():
        time_val = row[time_col]
        try:
            time_val = int(time_val)
        except (ValueError, TypeError):
            pass
        period_to_date[int(row["Period"])] = {
            "Year": int(row[year_col]),
            "Time": time_val,
        }

    if not period_to_date:
        return None

    first_period = min(period_to_date.keys())
    last_period = max(period_to_date.keys())
    first_year = period_to_date[first_period]["Year"]
    first_time = period_to_date[first_period]["Time"]
    last_year = period_to_date[last_period]["Year"]
    last_time = period_to_date[last_period]["Time"]
    is_text_based = not isinstance(first_time, int)

    labels = []
    total_periods = num_actual + num_forecast
    for i in range(total_periods):
        period = i + 1
        if period in period_to_date:
            year = period_to_date[period]["Year"]
            time = period_to_date[period]["Time"]
            labels.append(f"{year}/{time}")
        elif period <= num_actual:
            offset = period - first_period
            labels.append(calculate_date_from_reference(
                first_year, first_time, offset, time_col, is_text_based))
        else:
            offset = period - last_period
            labels.append(calculate_date_from_reference(
                last_year, last_time, offset, time_col, is_text_based))
    return labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_build_period_labels.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_build_period_labels.py pages/multi_model.py
git commit -m "feat: add build_period_labels helper for chart time labels"
```

---

### Task 2: `build_period_labels` — forward forecast extrapolation

**Files:**
- Modify: `tests/test_build_period_labels.py`

**Interfaces:**
- Consumes: `build_period_labels` from Task 1 (signature unchanged).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_period_labels.py`:

```python
def test_forward_extrapolation_numeric_month_rollover():
    # Actuals for 2024/11, 2024/12; forecast 3 periods -> Jan/Feb/Mar 2025.
    df = _df([1, 2], [2024, 2024], [11, 12])
    labels = build_period_labels(df, "Year", "Month", num_actual=2, num_forecast=3)
    assert labels == ["2024/11", "2024/12", "2025/1", "2025/2", "2025/3"]


def test_forward_extrapolation_text_month_rollover():
    df = _df([1, 2], [2024, 2024], ["Nov", "Dec"])
    labels = build_period_labels(df, "Year", "Month", num_actual=2, num_forecast=2)
    assert labels == ["2024/Nov", "2024/Dec", "2025/Jan", "2025/Feb"]


def test_forward_extrapolation_weekly_rollover():
    # time_col name contains "week" -> max_time 52.
    df = pd.DataFrame({"Period": [1, 2], "Year": [2024, 2024], "Week": [51, 52]})
    labels = build_period_labels(df, "Year", "Week", num_actual=2, num_forecast=2)
    assert labels == ["2024/51", "2024/52", "2025/1", "2025/2"]
```

- [ ] **Step 2: Run test to verify it passes**

These exercise existing `calculate_date_from_reference` paths through the Task 1 implementation; they should pass without code changes.

Run: `.venv/Scripts/python.exe -m pytest tests/test_build_period_labels.py -v`
Expected: PASS (5 passed)

- [ ] **Step 3: Commit**

```bash
git add tests/test_build_period_labels.py
git commit -m "test: cover forward forecast extrapolation in build_period_labels"
```

---

### Task 3: `build_period_labels` — backward gap, group filter, and None fallbacks

**Files:**
- Modify: `tests/test_build_period_labels.py`

**Interfaces:**
- Consumes: `build_period_labels` from Task 1.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_period_labels.py`:

```python
def test_backward_extrapolation_leading_gap():
    # Mapping starts at period 3 (2024/3); periods 1,2 extrapolate backward.
    df = _df([3, 4], [2024, 2024], [3, 4])
    labels = build_period_labels(df, "Year", "Month", num_actual=4, num_forecast=0)
    assert labels == ["2024/1", "2024/2", "2024/3", "2024/4"]


def test_group_filter_returns_only_that_group_timeline():
    df = pd.DataFrame({
        "Period": [1, 2, 1, 2],
        "Year": [2024, 2024, 2024, 2024],
        "Month": [1, 2, 6, 7],
        "Product": ["A", "A", "B", "B"],
    })
    labels = build_period_labels(
        df, "Year", "Month", num_actual=2, num_forecast=0,
        group_dict={"Product": "B"})
    assert labels == ["2024/6", "2024/7"]


def test_returns_none_when_df_missing():
    assert build_period_labels(None, "Year", "Month", 3, 0) is None


def test_returns_none_when_column_missing():
    df = _df([1, 2], [2024, 2024], [1, 2])
    assert build_period_labels(df, "Year", "Nonexistent", 2, 0) is None


def test_returns_none_when_group_filter_empties_frame():
    df = pd.DataFrame({
        "Period": [1], "Year": [2024], "Month": [1], "Product": ["A"],
    })
    result = build_period_labels(
        df, "Year", "Month", num_actual=1, num_forecast=0,
        group_dict={"Product": "Z"})
    assert result is None
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_build_period_labels.py -v`
Expected: PASS (10 passed)

- [ ] **Step 3: Commit**

```bash
git add tests/test_build_period_labels.py
git commit -m "test: cover backward gap, group filter, and None fallbacks"
```

---

### Task 4: Add `x_labels` parameter to `Visualizer.plot_forecast`

**Files:**
- Modify: `utils/visualizer.py:12` (signature) and `utils/visualizer.py:89-104` (layout block)
- Test: `tests/test_plot_forecast_labels.py` (create)

**Interfaces:**
- Produces: `Visualizer.plot_forecast(actual, fitted, forecast, title="Forecast", x_labels=None)`. When `x_labels` is a list, the returned Plotly figure's x-axis uses `tickmode='array'` with `tickvals`/`ticktext` derived from `x_labels`, and `xaxis.title.text == "Time"`. When `x_labels is None`, `xaxis.title.text == "Period"` and no tickvals are set.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plot_forecast_labels.py`:

```python
from utils.visualizer import Visualizer


def test_default_axis_is_period_with_no_tickvals():
    fig = Visualizer.plot_forecast(
        actual=[10, 12, 11], fitted=[10, 12, 11], forecast=[13, 14])
    assert fig.layout.xaxis.title.text == "Period"
    assert fig.layout.xaxis.tickvals is None


def test_labels_set_time_axis_and_ticktext():
    labels = ["2024/1", "2024/2", "2024/3", "2024/4", "2024/5"]
    fig = Visualizer.plot_forecast(
        actual=[10, 12, 11], fitted=[10, 12, 11], forecast=[13, 14],
        x_labels=labels)
    assert fig.layout.xaxis.title.text == "Time"
    # One tick per period (3 actual + 2 forecast = 5 positions, 0..4).
    assert list(fig.layout.xaxis.tickvals) == [0, 1, 2, 3, 4]
    assert list(fig.layout.xaxis.ticktext) == labels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_plot_forecast_labels.py -v`
Expected: FAIL — `test_labels_set_time_axis_and_ticktext` fails because `x_labels` is not accepted / axis title is "Period".

- [ ] **Step 3: Implement the parameter**

In `utils/visualizer.py`, change the signature at line 12 from:

```python
    def plot_forecast(actual, fitted, forecast, title="Forecast"):
```

to:

```python
    def plot_forecast(actual, fitted, forecast, title="Forecast", x_labels=None):
```

Then replace the `fig.update_layout(...)` block (lines 89-104) with:

```python
        xaxis_title = 'Period'
        if x_labels:
            xaxis_title = 'Time'

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title='Value',
            hovermode='x unified',
            height=500,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        if x_labels:
            tickvals = list(range(len(x_labels)))
            fig.update_xaxes(
                tickmode='array',
                tickvals=tickvals,
                ticktext=list(x_labels)
            )

        return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_plot_forecast_labels.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add utils/visualizer.py tests/test_plot_forecast_labels.py
git commit -m "feat: add optional x_labels time axis to plot_forecast"
```

---

### Task 5: Wire labels into the per-group chart call site

**Files:**
- Modify: `pages/multi_model.py:740-772` (per-group display block)

**Interfaces:**
- Consumes: `build_period_labels` (Task 1), `plot_forecast(..., x_labels=...)` (Task 4). `settings` dict holds `df_processed`, `year_col`, `time_col`, `forecast_periods` (set at `pages/multi_model.py:555-556` / `621-622`).

- [ ] **Step 1: Build labels and pass them at the per-group call**

In `pages/multi_model.py`, locate the per-group `viz.plot_forecast` call (around line 766). Immediately before it, add label construction; the `group_name` for this block is the loop variable already in scope.

Replace:

```python
                                # Plot with fitted line that connects to forecast
                                fig = viz.plot_forecast(
                                    result['full_data'],
                                    full_fitted,
                                    future_forecast,
                                    title=f"Forecast for {group_name}"
                                )
```

with:

```python
                                # Build Year/Time labels for this group's timeline
                                group_dict = {}
                                for part in group_name.split(', '):
                                    if '=' in part:
                                        col, val = part.split('=', 1)
                                        group_dict[col] = val
                                x_labels = build_period_labels(
                                    settings.get('df_processed'),
                                    settings.get('year_col'),
                                    settings.get('time_col'),
                                    len(result['full_data']),
                                    settings['forecast_periods'],
                                    group_dict=group_dict
                                )

                                # Plot with fitted line that connects to forecast
                                fig = viz.plot_forecast(
                                    result['full_data'],
                                    full_fitted,
                                    future_forecast,
                                    title=f"Forecast for {group_name}",
                                    x_labels=x_labels
                                )
```

- [ ] **Step 2: Manual smoke check**

Run: `.venv/Scripts/python.exe -m streamlit run app.py` (requires `FUTURE1_USERS` env var set; see CLAUDE.md). Upload a grouped CSV, run a filtered single-group forecast, and confirm the per-group chart x-axis shows `Year/Time` labels.
Expected: x-axis reads e.g. `2024/1 ... 2025/3` instead of `0..n`.

If the app cannot be launched in this environment, instead run a syntax check:
Run: `.venv/Scripts/python.exe -c "import ast; ast.parse(open('pages/multi_model.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pages/multi_model.py
git commit -m "feat: time labels on per-group forecast chart"
```

---

### Task 6: Wire labels into the aggregated and overall-no-groups chart call sites

**Files:**
- Modify: `pages/multi_model.py:865-870` (aggregated/overall-sum call)
- Modify: `pages/multi_model.py:1063-1068` (overall-no-groups call)

**Interfaces:**
- Consumes: `build_period_labels` (Task 1), `plot_forecast(..., x_labels=...)` (Task 4). Aggregated block has `aggregated_y` and `aggregated_future_forecast` in scope; overall block has `results['full_data']` and `future_forecast` in scope.

- [ ] **Step 1: Wire the aggregated/overall-sum call**

Replace (around line 865):

```python
                            fig = viz.plot_forecast(
                                aggregated_y,
                                full_fitted,
                                aggregated_future_forecast,
                                title=f"Overall Forecast (Sum of {len(matching_groups)} groups)"
                            )
```

with:

```python
                            x_labels = build_period_labels(
                                settings.get('df_processed'),
                                settings.get('year_col'),
                                settings.get('time_col'),
                                len(aggregated_y),
                                len(aggregated_future_forecast)
                            )
                            fig = viz.plot_forecast(
                                aggregated_y,
                                full_fitted,
                                aggregated_future_forecast,
                                title=f"Overall Forecast (Sum of {len(matching_groups)} groups)",
                                x_labels=x_labels
                            )
```

- [ ] **Step 2: Wire the overall-no-groups call**

Replace (around line 1063):

```python
                fig = viz.plot_forecast(
                    results['full_data'],
                    full_fitted,
                    future_forecast,
                    title=f"Future Forecast using {best_name}"
                )
```

with:

```python
                x_labels = build_period_labels(
                    settings.get('df_processed'),
                    settings.get('year_col'),
                    settings.get('time_col'),
                    len(results['full_data']),
                    settings['forecast_periods']
                )
                fig = viz.plot_forecast(
                    results['full_data'],
                    full_fitted,
                    future_forecast,
                    title=f"Future Forecast using {best_name}",
                    x_labels=x_labels
                )
```

- [ ] **Step 3: Syntax check**

Run: `.venv/Scripts/python.exe -c "import ast; ast.parse(open('pages/multi_model.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pages/multi_model.py
git commit -m "feat: time labels on aggregated and overall forecast charts"
```

---

### Task 7: Refactor download Date columns to reuse `build_period_labels`

**Files:**
- Modify: `pages/multi_model.py:884-987` (group-wise download loop)
- Modify: `pages/multi_model.py:1074-1146+` (overall download loop)

**Interfaces:**
- Consumes: `build_period_labels` (Task 1). Removes the duplicated inline `period_to_date` / `calculate_date_from_reference` blocks, sourcing the `Date` column from the same label list the charts use.

- [ ] **Step 1: Refactor the group-wise download loop**

In the group-wise download loop (starts ~line 887), the per-row `Date` is currently computed from an inline `period_to_date` map and extrapolation branches (lines ~899-972). Replace that per-group date computation so labels come from the helper. Within the `for group_name, result in group_results.items():` loop, after `group_dict` is parsed (lines ~893-897) and `num_actual` / `num_forecast` / `total_periods` are known (lines ~924-930), compute:

```python
                date_labels = build_period_labels(
                    df_processed, year_col, time_col,
                    num_actual, num_forecast, group_dict=group_dict
                )
```

Then in the `for i in range(total_periods):` row loop, replace the entire `Date` block (the `if period in period_to_date: ... else: row_data['Date'] = ''` branch, lines ~952-972) with:

```python
                    # Date column (Year/Time) from shared label builder
                    if date_labels is not None and i < len(date_labels):
                        row_data['Date'] = date_labels[i]
                    else:
                        row_data['Date'] = ''
```

Delete the now-unused inline `period_to_date` construction and the `first_year/first_time/last_year/...` extraction for this loop (lines ~899-944), since `build_period_labels` encapsulates them.

- [ ] **Step 2: Refactor the overall download loop**

In the overall (no-groups) download loop (starts ~line 1074), replace the inline `period_to_date` block (lines ~1079-1117) with:

```python
                num_actual = len(results['full_data'])
                num_forecast = settings['forecast_periods']
                total_periods = num_actual + num_forecast
                quantities = list(results['full_data']) + list(future_forecast)

                date_labels = build_period_labels(
                    df_processed, year_col, time_col, num_actual, num_forecast
                )
```

(keeping whichever of `num_actual`/`num_forecast`/`total_periods`/`quantities` assignments already exist — do not duplicate them). Then in that loop's `for i in range(total_periods):` body, replace the `Date` branch (lines ~1126-1146) with:

```python
                    # Date column (Year/Time) from shared label builder
                    if date_labels is not None and i < len(date_labels):
                        row_data['Date'] = date_labels[i]
                    else:
                        row_data['Date'] = ''
```

- [ ] **Step 3: Syntax check**

Run: `.venv/Scripts/python.exe -c "import ast; ast.parse(open('pages/multi_model.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Verify all unit tests still pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: PASS (12 passed across both test files)

- [ ] **Step 5: Manual parity check (if app launchable)**

Launch the app, run a grouped forecast, download the Excel, and confirm the `Date` column values match the chart x-axis labels for the same group.
Expected: identical `Year/Time` strings in chart ticks and the Excel `Date` column.

- [ ] **Step 6: Commit**

```bash
git add pages/multi_model.py
git commit -m "refactor: derive download Date column from build_period_labels"
```

---

## Self-Review Notes

- **Spec coverage:** Year/Time labels on chart (Tasks 4-6) ✓; format matches download (Task 1 uses `f"{year}/{time}"`) ✓; forward + backward extrapolation (Tasks 2-3) ✓; auto-thin ticks — Plotly default tick-skipping applies once `tickvals`/`ticktext` are set, no rotation forced (Task 4) ✓; None fallback to integer periods (Tasks 1, 3, 4) ✓; three call sites wired (Tasks 5-6) ✓; shared-helper refactor of download (Task 7) ✓; unit tests for all helper branches (Tasks 1-3) ✓.
- **Placeholder scan:** none — every code step shows full code.
- **Type consistency:** `build_period_labels(df_processed, year_col, time_col, num_actual, num_forecast, group_dict=None) -> list[str] | None` is used identically across Tasks 1, 5, 6, 7. `plot_forecast(..., x_labels=None)` consistent across Tasks 4, 5, 6.

## Notes / Risks

- **Auto-thin behavior:** Plotly with `tickmode='array'` shows every supplied tick. For very long series this can still crowd. If, during manual review, ticks overlap badly, the agreed design is "let Plotly auto-thin" — switch `tickmode='array'` to relying on `tickvals` only with `nticks` cap, OR drop to default `tickmode='auto'` with a custom hovertext. This is a tuning follow-up, not a blocker; raise it during code review if observed.
- The download refactor (Task 7) touches large inline blocks; the syntax check + full test run + manual parity check are the guard rails. Line numbers are approximate — locate by the surrounding code shown.
