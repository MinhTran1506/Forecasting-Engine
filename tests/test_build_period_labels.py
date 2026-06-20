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


def test_group_filter_with_non_contiguous_index():
    # Simulates df_processed after negative-value filtering: rows are dropped
    # without reset_index, leaving an index with gaps. The group filter must
    # select rows by label (Product == "B") and stay correctly aligned despite
    # the gapped index.
    df = pd.DataFrame({
        "Period": [1, 2, 3, 4],
        "Year": [2024, 2024, 2024, 2024],
        "Month": [1, 2, 3, 4],
        "Product": ["A", "B", "A", "B"],
    })
    df = df.drop(index=[0, 2])  # index becomes [1, 3]; Product B rows -> Periods 2 and 4
    labels = build_period_labels(
        df, "Year", "Month", num_actual=2, num_forecast=0,
        group_dict={"Product": "B"})
    # Product B mapping: {2: 2024/2, 4: 2024/4}, first_period=2.
    # Period 1 (<= num_actual, not in mapping) -> backward extrapolation
    #   calculate_date_from_reference(2024, 2, offset=-1, "Month") -> "2024/1".
    # Period 2 (in mapping) -> "2024/2".
    assert labels == ["2024/1", "2024/2"]
