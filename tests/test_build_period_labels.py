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
