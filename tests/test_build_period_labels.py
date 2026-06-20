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
