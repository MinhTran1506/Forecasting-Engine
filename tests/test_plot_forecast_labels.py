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
