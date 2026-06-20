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
