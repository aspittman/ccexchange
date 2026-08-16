import pandas as pd

from ccexchange.data import completed_only


def test_future_candle_is_excluded():
    idx = pd.to_datetime(["2025-01-01T00:00Z", "2025-01-02T00:00Z"])
    frame = pd.DataFrame({"close": [1, 2]}, index=idx)
    result = completed_only(frame, pd.Timestamp("2025-01-01T12:00Z"))
    assert list(result.close) == [1]
