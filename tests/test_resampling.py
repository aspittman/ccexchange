import pandas as pd

from ccexchange.data import resample_completed_hourly


def test_four_hour_ohlcv_aggregation():
    idx = pd.date_range("2025-01-01T01:00Z", periods=4, freq="h")
    frame = pd.DataFrame(
        {
            "open": [1, 2, 3, 4],
            "high": [2, 3, 4, 5],
            "low": [0, 1, 2, 3],
            "close": [1.5, 2.5, 3.5, 4.5],
            "volume": [1] * 4,
        },
        index=idx,
    )
    result = resample_completed_hourly(frame, 4)
    assert result.iloc[-1].open == 1
    assert result.iloc[-1].close == 4.5
    assert result.iloc[-1].volume == 4
