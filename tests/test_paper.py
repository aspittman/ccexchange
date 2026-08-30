from datetime import datetime, timezone

import pandas as pd

from ccexchange.execution import AccountSnapshot, FilledOrder, OrderUpdate
from ccexchange.paper import PaperRecorder


def test_paper_recorder_appends_bars_with_normalized_timestamps(tmp_path):
    recorder = PaperRecorder(tmp_path)
    initial = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"]),
    )
    initial.index.name = "timestamp"
    updated = pd.DataFrame(
        {"close": [102.0, 103.0]},
        index=pd.to_datetime(["2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z"]),
    )
    updated.index.name = "timestamp"

    recorder.record_bars({"BTC/USD": initial}, "1Day")
    recorder.record_bars({"BTC/USD": updated}, "1Day")

    saved = pd.read_csv(tmp_path / "bars" / "1Day" / "BTC_USD.csv", parse_dates=["timestamp"])
    assert saved["timestamp"].tolist() == list(pd.date_range("2025-01-01", periods=3, tz="UTC"))
    assert saved["close"].tolist() == [100.0, 102.0, 103.0]


def test_paper_recorder_deduplicates_fills_and_reports_round_trip(tmp_path):
    recorder = PaperRecorder(tmp_path)
    fills = [
        FilledOrder("buy-1", "BTC/USD", "buy", 1, 100, "2025-01-01T00:00:00+00:00"),
        FilledOrder("sell-1", "BTC/USD", "sell", 1, 110, "2025-01-02T00:00:00+00:00"),
    ]
    recorder.record_fills(fills)
    recorder.record_fills(fills)
    recorder.record_order_updates(
        [
            OrderUpdate(
                "buy-1", "ccexchange-buy", "BTC/USD", "buy", "filled", 1, 1,
                "2025-01-01T00:00:00+00:00", "2025-01-01T00:00:02+00:00",
            ),
            OrderUpdate(
                "rejected-1", "ccexchange-rejected", "ETH/USD", "buy", "rejected", 1, 0,
                "2025-01-01T00:00:00+00:00", "2025-01-01T00:00:01+00:00",
            ),
            OrderUpdate(
                "partial-1", "ccexchange-partial", "ETH/USD", "buy", "canceled", 2, 1,
                "2025-01-01T00:00:00+00:00", "2025-01-01T00:00:03+00:00",
            ),
        ]
    )
    experiment_id = recorder.register_experiment({"timeframe": "1Day", "adx": 14})
    recorder.record_order(
        {
            "order_id": "buy-1",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "symbol": "BTC/USD",
            "side": "buy",
            "experiment_id": experiment_id,
            "regime": "UPTREND",
            "timeframe": "1Day",
            "score_band": "80-89",
            "component_adx_strength": 12,
        }
    )
    recorder.record_order(
        {
            "order_id": "sell-1",
            "timestamp": "2025-01-02T00:00:00+00:00",
            "symbol": "BTC/USD",
            "side": "sell",
            "reason": "ATR stop",
        }
    )
    recorder.record_snapshot(
        datetime(2025, 1, 1, tzinfo=timezone.utc), AccountSnapshot(1000, 1000, {})
    )
    recorder.record_snapshot(
        datetime(2025, 1, 2, tzinfo=timezone.utc), AccountSnapshot(1010, 1010, {})
    )
    report = recorder.report()
    assert report["metrics"]["number_of_fills"] == 2
    assert report["metrics"]["number_of_round_trips"] == 1
    assert report["round_trips"][0]["pnl"] == 10
    assert report["technique_analysis"]["sample_size"] == 1
    assert report["technique_analysis"]["groups"]
    assert report["metrics"]["rejected_orders"] == 1
    assert report["metrics"]["canceled_orders"] == 1
    assert report["metrics"]["partially_filled_orders"] == 1
