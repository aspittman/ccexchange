from datetime import datetime, timezone

from ccexchange.config import load_config
from ccexchange.execution import FilledOrder
from ccexchange.runtime import _bot_performance, _risk_block
from ccexchange.state import BotState


def test_daily_loss_suspends_new_entries_but_records_expiry():
    cfg = load_config("config/default.yaml")
    state = BotState(peak_equity=100_000, day_start_equity=100_000, equity_day="2025-01-01")
    now = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
    assert _risk_block(state, 96_000, cfg, now) == "maximum daily portfolio loss"
    assert state.suspended_until


def test_bot_performance_uses_attributed_fills_and_marks_open_position():
    fills = [
        FilledOrder("1", "BTC/USD", "buy", 2.0, 100.0, "2025-01-01T00:00:00+00:00"),
        FilledOrder("2", "BTC/USD", "sell", 1.0, 120.0, "2025-01-02T00:00:00+00:00"),
    ]

    positions, pnl, return_percent = _bot_performance(fills, {"BTC/USD": 110.0}, 1_000.0)

    assert positions == 1
    assert pnl == 30.0
    assert return_percent == 3.0


def test_bot_performance_is_zero_without_bot_fills():
    assert _bot_performance([], {"BTC/USD": 999.0}, 100_000.0) == (0, 0.0, 0.0)
