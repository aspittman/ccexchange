from datetime import datetime, timezone

from ccexchange.config import load_config
from ccexchange.runtime import _risk_block
from ccexchange.state import BotState


def test_daily_loss_suspends_new_entries_but_records_expiry():
    cfg = load_config("config/default.yaml")
    state = BotState(peak_equity=100_000, day_start_equity=100_000, equity_day="2025-01-01")
    now = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
    assert _risk_block(state, 96_000, cfg, now) == "maximum daily portfolio loss"
    assert state.suspended_until
