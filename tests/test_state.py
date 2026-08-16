from ccexchange.state import BotState, ManagedPosition, StateStore


def test_state_round_trip_is_atomic(tmp_path):
    store = StateStore(tmp_path / "state" / "bot.json")
    state = BotState(
        positions={"BTC/USD": ManagedPosition("BTC/USD", 100, 1, 90, 110, "now")},
        last_processed_candle="candle",
        pending_orders={"ETH/USD": "buy"},
    )
    store.save(state)
    loaded = store.load()
    assert loaded.positions["BTC/USD"].stop == 90
    assert loaded.last_processed_candle == "candle"
    assert loaded.pending_orders == {"ETH/USD": "buy"}
